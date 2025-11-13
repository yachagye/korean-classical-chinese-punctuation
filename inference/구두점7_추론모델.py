"""
한국 고전한문 구두점 지정 모델
학습된 모델을 사용하여 실제 텍스트에 구두점을 추가하는 추론 모델
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict
import argparse
from tqdm import tqdm
import re


class PunctuationPredictor:
    """한국 고전한문 구두점 예측기"""

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 학습된 체크포인트 경로
            device: 'cuda', 'cpu', 또는 'auto'
        """
        self.checkpoint_path = checkpoint_path

        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"디바이스: {self.device}")

        # 7개 구두점 (학습 시와 동일)
        self.punctuations = [
            ',', '。', '·', '?', '!', '《', '》'
        ]

        # 모델 로드
        self._load_model()

    def _load_model(self):
        """체크포인트에서 모델 로드"""
        print(f"모델 로딩: {self.checkpoint_path}")

        # Lightning 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # 하이퍼파라미터 추출
        hparams = checkpoint['hyper_parameters']
        self.model_name = hparams['model_name']
        self.num_labels = hparams['num_labels']
        self.threshold = hparams.get('threshold', 0.5)

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 모델 재구성
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(hparams.get('dropout_rate', 0.1))
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        # state_dict 로드
        state_dict = checkpoint['state_dict']

        # Lightning 모듈의 state_dict를 일반 PyTorch 모델로 변환
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key
            elif key.startswith('classifier.'):
                new_key = key
            else:
                continue
            new_state_dict[new_key] = value

        # 모델에 가중치 로드
        self.bert.load_state_dict({k[5:]: v for k, v in new_state_dict.items() if k.startswith('bert.')})
        self.classifier.load_state_dict({k[11:]: v for k, v in new_state_dict.items() if k.startswith('classifier.')})

        # 모델을 디바이스로 이동 및 평가 모드
        self.bert = self.bert.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.bert.eval()

        print("모델 로딩 완료!")

    def predict(self, text: str, batch_size: int = 1) -> str:
        """
        텍스트에 구두점 추가

        Args:
            text: 구두점을 추가할 텍스트
            batch_size: 배치 크기 (긴 텍스트 처리용)

        Returns:
            구두점이 추가된 텍스트
        """
        if not text:
            return text

        # 긴 텍스트는 청크로 분할 처리
        if len(text) > 400:
            return self._predict_long_text(text, batch_size)

        # 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.bert(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            sequence_output = self.dropout(outputs.last_hidden_state)
            logits = self.classifier(sequence_output)
            predictions = torch.sigmoid(logits)

        # 예측값을 numpy로 변환
        preds = predictions[0].cpu().numpy()

        # 이진화
        preds_binary = (preds > self.threshold).astype(int)

        # 후처리
        preds_processed = self._post_process(preds_binary, preds)

        # 결과 생성
        result = self._build_result(text, inputs, preds_processed)

        return result

    def _predict_long_text(self, text: str, batch_size: int) -> str:
        """긴 텍스트 처리 (청크 단위)"""
        # 문장 단위로 분할 (기본 구두점 기준)
        sentences = self._split_sentences(text)

        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_results = []

            for sent in batch:
                if sent.strip():
                    pred = self.predict(sent, batch_size=1)
                    batch_results.append(pred)
                else:
                    batch_results.append(sent)

            results.extend(batch_results)

        # 결과 병합
        return ''.join(results)

    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        # 기존 구두점이 있으면 그것으로 분할
        if any(p in text for p in ['。', '?', '!']):
            pattern = r'([。?!])'
            parts = re.split(pattern, text)
            sentences = []
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    sentences.append(parts[i] + parts[i + 1])
                else:
                    sentences.append(parts[i])
            return sentences

        # 없으면 청크 크기로 분할
        chunk_size = 400
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def _post_process(self, predictions: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """예측 결과 후처리 - 신뢰도 기반"""
        processed = predictions.copy()

        invalid_pairs = [
            ('。', '?'), ('。', '!'), ('?', '!'),
        ]

        for punct1, punct2 in invalid_pairs:
            if punct1 in self.punctuations and punct2 in self.punctuations:
                idx1 = self.punctuations.index(punct1)
                idx2 = self.punctuations.index(punct2)

                both_mask = (processed[:, idx1] == 1) & (processed[:, idx2] == 1)
                if both_mask.any():
                    # 신뢰도 비교
                    for i in np.where(both_mask)[0]:
                        if scores[i, idx1] >= scores[i, idx2]:
                            processed[i, idx2] = 0
                        else:
                            processed[i, idx1] = 0

        return processed

    def _build_result(self, text: str, inputs: Dict, predictions: np.ndarray) -> str:
        """토큰 예측을 원문에 적용하여 결과 생성"""
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        result = ""
        char_idx = 0

        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token == '[UNK]' or not token.startswith('##'):
                if char_idx < len(text):
                    result += text[char_idx]

                    # 예측된 구두점 추가
                    if i < len(predictions):
                        for idx in range(self.num_labels):
                            if predictions[i][idx] == 1:
                                result += self.punctuations[idx]

                    char_idx += 1

        # 남은 문자 처리
        if char_idx < len(text):
            result += text[char_idx:]

        return result

    def predict_file(self, input_path: str, output_path: str,
                     encoding: str = 'utf-8', batch_size: int = 1):
        """파일 단위 처리"""
        print(f"입력 파일: {input_path}")
        print(f"출력 파일: {output_path}")

        # 입력 파일 읽기
        with open(input_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        # 진행률 표시와 함께 처리
        results = []
        for line in tqdm(lines, desc="구두점 추가 중"):
            line = line.strip()
            if line:
                result = self.predict(line, batch_size=batch_size)
                results.append(result)
            else:
                results.append('')

        # 결과 저장
        with open(output_path, 'w', encoding=encoding) as f:
            for result in results:
                f.write(result + '\n')

        print(f"완료! 결과가 {output_path}에 저장되었습니다.")

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """배치 단위 예측"""
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="배치 처리 중"):
            batch = texts[i:i + batch_size]
            batch_results = []

            for text in batch:
                if text.strip():
                    result = self.predict(text)
                    batch_results.append(result)
                else:
                    batch_results.append(text)

            results.extend(batch_results)

        return results

    def update_threshold(self, new_threshold: float):
        """예측 임계값 업데이트"""
        self.threshold = new_threshold
        print(f"임계값이 {new_threshold}로 업데이트되었습니다.")

def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(description='한국 고전한문 구두점 예측')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='학습된 체크포인트 경로')
    parser.add_argument('--input', type=str, help='입력 파일 경로')
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--text', type=str, help='직접 입력 텍스트')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='배치 크기')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='디바이스')
    parser.add_argument('--encoding', type=str, default='utf-8',
                        help='파일 인코딩')

    args = parser.parse_args()

    # 예측기 초기화
    predictor = PunctuationPredictor(args.checkpoint, device=args.device)

    # 모드에 따라 처리
    if args.text:
        # 직접 입력 텍스트 처리
        result = predictor.predict(args.text, batch_size=args.batch_size)
        print(f"\n원문: {args.text}")
        print(f"결과: {result}")

    elif args.input and args.output:
        # 파일 처리
        predictor.predict_file(
            args.input,
            args.output,
            encoding=args.encoding,
            batch_size=args.batch_size
        )

    else:
        # 대화형 모드
        print("\n대화형 모드 (종료: 'quit' 또는 'exit')")
        print("-" * 50)

        while True:
            text = input("\n텍스트 입력: ").strip()

            if text.lower() in ['quit', 'exit']:
                break

            if text:
                result = predictor.predict(text)
                print(f"결과: {result}")


if __name__ == "__main__":
    main()