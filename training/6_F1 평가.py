"""
F1 스코어 평가 스크립트
단일 도메인 평가 후 종료
- 실행 흐름
1. 체크포인트 선택
2. 도메인 이름 입력 (예: "등록")
3. val.jsonl 파일 선택
4. 배치 크기, 디바이스 설정
5. 평가 진행
6. 결과 출력
7. CSV,TXT 파일 저장
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from datetime import datetime


class DomainDataset(Dataset):
    """도메인별 데이터셋 - 평가용"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = Path(data_path)

        # 7개 구두점
        self.punctuations = [
            ',', '。', '·', '?', '!', '《', '》'
        ]
        self.num_labels = 7

        # 데이터 로드 (메모리 효율적)
        print(f"데이터 로딩: {data_path}")
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                try:
                    sample = json.loads(line.strip())
                    self.samples.append(sample)
                except:
                    continue

        print(f"로드 완료: {len(self.samples)}개 샘플")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text = sample['c']
        length = sample['n']

        # 라벨 복원
        labels = []
        for indices in sample['l']:
            label_vec = [0] * self.num_labels
            for idx_val in indices:
                if 0 <= idx_val < self.num_labels:
                    label_vec[idx_val] = 1
            labels.append(label_vec)

        # 패딩
        while len(labels) < self.max_length:
            labels.append([0] * self.num_labels)
        labels = labels[:self.max_length]

        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 라벨 정렬
        aligned_labels = self._align_labels(text, labels, length, encoding)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.float),
            'length': length
        }

    def _align_labels(self, text: str, labels: List[List[int]], length: int,
                      encoding) -> List[List[float]]:
        """토큰과 라벨 정렬"""
        aligned_labels = [[0.0] * self.num_labels for _ in range(self.max_length)]
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        char_idx = 0
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token == '[UNK]' or not token.startswith('##'):
                if char_idx < length and char_idx < len(labels):
                    aligned_labels[i] = [float(x) for x in labels[char_idx]]
                    char_idx += 1
                elif char_idx >= length:
                    break

        return aligned_labels


class DomainEvaluator:
    """도메인별 평가기"""

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 학습된 체크포인트 경로
            device: 디바이스 설정
        """
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"디바이스: {self.device}")

        # 7개 구두점
        self.punctuations = [
            ',', '。', '·', '?', '!', '《', '》'
        ]
        self.num_labels = 7

        # 모델 로드
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """모델 로드"""
        print(f"모델 로딩: {checkpoint_path}")

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 하이퍼파라미터
        hparams = checkpoint['hyper_parameters']
        self.model_name = hparams['model_name']
        self.threshold = hparams.get('threshold', 0.5)

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 모델 구성
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(hparams.get('dropout_rate', 0.1))
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        # state_dict 로드
        state_dict = checkpoint['state_dict']
        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_state_dict[key] = value
            elif key.startswith('classifier.'):
                new_state_dict[key] = value

        # 가중치 로드
        self.bert.load_state_dict({k[5:]: v for k, v in new_state_dict.items()
                                   if k.startswith('bert.')})
        self.classifier.load_state_dict({k[11:]: v for k, v in new_state_dict.items()
                                         if k.startswith('classifier.')})

        # GPU로 이동 및 평가 모드
        self.bert = self.bert.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.bert.eval()

        print("모델 로딩 완료!")

    def evaluate_domain(self, data_loader: DataLoader, domain_name: str) -> Dict:
        """단일 도메인 평가"""
        print(f"\n도메인 평가 중: {domain_name}")

        # 메트릭 초기화
        metrics = {
            'tp': torch.zeros(self.num_labels, device=self.device),
            'fp': torch.zeros(self.num_labels, device=self.device),
            'fn': torch.zeros(self.num_labels, device=self.device),
            'total_samples': 0,
            'total_chars': 0
        }

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"평가 중 - {domain_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['length']

                # Forward
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = self.dropout(outputs.last_hidden_state)
                logits = self.classifier(sequence_output)

                # 예측
                preds = torch.sigmoid(logits)
                preds_binary = (preds > self.threshold).float()

                # 마스크 적용
                valid_mask = attention_mask.unsqueeze(-1).bool()
                preds_masked = preds_binary * valid_mask
                labels_masked = labels * valid_mask

                # 메트릭 계산
                for i in range(self.num_labels):
                    metrics['tp'][i] += ((preds_masked[:, :, i] == 1) &
                                         (labels_masked[:, :, i] == 1)).sum()
                    metrics['fp'][i] += ((preds_masked[:, :, i] == 1) &
                                         (labels_masked[:, :, i] == 0)).sum()
                    metrics['fn'][i] += ((preds_masked[:, :, i] == 0) &
                                         (labels_masked[:, :, i] == 1)).sum()

                metrics['total_samples'] += valid_mask.sum()
                metrics['total_chars'] += lengths.sum().item()

        # F1 계산
        results = self._calculate_metrics(metrics)
        results['domain'] = domain_name
        results['total_samples'] = metrics['total_samples'].item()
        results['total_chars'] = metrics['total_chars']

        return results

    def _calculate_metrics(self, metrics: Dict) -> Dict:
        """메트릭 계산"""
        results = {}

        # 전체 F1
        tp_sum = metrics['tp'].sum().item()
        fp_sum = metrics['fp'].sum().item()
        fn_sum = metrics['fn'].sum().item()

        overall_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
        overall_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
        overall_f1 = (2 * overall_precision * overall_recall /
                      (overall_precision + overall_recall)
                      if (overall_precision + overall_recall) > 0 else 0)

        results['overall_f1'] = overall_f1
        results['overall_precision'] = overall_precision
        results['overall_recall'] = overall_recall

        # 구두점별 F1
        punct_metrics = {}
        for i, punct in enumerate(self.punctuations):
            tp = metrics['tp'][i].item()
            fp = metrics['fp'][i].item()
            fn = metrics['fn'][i].item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            punct_metrics[punct] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }

        results['punct_metrics'] = punct_metrics

        return results

    def _print_domain_results(self, results: Dict):
        """도메인 결과 출력"""
        print(f"\n{'=' * 60}")
        print(f"도메인: {results['domain']}")
        print(f"{'=' * 60}")
        print(f"전체 F1 스코어: {results['overall_f1']:.4f}")
        print(f"Precision: {results['overall_precision']:.4f}")
        print(f"Recall: {results['overall_recall']:.4f}")
        print(f"샘플 수: {results['total_samples']:,}")
        print(f"문자 수: {results['total_chars']:,}")

        print(f"\n구두점별 F1 스코어:")
        for punct, metrics in sorted(results['punct_metrics'].items(),
                                     key=lambda x: x[1]['f1'], reverse=True):
            print(f"  {punct}: F1={metrics['f1']:.4f}, "
                  f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                  f"Support={metrics['support']:.0f}")


def main():
    """메인 함수 - 단일 도메인 평가"""
    print("=" * 60)
    print("도메인별 F1 스코어 평가")
    print("=" * 60)

    # 파일 선택
    root = tk.Tk()
    root.withdraw()

    # 1. 체크포인트 선택
    print("\n1. 평가할 체크포인트를 선택하세요...")
    checkpoint_path = filedialog.askopenfilename(
        title="체크포인트 파일 선택",
        filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
    )

    if not checkpoint_path:
        print("체크포인트가 선택되지 않았습니다.")
        return

    print(f"선택된 체크포인트: {checkpoint_path}")

    # 2. 도메인 이름 입력
    domain_name = input("\n도메인 이름 입력: ").strip()
    if not domain_name:
        print("도메인 이름이 입력되지 않았습니다.")
        return

    # 3. val.jsonl 파일 선택
    print(f"\n{domain_name}의 val.jsonl 파일을 선택하세요...")
    val_file = filedialog.askopenfilename(
        title=f"{domain_name} - val.jsonl 선택",
        filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
    )

    if not val_file:
        print("파일이 선택되지 않았습니다.")
        return

    print(f"선택된 파일: {val_file}")

    # 4. 설정
    batch_size = int(input("\n배치 크기 (기본: 32): ") or "32")
    device = input("디바이스 (auto/cpu/cuda) [auto]: ") or "auto"

    # 5. 평가 실행
    print("\n평가를 시작합니다...")
    print("=" * 60)

    try:
        # 평가기 생성
        evaluator = DomainEvaluator(checkpoint_path, device)

        # 데이터셋 생성
        dataset = DomainDataset(val_file, evaluator.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 평가
        results = evaluator.evaluate_domain(data_loader, domain_name)

        # 결과 출력
        evaluator._print_domain_results(results)

        # 6. CSV 저장
        print("\n결과를 CSV로 저장합니다...")

        # DataFrame 생성
        row = {
            'Domain': results['domain'],
            'F1_Score': results['overall_f1'],
            'Precision': results['overall_precision'],
            'Recall': results['overall_recall'],
            'Samples': results['total_samples'],
            'Characters': results['total_chars']
        }

        # 구두점별 F1 추가
        for punct, metrics in results['punct_metrics'].items():
            row[f'F1_{punct}'] = metrics['f1']
            row[f'P_{punct}'] = metrics['precision']
            row[f'R_{punct}'] = metrics['recall']

        df = pd.DataFrame([row])

        # 저장
        save_path = filedialog.asksaveasfilename(
            title="결과 저장",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{domain_name}_evaluation.csv"
        )

        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"CSV 저장 완료: {save_path}")

            # 텍스트 요약도 저장
            summary_path = Path(save_path).with_suffix('.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"도메인 평가 결과: {domain_name}\n")
                f.write(f"체크포인트: {checkpoint_path}\n")
                f.write(f"평가 파일: {val_file}\n")
                f.write(f"평가 시간: {datetime.now()}\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"전체 F1 스코어: {results['overall_f1']:.4f}\n")
                f.write(f"Precision: {results['overall_precision']:.4f}\n")
                f.write(f"Recall: {results['overall_recall']:.4f}\n")
                f.write(f"샘플 수: {results['total_samples']:,}\n")
                f.write(f"문자 수: {results['total_chars']:,}\n\n")

                f.write("구두점별 F1 스코어:\n")
                for punct, metrics in sorted(results['punct_metrics'].items(),
                                             key=lambda x: x[1]['f1'], reverse=True):
                    f.write(f"  {punct}: F1={metrics['f1']:.4f}, "
                            f"P={metrics['precision']:.4f}, "
                            f"R={metrics['recall']:.4f}\n")

            print(f"텍스트 요약 저장 완료: {summary_path}")

        print("\n평가 완료!")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()