"""
고전 한문 구두점 예측 학습데이터 생성
곡선 따옴표 처리 수정 버전
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


class PunctuationDataGenerator:
    """한문 구두점 예측 데이터 생성기"""

    def __init__(self):
        # 7개 구두점 (전처리 분석 결과)
        self.punctuations = [
            ',',  # 쉼표
            '。',  # 마침표
            '·',  # 중간점
            '?',  # 물음표
            '!',  # 느낌표
            '《',  # 여는 서명
            '》'  # 닫는 서명
        ]

        self.punct_to_idx = {p: i for i, p in enumerate(self.punctuations)}
        self.num_punctuations = len(self.punctuations)

    def is_chinese_char(self, char):
        """한자 판별"""
        code = ord(char)
        return (0x4E00 <= code <= 0x9FFF or  # CJK Unified
                0x3400 <= code <= 0x4DBF or  # Extension A
                0x20000 <= code <= 0x2A6DF or  # Extension B
                0x2A700 <= code <= 0x2B73F or  # Extension C
                0x2B740 <= code <= 0x2B81F or  # Extension D
                0x2B820 <= code <= 0x2CEAF or  # Extension E
                0x2CEB0 <= code <= 0x2EBEF or  # Extension F
                0x30000 <= code <= 0x3134F or  # Extension G
                0x31350 <= code <= 0x323AF or  # Extension H
                0x2F800 <= code <= 0x2FA1F or  # Compatibility Supplement
                0xFA00 <= code <= 0xFA6F or  # CJK Compatibility Ideographs
                0xFA70 <= code <= 0xFADF or  # CJK Compatibility Ideographs
                0xF900 <= code <= 0xFAFF or  # Compatibility
                0x2F00 <= code <= 0x2FDF or  # Kangxi Radicals
                0x2E80 <= code <= 0x2EFF)  # Radicals Supplement

    def create_labels(self, text):
        """텍스트에서 한자와 구두점 라벨 추출"""
        if not text:
            return [], []

        chars = []
        labels = []

        i = 0
        while i < len(text):
            if self.is_chinese_char(text[i]):
                chars.append(text[i])
                punct_indices = []

                # 현재 한자 다음 위치부터 탐색
                j = i + 1
                while j < len(text) and not self.is_chinese_char(text[j]):
                    if text[j] in self.punct_to_idx:
                        punct_indices.append(self.punct_to_idx[text[j]])
                    j += 1

                labels.append(punct_indices)
            i += 1

        return chars, labels

    def create_examples(self, chars, labels, max_length=512, overlap=50):
        """시퀀스를 고정 길이 예제로 분할"""
        if not chars:
            return []

        examples = []

        # 짧은 시퀀스는 하나의 예제로
        if len(chars) <= max_length:
            padded_labels = [l.copy() for l in labels]
            if len(chars) < max_length:
                padded_labels.extend([[] for _ in range(max_length - len(chars))])

            return [{
                'c': ''.join(chars),
                'l': padded_labels,
                'n': len(chars)
            }]

        # 긴 시퀀스는 sliding window
        stride = max_length - overlap

        for start in range(0, len(chars), stride):
            end = min(start + max_length, len(chars))

            # 청크 추출
            chunk_chars = chars[start:end]
            chunk_labels = [l.copy() for l in labels[start:end]]

            # 패딩
            actual_length = len(chunk_chars)
            if actual_length < max_length:
                chunk_labels.extend([[] for _ in range(max_length - actual_length)])

            examples.append({
                'c': ''.join(chunk_chars),
                'l': chunk_labels,
                'n': actual_length
            })

            if end >= len(chars):
                break

        return examples

    def process_file(self, file_path):
        """파일 처리 - 각 줄(기사)을 독립적으로 처리"""
        all_examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 빈 줄은 건너뛰기
                line = line.strip()
                if not line:
                    continue

                try:
                    # 각 줄(기사)을 독립적으로 처리
                    chars, labels = self.create_labels(line)
                    if chars:
                        # 이 줄에서 생성된 예제들
                        line_examples = self.create_examples(chars, labels)
                        all_examples.extend(line_examples)
                except Exception as e:
                    logging.warning(f"Error in {file_path.name} line {line_num}: {e}")

        return all_examples

    def generate_dataset(self, input_dir, output_dir, train_ratio=0.9):
        """데이터셋 생성"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 로깅 설정
        log_file = output_path / f"generation_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        # 파일 수집
        txt_files = list(input_path.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No txt files found in {input_path}")

        logging.info(f"Found {len(txt_files)} files")

        # 임시 파일에 저장
        temp_file = output_path / "temp_all.jsonl"
        total_examples = 0
        punct_count = {p: 0 for p in self.punctuations}  # 구두점별 카운트

        with open(temp_file, 'w', encoding='utf-8') as f:
            for txt_file in tqdm(txt_files, desc="Processing"):
                try:
                    examples = self.process_file(txt_file)
                    for ex in examples:
                        # 구두점별 카운트
                        for label_list in ex['l'][:ex['n']]:
                            for idx in label_list:
                                if 0 <= idx < self.num_punctuations:
                                    punct_count[self.punctuations[idx]] += 1

                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                        total_examples += 1
                    logging.info(f"{txt_file.name}: {len(examples)} examples")
                except Exception as e:
                    logging.error(f"Error in {txt_file.name}: {e}")

        # 셔플 및 분할
        logging.info(f"Total: {total_examples} examples")

        indices = np.arange(total_examples)
        np.random.seed(42)
        np.random.shuffle(indices)

        split_idx = int(total_examples * train_ratio)
        train_indices = set(indices[:split_idx])

        # 분할 저장
        train_count = val_count = 0

        with open(temp_file, 'r', encoding='utf-8') as f_in, \
                open(output_path / "train.jsonl", 'w', encoding='utf-8') as f_train, \
                open(output_path / "val.jsonl", 'w', encoding='utf-8') as f_val:

            for i, line in enumerate(tqdm(f_in, total=total_examples, desc="Splitting")):
                if i in train_indices:
                    f_train.write(line)
                    train_count += 1
                else:
                    f_val.write(line)
                    val_count += 1

        # 임시 파일 삭제
        temp_file.unlink()

        # 통계
        total_chars = 0
        total_puncts = 0
        with open(output_path / "train.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                total_chars += data['n']
                total_puncts += sum(len(l) for l in data['l'][:data['n']])

        logging.info(f"Total characters: {total_chars:,}")
        logging.info(f"Total punctuations: {total_puncts:,}")
        logging.info(f"Avg puncts per char: {total_puncts / total_chars:.3f}")
        logging.info("\n구두점별 분포:")
        for punct, count in punct_count.items():
            ratio = count / total_puncts * 100 if total_puncts > 0 else 0
            logging.info(f"  {punct}: {count:,}개 ({ratio:.1f}%)")

        # 설정 저장
        config = {
            'num_punctuations': self.num_punctuations,
            'punctuations': self.punctuations,
            'punct_to_idx': self.punct_to_idx,
            'max_length': 512,
            'train_examples': train_count,
            'val_examples': val_count
        }

        with open(output_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logging.info(f"Train: {train_count}, Val: {val_count}")
        return train_count, val_count


def main():
    """메인 함수"""
    print("고전 한문 구두점 예측 학습데이터 생성")
    print("=" * 50)

    generator = PunctuationDataGenerator()

    # 폴더 선택
    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="입력 폴더 선택")
    if not input_dir:
        print("취소됨")
        return

    output_dir = filedialog.askdirectory(title="출력 폴더 선택")
    if not output_dir:
        print("취소됨")
        return

    # 실행
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print("처리 중...")

    try:
        train, val = generator.generate_dataset(input_dir, output_dir)
        print(f"\n완료! Train: {train:,}, Val: {val:,}")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()