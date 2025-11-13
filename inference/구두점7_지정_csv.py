"""
CSV 파일의 한국 고전한문 텍스트에 구두점 추가
사용자가 처리할 열의 헤더를 직접 선택 가능
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import sys
import csv
from tkinter import filedialog
import tkinter as tk
from typing import List, Optional, Dict

# 같은 폴더의 구두점7_추론모델.py import
from 구두점7_추론모델 import PunctuationPredictor


class CSVPunctuationProcessor:
    """CSV 파일 구두점 처리기"""

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            checkpoint_path: 학습된 모델 체크포인트 경로
            device: 디바이스 설정
        """
        print(f"모델 초기화 중...")
        self.predictor = PunctuationPredictor(checkpoint_path, device)
        print(f"모델 로딩 완료! (디바이스: {self.predictor.device})")

        # 처리 통계
        self.stats = {
            'total_rows': 0,
            'processed_cells': 0,
            'failed_cells': 0,
            'total_chars': 0,
            'processing_time': 0
        }

        # 기본 임계값 설정
        if not hasattr(self.predictor, 'threshold') or self.predictor.threshold is None:
            self.predictor.threshold = 0.5

    def get_csv_headers(self, file_path: Path, encoding: str = 'utf-8') -> List[str]:
        """CSV 파일의 헤더 목록 가져오기"""
        try:
            # 다양한 인코딩 시도
            encodings = [encoding, 'utf-8', 'utf-8-sig', 'cp949', 'euc-kr']

            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, nrows=0, encoding=enc)
                    print(f"파일 인코딩: {enc}")
                    return list(df.columns)
                except UnicodeDecodeError:
                    continue

            raise ValueError("CSV 파일 인코딩을 감지할 수 없습니다")

        except Exception as e:
            print(f"헤더 읽기 실패: {e}")
            return []

    def process_csv(self, input_path: str, selected_columns: List[str],
                    output_dir: str = None, encoding: str = 'utf-8',
                    chunk_size: int = 100):
        """
        CSV 파일 처리

        Args:
            input_path: 입력 CSV 파일 경로
            selected_columns: 처리할 열 이름 리스트
            output_dir: 출력 디렉토리 (None이면 입력 파일과 같은 위치)
            encoding: 파일 인코딩
            chunk_size: 청크 단위 처리 크기
        """
        start_time = time.time()

        # 경로 설정
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

        # 출력 경로 설정
        threshold_str = f"{self.predictor.threshold:.1f}"
        if output_dir:
            output_path = Path(output_dir) / f"{input_file.stem}_구두점_{threshold_str}{input_file.suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_file.parent / f"{input_file.stem}_구두점_{threshold_str}{input_file.suffix}"

        print(f"\n입력 파일: {input_file}")
        print(f"출력 파일: {output_path}")
        print(f"처리할 열: {', '.join(selected_columns)}")
        print(f"처리 시작...\n")

        # 인코딩 자동 감지
        detected_encoding = self._detect_encoding(input_file)
        if detected_encoding:
            encoding = detected_encoding

        try:
            # 먼저 전체 행 수 확인
            total_rows = sum(1 for _ in open(input_file, 'r', encoding=encoding)) - 1  # 헤더 제외
            self.stats['total_rows'] = total_rows
            print(f"전체 행 수: {total_rows:,}")

            # 청크 단위로 처리
            processed_chunks = []
            error_log = []

            with tqdm(total=total_rows, desc="처리 진행") as pbar:
                for chunk_df in pd.read_csv(input_file, encoding=encoding,
                                            chunksize=chunk_size):
                    # 선택된 열만 처리
                    processed_chunk = self._process_chunk(
                        chunk_df, selected_columns, error_log
                    )
                    processed_chunks.append(processed_chunk)

                    # 진행률 업데이트
                    pbar.update(len(chunk_df))

            # 모든 청크 합치기
            result_df = pd.concat(processed_chunks, ignore_index=True)

            # 결과 저장
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig',
                             quoting=csv.QUOTE_NONNUMERIC)

            # 처리 시간 계산
            self.stats['processing_time'] = time.time() - start_time

            # 결과 요약 출력
            self._print_summary(output_path, error_log)

            # 오류 로그 저장 (있는 경우)
            if error_log:
                self._save_error_log(output_path, error_log)

            return str(output_path)

        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
            raise

    def _detect_encoding(self, file_path: Path) -> Optional[str]:
        """파일 인코딩 자동 감지"""
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1']

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.read(1000)  # 처음 1000자만 테스트
                return enc
            except UnicodeDecodeError:
                continue

        return None

    def _process_chunk(self, chunk_df: pd.DataFrame, selected_columns: List[str],
                       error_log: list) -> pd.DataFrame:
        """청크 단위 처리"""

        # 선택된 각 열에 대해 새 열 생성
        for col in selected_columns:
            if col not in chunk_df.columns:
                print(f"경고: '{col}' 열이 존재하지 않습니다.")
                continue

            new_col_name = f"{col}_구두점"
            chunk_df[new_col_name] = ''

            # 각 행 처리
            for idx in chunk_df.index:
                try:
                    text = str(chunk_df.at[idx, col])

                    if pd.isna(chunk_df.at[idx, col]) or text.strip() == '' or text == 'nan':
                        # 빈 텍스트는 그대로
                        chunk_df.at[idx, new_col_name] = ''
                    else:
                        # 구두점 예측
                        result = self.predictor.predict(text.strip())
                        chunk_df.at[idx, new_col_name] = result

                        self.stats['processed_cells'] += 1
                        self.stats['total_chars'] += len(text)

                except Exception as e:
                    # 오류 처리
                    self.stats['failed_cells'] += 1
                    error_msg = {
                        'row_index': idx,
                        'column': col,
                        'error': str(e),
                        'text_preview': text[:100] if len(text) > 100 else text
                    }
                    error_log.append(error_msg)

                    # 오류 발생 시 원문 그대로 저장
                    chunk_df.at[idx, new_col_name] = text

        # 열 순서 재정렬: 각 원본 열 바로 뒤에 구두점 열 배치
        final_columns = []
        added_punct_cols = set()  # 이미 추가된 구두점 열 추적

        for col in chunk_df.columns:
            # 구두점 열이 아닌 경우만 추가
            if not col.endswith('_구두점'):
                final_columns.append(col)
                # 해당 열의 구두점 열이 있으면 바로 뒤에 추가
                punct_col = f"{col}_구두점"
                if col in selected_columns and punct_col in chunk_df.columns:
                    final_columns.append(punct_col)
                    added_punct_cols.add(punct_col)

        # 혹시 빠진 구두점 열이 있다면 마지막에 추가
        for col in chunk_df.columns:
            if col.endswith('_구두점') and col not in added_punct_cols:
                final_columns.append(col)

        return chunk_df[final_columns]

    def _print_summary(self, output_path: Path, error_log: list):
        """처리 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("처리 완료!")
        print("=" * 60)
        print(f"전체 행: {self.stats['total_rows']:,}")
        print(f"처리된 셀: {self.stats['processed_cells']:,}")
        print(f"실패한 셀: {self.stats['failed_cells']:,}")
        print(f"처리된 문자 수: {self.stats['total_chars']:,}")
        print(f"처리 시간: {self.stats['processing_time']:.1f}초")

        if self.stats['processing_time'] > 0:
            print(f"처리 속도: {self.stats['total_chars'] / self.stats['processing_time']:.0f} 문자/초")

        total_cells = self.stats['processed_cells'] + self.stats['failed_cells']
        if total_cells > 0:
            success_rate = (self.stats['processed_cells'] / total_cells * 100)
            print(f"성공률: {success_rate:.1f}%")

        print(f"\n결과 파일: {output_path}")

        if error_log:
            print(f"\n⚠️  {len(error_log)}개 셀에서 오류 발생")
            print("상세 내용은 오류 로그 파일을 확인하세요.")

    def _save_error_log(self, output_path: Path, error_log: list):
        """오류 로그 저장"""
        error_log_path = output_path.parent / f"{output_path.stem}_errors.log"

        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"처리 시간: {datetime.now()}\n")
            f.write(f"오류 발생 셀 수: {len(error_log)}\n")
            f.write("=" * 60 + "\n\n")

            for error in error_log:
                f.write(f"행 인덱스: {error['row_index']}\n")
                f.write(f"열: {error['column']}\n")
                f.write(f"오류: {error['error']}\n")
                f.write(f"텍스트 미리보기: {error['text_preview']}\n")
                f.write("-" * 40 + "\n")

        print(f"오류 로그 저장: {error_log_path}")


def select_columns_from_list(headers: List[str]) -> List[str]:
    """헤더 목록에서 열 선택"""
    print("\n사용 가능한 열:")
    print("=" * 60)
    for i, header in enumerate(headers, 1):
        print(f"{i:3d}. {header}")
    print("=" * 60)

    print("\n처리할 열 번호를 입력하세요")
    print("(쉼표로 구분, 범위 지정 가능. 예: 1,3,5 또는 1-3,5 또는 all)")
    selected = input("선택: ").strip().lower()

    selected_columns = []

    if selected == 'all':
        selected_columns = headers
    else:
        try:
            indices = []
            for part in selected.split(','):
                part = part.strip()
                if '-' in part:
                    # 범위 처리 (예: 1-3)
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start - 1, end))
                else:
                    # 단일 번호
                    indices.append(int(part) - 1)

            # 중복 제거 및 정렬
            indices = sorted(set(indices))
            selected_columns = [headers[i] for i in indices if 0 <= i < len(headers)]

        except (ValueError, IndexError) as e:
            print(f"잘못된 입력입니다: {e}")
            return []

    return selected_columns


def interactive_mode():
    """대화형 모드로 실행"""
    print("=" * 60)
    print("한국 고전한문 구두점 지정 프로그램 (CSV 열 선택 버전)")
    print("=" * 60)

    # 1. 체크포인트 파일 선택
    print("\n1. 모델 체크포인트 파일을 선택하세요...")

    root = tk.Tk()
    root.withdraw()
    checkpoint_path = filedialog.askopenfilename(
        title="체크포인트 파일 선택",
        filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
    )
    root.destroy()

    if not checkpoint_path:
        print("체크포인트가 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    print(f"   선택된 체크포인트: {checkpoint_path}")

    # 2. 입력 CSV 파일 선택
    print("\n2. 처리할 CSV 파일을 선택하세요...")

    root = tk.Tk()
    root.withdraw()
    input_path = filedialog.askopenfilename(
        title="입력 CSV 파일 선택",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()

    if not input_path:
        print("입력 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    input_path = Path(input_path)
    print(f"   선택된 입력 파일: {input_path}")

    # 프로세서 생성 (헤더 읽기용)
    try:
        processor = CSVPunctuationProcessor(checkpoint_path, 'auto')

        # 3. CSV 헤더 읽기 및 열 선택
        print("\n3. CSV 파일의 헤더를 읽는 중...")
        headers = processor.get_csv_headers(input_path)

        if not headers:
            print("CSV 파일에서 헤더를 읽을 수 없습니다.")
            return

        print(f"   발견된 열: {len(headers)}개")

        # 열 선택
        selected_columns = select_columns_from_list(headers)

        if not selected_columns:
            print("선택된 열이 없습니다. 프로그램을 종료합니다.")
            return

        print(f"\n선택된 열 ({len(selected_columns)}개): {', '.join(selected_columns)}")

        # 4. 출력 위치 선택
        print("\n4. 결과 저장 위치를 선택하세요.")
        print("   [1] 입력 파일과 같은 폴더 (기본)")
        print("   [2] 다른 폴더 선택")

        choice = input("   선택 (1 또는 2) [1]: ").strip() or "1"

        output_dir = None
        if choice == "2":
            root = tk.Tk()
            root.withdraw()
            output_dir = filedialog.askdirectory(title="출력 폴더 선택")
            root.destroy()

            if not output_dir:
                print("   입력 파일과 같은 폴더에 저장합니다.")
            else:
                print(f"   선택된 출력 폴더: {output_dir}")

        # 5. 디바이스 선택
        print("\n5. 처리 디바이스를 선택하세요.")
        print("   [1] 자동 선택 (GPU 사용 가능 시 GPU)")
        print("   [2] CPU 강제 사용")
        print("   [3] GPU 강제 사용")

        device_choice = input("   선택 (1, 2, 또는 3) [1]: ").strip() or "1"

        device_map = {"1": "auto", "2": "cpu", "3": "cuda"}
        device = device_map.get(device_choice, "auto")

        # 6. 청크 크기
        chunk_input = input("\n6. 청크 크기 (기본: 100, 메모리 부족시 줄이세요): ").strip()
        chunk_size = int(chunk_input) if chunk_input else 100

        # 7. 임계값
        threshold_input = input("\n7. 예측 임계값 (0.1~0.9, 기본: 0.5): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.5

        # 8. 처리 시작
        print("\n" + "=" * 60)
        print("처리를 시작합니다...")
        print("=" * 60)

        # 디바이스 변경이 필요한 경우 프로세서 재생성
        if device != 'auto':
            processor = CSVPunctuationProcessor(checkpoint_path, device)

        processor.predictor.threshold = threshold
        print(f"임계값: {threshold}")

        # CSV 처리
        output_path = processor.process_csv(
            str(input_path),
            selected_columns,
            output_dir,
            chunk_size=chunk_size
        )

        print("\n" + "=" * 60)
        print("모든 처리가 완료되었습니다!")
        print(f"결과 파일: {output_path}")
        print("=" * 60)

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    import argparse

    # 명령줄 인자가 있는지 확인
    if len(sys.argv) > 1:
        # 명령줄 모드
        parser = argparse.ArgumentParser(
            description='CSV 파일의 선택된 열에 구두점 추가'
        )

        parser.add_argument('--checkpoint', type=str, required=True,
                            help='학습된 체크포인트 경로')
        parser.add_argument('--input', type=str, required=True,
                            help='입력 CSV 파일 경로')
        parser.add_argument('--columns', type=str, required=True,
                            help='처리할 열 이름 (쉼표로 구분)')
        parser.add_argument('--output-dir', type=str, default=None,
                            help='출력 디렉토리')
        parser.add_argument('--device', type=str, default='auto',
                            choices=['auto', 'cuda', 'cpu'])
        parser.add_argument('--encoding', type=str, default='utf-8')
        parser.add_argument('--chunk-size', type=int, default=100)
        parser.add_argument('--threshold', type=float, default=0.5)

        args = parser.parse_args()

        # 처리
        processor = CSVPunctuationProcessor(args.checkpoint, args.device)
        processor.predictor.threshold = args.threshold

        # 열 이름 파싱
        selected_columns = [col.strip() for col in args.columns.split(',')]

        processor.process_csv(
            args.input,
            selected_columns,
            args.output_dir,
            chunk_size=args.chunk_size,
            encoding=args.encoding
        )
    else:
        # 대화형 모드
        interactive_mode()

        # 종료 전 대기
        input("\n엔터를 누르면 종료합니다...")


if __name__ == "__main__":
    main()