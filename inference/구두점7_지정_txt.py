"""
TXT 파일의 한국 고전한문 텍스트에 구두점 추가
단일 파일 또는 폴더 내 모든 TXT 파일 처리 가능
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Dict

# 같은 폴더의 구두점7_추론모델.py import
from 구두점7_추론모델 import PunctuationPredictor


class TXTPunctuationProcessor:
    """TXT 파일 구두점 처리기"""

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
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_lines': 0,
            'processed_lines': 0,
            'failed_lines': 0,
            'total_chars': 0,
            'processing_time': 0
        }

        # 기본 임계값 설정
        if not hasattr(self.predictor, 'threshold') or self.predictor.threshold is None:
            self.predictor.threshold = 0.5

    def process_txt_file(self, input_path: Path, output_dir: Path = None) -> Optional[Path]:
        """
        단일 TXT 파일 처리

        Args:
            input_path: 입력 TXT 파일 경로
            output_dir: 출력 디렉토리 (None이면 입력 파일과 같은 위치)

        Returns:
            출력 파일 경로 (실패시 None)
        """
        start_time = time.time()

        # 출력 경로 설정
        threshold_str = f"{self.predictor.threshold:.1f}"
        if output_dir:
            output_path = output_dir / f"{input_path.stem}_구두점_{threshold_str}.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path.parent / f"{input_path.stem}_구두점_{threshold_str}.txt"

        print(f"\n처리 중: {input_path.name}")

        try:
            # 파일 읽기
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.stats['total_lines'] += len(lines)

            # 결과 저장용 리스트
            processed_lines = []
            error_lines = []

            # 각 줄 처리 (진행바 표시)
            for i, line in enumerate(tqdm(lines, desc=f"{input_path.name}", leave=False)):
                line = line.rstrip('\n')

                if not line.strip():
                    # 빈 줄은 그대로 유지
                    processed_lines.append(line)
                    self.stats['processed_lines'] += 1
                else:
                    try:
                        # 구두점 예측
                        result = self.predictor.predict(line.strip())
                        processed_lines.append(result)

                        self.stats['processed_lines'] += 1
                        self.stats['total_chars'] += len(line)

                    except Exception as e:
                        # 오류 발생시 원문 그대로 저장
                        processed_lines.append(line)
                        self.stats['failed_lines'] += 1

                        error_lines.append({
                            'line_num': i + 1,
                            'text': line[:100],
                            'error': str(e)
                        })

            # 결과 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(processed_lines))

            # 오류 로그 저장 (있는 경우)
            if error_lines:
                self._save_error_log(output_path, error_lines)

            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            self.stats['processed_files'] += 1

            print(f"  ✓ 완료: {len(lines)}줄 처리 ({processing_time:.1f}초)")

            return output_path

        except Exception as e:
            print(f"  ✗ 실패: {str(e)}")
            self.stats['failed_files'] += 1
            return None

    def process_folder(self, folder_path: Path, output_dir: Path = None,
                       recursive: bool = False) -> List[Path]:
        """
        폴더 내 모든 TXT 파일 처리

        Args:
            folder_path: 입력 폴더 경로
            output_dir: 출력 디렉토리 (None이면 각 파일과 같은 위치)
            recursive: 하위 폴더 포함 여부

        Returns:
            처리된 파일 경로 리스트
        """
        # TXT 파일 찾기
        if recursive:
            txt_files = list(folder_path.rglob("*.txt"))
        else:
            txt_files = list(folder_path.glob("*.txt"))

        # 이미 처리된 파일 제외 (구두점이 파일명에 포함된 경우)
        txt_files = [f for f in txt_files if "구두점" not in f.stem]

        if not txt_files:
            print("처리할 TXT 파일이 없습니다.")
            return []

        print(f"\n발견된 TXT 파일: {len(txt_files)}개")
        self.stats['total_files'] = len(txt_files)

        processed_files = []

        # 각 파일 처리
        for txt_file in tqdm(txt_files, desc="전체 진행"):
            # 출력 디렉토리 설정 (폴더 구조 유지)
            if output_dir and recursive:
                relative_path = txt_file.parent.relative_to(folder_path)
                file_output_dir = output_dir / relative_path
            else:
                file_output_dir = output_dir

            output_path = self.process_txt_file(txt_file, file_output_dir)
            if output_path:
                processed_files.append(output_path)

        return processed_files

    def _save_error_log(self, output_path: Path, error_lines: List[Dict]):
        """오류 로그 저장"""
        error_log_path = output_path.parent / f"{output_path.stem}_errors.log"

        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"처리 시간: {datetime.now()}\n")
            f.write(f"오류 발생 줄 수: {len(error_lines)}\n")
            f.write("=" * 60 + "\n\n")

            for error in error_lines:
                f.write(f"줄 번호: {error['line_num']}\n")
                f.write(f"텍스트: {error['text']}\n")
                f.write(f"오류: {error['error']}\n")
                f.write("-" * 40 + "\n")

    def print_summary(self):
        """처리 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("처리 완료!")
        print("=" * 60)

        if self.stats['total_files'] > 0:
            print(f"전체 파일: {self.stats['total_files']:,}")
            print(f"성공: {self.stats['processed_files']:,}")
            print(f"실패: {self.stats['failed_files']:,}")

        print(f"처리된 줄 수: {self.stats['processed_lines']:,}")
        print(f"실패한 줄 수: {self.stats['failed_lines']:,}")
        print(f"처리된 문자 수: {self.stats['total_chars']:,}")
        print(f"전체 처리 시간: {self.stats['processing_time']:.1f}초")

        if self.stats['processing_time'] > 0:
            print(f"처리 속도: {self.stats['total_chars'] / self.stats['processing_time']:.0f} 문자/초")

        if self.stats['total_lines'] > 0:
            success_rate = (self.stats['processed_lines'] / self.stats['total_lines'] * 100)
            print(f"성공률: {success_rate:.1f}%")


def select_file_or_folder():
    """파일 또는 폴더 선택 대화상자"""
    print("\n입력 방식을 선택하세요:")
    print("  [1] 단일 파일 처리")
    print("  [2] 폴더 내 모든 파일 처리")

    choice = input("선택 (1 또는 2): ").strip()

    root = tk.Tk()
    root.withdraw()

    if choice == "1":
        # 파일 선택
        path = filedialog.askopenfilename(
            title="TXT 파일 선택",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        is_file = True
    else:
        # 폴더 선택
        path = filedialog.askdirectory(title="폴더 선택")
        is_file = False

    root.destroy()

    return path, is_file


def interactive_mode():
    """대화형 모드로 실행"""
    print("=" * 60)
    print("한국 고전한문 구두점 지정 프로그램 (TXT 버전)")
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

    # 2. 입력 파일/폴더 선택
    print("\n2. 처리할 파일 또는 폴더를 선택하세요...")
    input_path, is_file = select_file_or_folder()

    if not input_path:
        print("입력이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    print(f"   선택된 {'파일' if is_file else '폴더'}: {input_path}")

    # 3. 출력 위치 선택
    print("\n3. 결과 저장 위치를 선택하세요.")
    print("   [1] 원본과 같은 위치 (기본)")
    print("   [2] 다른 폴더 선택")

    choice = input("   선택 (1 또는 2) [1]: ").strip() or "1"

    output_dir = None
    if choice == "2":
        root = tk.Tk()
        root.withdraw()
        output_dir = filedialog.askdirectory(title="출력 폴더 선택")
        root.destroy()

        if output_dir:
            output_dir = Path(output_dir)
            print(f"   선택된 출력 폴더: {output_dir}")
        else:
            print("   원본과 같은 위치에 저장합니다.")

    # 4. 하위 폴더 포함 여부 (폴더 선택시)
    recursive = False
    if not is_file:
        print("\n4. 하위 폴더도 포함하시겠습니까?")
        rec_choice = input("   포함 (y/n) [n]: ").strip().lower()
        recursive = rec_choice == 'y'

    # 5. 디바이스 선택
    print(f"\n{5 if not is_file else 4}. 처리 디바이스를 선택하세요.")
    print("   [1] 자동 선택 (GPU 사용 가능 시 GPU)")
    print("   [2] CPU 강제 사용")
    print("   [3] GPU 강제 사용")

    device_choice = input("   선택 (1, 2, 또는 3) [1]: ").strip() or "1"

    device_map = {"1": "auto", "2": "cpu", "3": "cuda"}
    device = device_map.get(device_choice, "auto")

    # 6. 임계값 설정
    threshold_input = input(f"\n{6 if not is_file else 5}. 예측 임계값 (0.1~0.9, 기본: 0.5): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.5

    # 7. 처리 시작
    print("\n" + "=" * 60)
    print("처리를 시작합니다...")
    print("=" * 60)

    try:
        # 프로세서 생성
        processor = TXTPunctuationProcessor(checkpoint_path, device)
        processor.predictor.threshold = threshold
        print(f"임계값: {threshold}")

        # 처리 실행
        input_path = Path(input_path)

        if is_file:
            # 단일 파일 처리
            output_path = processor.process_txt_file(input_path, output_dir)
            if output_path:
                print(f"\n결과 파일: {output_path}")
        else:
            # 폴더 처리
            processed_files = processor.process_folder(input_path, output_dir, recursive)
            if processed_files:
                print(f"\n처리된 파일 수: {len(processed_files)}")

        # 요약 출력
        processor.print_summary()

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
            description='TXT 파일의 한문 텍스트에 구두점 추가'
        )

        parser.add_argument('--checkpoint', type=str, required=True,
                            help='학습된 체크포인트 경로')
        parser.add_argument('--input', type=str, required=True,
                            help='입력 TXT 파일 또는 폴더 경로')
        parser.add_argument('--output-dir', type=str, default=None,
                            help='출력 디렉토리')
        parser.add_argument('--device', type=str, default='auto',
                            choices=['auto', 'cuda', 'cpu'])
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--recursive', action='store_true',
                            help='하위 폴더 포함 (폴더 모드에서만)')

        args = parser.parse_args()

        # 처리
        processor = TXTPunctuationProcessor(args.checkpoint, args.device)
        processor.predictor.threshold = args.threshold

        input_path = Path(args.input)
        output_dir = Path(args.output_dir) if args.output_dir else None

        if input_path.is_file():
            # 파일 처리
            processor.process_txt_file(input_path, output_dir)
        elif input_path.is_dir():
            # 폴더 처리
            processor.process_folder(input_path, output_dir, args.recursive)
        else:
            print(f"오류: {input_path}는 유효한 파일 또는 폴더가 아닙니다.")
            sys.exit(1)

        processor.print_summary()

    else:
        # 대화형 모드
        interactive_mode()

        # 종료 전 대기
        input("\n엔터를 누르면 종료합니다...")


if __name__ == "__main__":
    main()