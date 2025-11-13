"""
고전한문 구두점 위치 검증 스크립트
단일 파일 또는 폴더 내 모든 파일 처리 가능
구두점 종류는 무시하고 위치만 비교
"""

from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from datetime import datetime


def validate_punctuation_positions(model_checkpoint, input_path, is_folder=False, recursive=False):
    """고전한문 구두점 위치 검증 메인 함수"""

    print("=" * 70)
    print("고전한문 구두점 위치 검증 시스템")
    print("=" * 70)

    # 1. 모델 로드
    print("\n[1/4] 모델 로딩...")
    from 구두점7_추론모델 import PunctuationPredictor
    predictor = PunctuationPredictor(model_checkpoint)
    print(f"   ✓ 모델 로드 완료 (디바이스: {predictor.device})")

    # 2. 파일 목록 구성
    print("\n[2/4] 파일 목록 구성...")

    if is_folder:
        input_path = Path(input_path)
        if recursive:
            # 하위 폴더 포함
            file_list = list(input_path.rglob("*.txt"))
            print(f"   ✓ 하위 폴더 포함: {len(file_list)}개 파일 발견")
        else:
            # 현재 폴더만
            file_list = list(input_path.glob("*.txt"))
            print(f"   ✓ 현재 폴더만: {len(file_list)}개 파일 발견")

        # 이미 처리된 파일 제외
        file_list = [f for f in file_list if "validation" not in f.stem.lower()]
        print(f"   ✓ 처리 대상: {len(file_list)}개 파일")
    else:
        file_list = [Path(input_path)]
        print(f"   ✓ 단일 파일 모드")

    if not file_list:
        print("처리할 파일이 없습니다.")
        return None

    # 3. 전체 통계 변수
    total_stats = {
        'total_files': len(file_list),
        'total_lines': 0,
        'total_chars': 0,
        'total_gold_puncts': 0,
        'total_pred_puncts': 0,
        'exact_matches': 0,
        'false_positives': 0,
        'false_negatives': 0
    }

    # 파일별 결과 저장
    file_results = []

    # 오류 분석용
    error_analysis = {
        'over_prediction': [],
        'under_prediction': [],
        'position_shift': []
    }

    # 전체 샘플 저장용
    all_samples = []

    print("\n[3/4] 검증 진행...")

    # 4. 각 파일 처리
    for file_idx, file_path in enumerate(tqdm(file_list, desc="파일 처리")):
        print(f"\n   처리 중: {file_path.name}")

        # 파일별 통계
        file_stats = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'lines': 0,
            'chars': 0,
            'gold_puncts': 0,
            'pred_puncts': 0,
            'exact': 0,
            'fp': 0,
            'fn': 0
        }

        # 파일 읽기
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"   ✗ 파일 읽기 실패: {e}")
            continue

        # 각 줄 처리 (진행바 추가)
        total = len(lines)
        bar_length = 30

        for line_idx, line in enumerate(lines):
            # 10줄마다 진행바 업데이트
            if line_idx % 10 == 0:
                progress = line_idx / total
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                pct = int(progress * 100)
                print(f"\r   [{bar}] {line_idx}/{total} ({pct}%)", end='', flush=True)

            line = line.strip()
            if not line:
                continue

            file_stats['lines'] += 1
            total_stats['total_lines'] += 1

            # 원본에서 한자와 구두점 위치 추출
            chinese_only, gold_positions = extract_chinese_and_positions(line)

            if not chinese_only or len(chinese_only) < 2:
                continue

            file_stats['chars'] += len(chinese_only)
            file_stats['gold_puncts'] += len(gold_positions)
            total_stats['total_chars'] += len(chinese_only)
            total_stats['total_gold_puncts'] += len(gold_positions)

            # 모델 예측
            predicted = predictor.predict(chinese_only)

            # 예측 결과에서 구두점 위치 추출
            _, pred_positions = extract_chinese_and_positions(predicted)
            file_stats['pred_puncts'] += len(pred_positions)
            total_stats['total_pred_puncts'] += len(pred_positions)

            # 위치 비교
            position_results = compare_positions(gold_positions, pred_positions)

            # 파일별 통계 업데이트
            file_stats['exact'] += position_results['exact']
            file_stats['fp'] += position_results['fp']
            file_stats['fn'] += position_results['fn']

            # 전체 통계 업데이트
            total_stats['exact_matches'] += position_results['exact']
            total_stats['false_positives'] += position_results['fp']
            total_stats['false_negatives'] += position_results['fn']

            # 오류 분석
            if len(pred_positions) > len(gold_positions) * 1.5:
                error_analysis['over_prediction'].append({
                    'file': file_path.name,
                    'line': line_idx + 1
                })
            elif len(pred_positions) < len(gold_positions) * 0.5:
                error_analysis['under_prediction'].append({
                    'file': file_path.name,
                    'line': line_idx + 1
                })

            # 샘플 저장 (각 파일에서 처음 10개씩)
            if line_idx < 10 and file_idx < 10:
                all_samples.append({
                    'file': file_path.name,
                    'line_no': line_idx + 1,
                    'original': line[:150],
                    'chinese_only': chinese_only[:150],
                    'predicted': predicted[:150],
                    'gold_positions': gold_positions[:20],
                    'pred_positions': pred_positions[:20],
                    'gold_count': len(gold_positions),
                    'pred_count': len(pred_positions)
                })

        # 파일별 메트릭 계산
        if file_stats['gold_puncts'] > 0 and file_stats['pred_puncts'] > 0:
            file_stats['metrics'] = calculate_file_metrics(file_stats)
        else:
            file_stats['metrics'] = None

        print(f"\r   [{'█' * 30}] {total}/{total} (100%) ✓")

        file_results.append(file_stats)

    print("\n[4/4] 결과 분석...")

    # 5. 전체 결과 계산 및 출력
    total_results = calculate_metrics(total_stats)

    if is_folder:
        print_folder_results(total_stats, total_results, file_results, error_analysis, all_samples)
    else:
        print_file_results(total_stats, total_results, error_analysis, all_samples)

    # 6. 결과 저장
    save_all_results(total_stats, total_results, file_results, error_analysis, all_samples, input_path, is_folder)

    return total_results


def calculate_file_metrics(file_stats):
    """파일별 메트릭 계산 - 정확 일치만"""
    metrics = {}

    if file_stats['gold_puncts'] > 0 and file_stats['pred_puncts'] > 0:
        precision = file_stats['exact'] / file_stats['pred_puncts']
        recall = file_stats['exact'] / file_stats['gold_puncts']
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics['f1_at_0'] = f1  # CSV 저장 호환성 위해 키 이름 유지

    return metrics


def print_folder_results(total_stats, total_results, file_results, error_analysis, samples):
    """폴더 처리 결과 출력"""

    print("\n" + "=" * 70)
    print("폴더 전체 검증 결과")
    print("=" * 70)

    # 전체 통계
    print("\n【전체 데이터 규모】")
    print(f"  처리 파일: {total_stats['total_files']:,}개")
    print(f"  총 줄 수: {total_stats['total_lines']:,}줄")
    print(f"  총 한자 수: {total_stats['total_chars']:,}자")
    print(f"  원본 구두점: {total_stats['total_gold_puncts']:,}개")
    print(f"  예측 구두점: {total_stats['total_pred_puncts']:,}개")

    # 정확 일치 성능만
    print("\n【정확 일치 성능】")
    if 'exact' in total_results:
        m = total_results['exact']
        print(f"  Precision: {m['precision']:.3f}")
        print(f"  Recall: {m['recall']:.3f}")
        print(f"  F1 Score: {m['f1']:.3f}")
        print(f"  정확 일치: {m['matches']:,}개")
        print(f"  오탐(FP): {total_stats['false_positives']:,}개")
        print(f"  미탐(FN): {total_stats['false_negatives']:,}개")


def print_file_results(total_stats, total_results, error_analysis, samples):
    """단일 파일 결과 출력 (기존 함수와 동일)"""
    print_results(total_stats, total_results, error_analysis, samples)


def save_all_results(total_stats, total_results, file_results, error_analysis, samples, input_path, is_folder):
    """모든 결과 저장"""

    print("\n" + "=" * 70)
    print("결과 자동 저장 중...")
    print("=" * 70)

    # 파일명 prefix 설정
    if is_folder:
        # 폴더명 사용
        folder_name = Path(input_path).name
        file_prefix = folder_name
        output_dir = Path(input_path)  # 같은 경로에 저장
    else:
        # 파일명 사용 (확장자 제외)
        file_name = Path(input_path).stem
        file_prefix = file_name
        output_dir = Path(input_path).parent  # 파일과 같은 경로에 저장

    # 1. 오류_분석_샘플.json
    sample_path = output_dir / f"{file_prefix}_오류분석_샘플.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'input': str(input_path),
                'test_date': datetime.now().isoformat(),
                'is_folder': is_folder,
                'total_files': total_stats['total_files']
            },
            'error_analysis': {
                'over_prediction_count': len(error_analysis['over_prediction']),
                'under_prediction_count': len(error_analysis['under_prediction']),
                'over_prediction_samples': error_analysis['over_prediction'][:100],
                'under_prediction_samples': error_analysis['under_prediction'][:100]
            },
            'sample_predictions': samples[:100]  # 샘플 100개
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ 오류 분석 샘플 저장: {sample_path.name}")

    # 2. 결과_종합.csv
    summary_csv_path = output_dir / f"{file_prefix}_결과_종합.csv"
    with open(summary_csv_path, 'w', encoding='utf-8-sig') as f:
        f.write("처리 파일 수,총 줄 수,총 한자 수,원본구두점 개수,예측구두점 개수,Precision,Recall,F1,정확 일치 수,오탐(FP) 수,미탐(FN) 수\n")
        if 'exact' in total_results:
            m = total_results['exact']
            f.write(f"{total_stats['total_files']},"
                    f"{total_stats['total_lines']},"
                    f"{total_stats['total_chars']},"
                    f"{total_stats['total_gold_puncts']},"
                    f"{total_stats['total_pred_puncts']},"
                    f"{m['precision']:.4f},"
                    f"{m['recall']:.4f},"
                    f"{m['f1']:.4f},"
                    f"{m['matches']},"
                    f"{total_stats['false_positives']},"
                    f"{total_stats['false_negatives']}\n")
    print(f"✓ 결과 종합 저장: {summary_csv_path.name}")

    # 3. 파일별_결과.csv
    if is_folder:
        # 폴더 모드: 여러 파일 결과
        file_csv_path = output_dir / f"{file_prefix}_파일별_결과.csv"
        with open(file_csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("파일명,줄 수,한자 수,원본구두점 개수,예측구두점 개수,정확일치 수,오탐(FP) 수,미탐(FN) 수,Precision,Recall,F1\n")
            for file_stat in file_results:
                if file_stat.get('metrics'):
                    precision = file_stat['exact'] / file_stat['pred_puncts'] if file_stat['pred_puncts'] > 0 else 0
                    recall = file_stat['exact'] / file_stat['gold_puncts'] if file_stat['gold_puncts'] > 0 else 0

                    f.write(f"{file_stat['file_name']},"
                            f"{file_stat['lines']},"
                            f"{file_stat['chars']},"
                            f"{file_stat['gold_puncts']},"
                            f"{file_stat['pred_puncts']},"
                            f"{file_stat['exact']},"
                            f"{file_stat['fp']},"
                            f"{file_stat['fn']},"
                            f"{precision:.4f},"
                            f"{recall:.4f},"
                            f"{file_stat['metrics'].get('f1_at_0', 0):.4f}\n")
        print(f"✓ 파일별 결과 저장: {file_csv_path.name}")
    else:
        # 단일 파일 모드: 파일별_결과 생략하거나 단일 결과 저장
        file_csv_path = output_dir / f"{file_prefix}_파일별_결과.csv"
        with open(file_csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("파일명,줄 수,한자 수,원본구두점 개수,예측구두점 개수,정확일치 수,오탐(FP) 수,미탐(FN) 수,Precision,Recall,F1\n")
            if 'exact' in total_results:
                m = total_results['exact']
                f.write(f"{Path(input_path).name},"
                        f"{total_stats['total_lines']},"
                        f"{total_stats['total_chars']},"
                        f"{total_stats['total_gold_puncts']},"
                        f"{total_stats['total_pred_puncts']},"
                        f"{total_stats['exact_matches']},"
                        f"{total_stats['false_positives']},"
                        f"{total_stats['false_negatives']},"
                        f"{m['precision']:.4f},"
                        f"{m['recall']:.4f},"
                        f"{m['f1']:.4f}\n")
        print(f"✓ 파일별 결과 저장: {file_csv_path.name}")

    print(f"\n저장 위치: {output_dir}")


# 기존 함수들은 그대로 유지
def extract_chinese_and_positions(text):
    """텍스트에서 한자 추출 및 구두점 위치 기록"""
    chinese_only = ""
    punct_positions = []
    char_pos = 0

    all_puncts = set('。、，．,.:;!?·‧․「」『』《》〈〉【】〔〕()[]''""')

    for char in text:
        if is_chinese_char(char):
            chinese_only += char
            char_pos += 1
        elif char in all_puncts:
            if char_pos > 0:
                punct_positions.append(char_pos - 1)

    return chinese_only, punct_positions


def is_chinese_char(char):
    """한자 판별 - 모든 CJK 영역 포함"""
    code = ord(char)
    return (0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or  # CJK Extension A
            0x20000 <= code <= 0x2A6DF or  # CJK Extension B
            0x2A700 <= code <= 0x2B73F or  # CJK Extension C
            0x2B740 <= code <= 0x2B81F or  # CJK Extension D
            0x2B820 <= code <= 0x2CEAF or  # CJK Extension E
            0x2CEB0 <= code <= 0x2EBEF or  # CJK Extension F
            0x30000 <= code <= 0x3134F or  # CJK Extension G
            0x31350 <= code <= 0x323AF or  # CJK Extension H
            0x2F800 <= code <= 0x2FA1F or  # CJK Compatibility Supplement
            0xF900 <= code <= 0xFAFF or   # CJK Compatibility Ideographs
            0xFA00 <= code <= 0xFA6F or   # CJK Compatibility Ideographs
            0xFA70 <= code <= 0xFADF or   # CJK Compatibility Ideographs
            0x2F00 <= code <= 0x2FDF or   # Kangxi Radicals
            0x2E80 <= code <= 0x2EFF)


def compare_positions(gold_positions, pred_positions):
    """구두점 위치 비교 """
    results = {
        'exact': 0,
        'fp': 0,
        'fn': 0
    }

    gold_set = set(gold_positions)
    pred_set = set(pred_positions)

    # 정확 일치만 계산
    exact_matches = gold_set & pred_set
    results['exact'] = len(exact_matches)

    # False Positive: 잘못 예측한 것
    results['fp'] = len(pred_set - exact_matches)

    # False Negative: 놓친 것
    results['fn'] = len(gold_set - exact_matches)

    return results


def calculate_metrics(stats):
    """평가 지표 계산 - 정확 일치만"""
    results = {}

    if stats['total_gold_puncts'] > 0 and stats['total_pred_puncts'] > 0:
        precision = stats['exact_matches'] / stats['total_pred_puncts']
        recall = stats['exact_matches'] / stats['total_gold_puncts']
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results['exact'] = {
            'matches': stats['exact_matches'],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    if stats['total_chars'] > 0:
        results['density'] = {
            'gold': stats['total_gold_puncts'] / stats['total_chars'],
            'pred': stats['total_pred_puncts'] / stats['total_chars']
        }

    return results


def print_results(stats, results, error_analysis, samples):
    """단일 파일 결과 출력"""
    print("\n" + "=" * 70)
    print("검증 결과 요약")
    print("=" * 70)

    print("\n【데이터 규모】")
    print(f"  검증 줄 수: {stats['total_lines']:,}줄")
    print(f"  총 한자 수: {stats['total_chars']:,}자")
    print(f"  원본 구두점: {stats['total_gold_puncts']:,}개")
    print(f"  예측 구두점: {stats['total_pred_puncts']:,}개")

    if 'density' in results:
        print(f"\n【구두점 밀도】")
        gold_per_100 = results['density']['gold'] * 100
        pred_per_100 = results['density']['pred'] * 100
        print(f"  원본: {gold_per_100:.1f}개/100자")
        print(f"  예측: {pred_per_100:.1f}개/100자")

    # 정확 일치 성능만
    print("\n【정확 일치 성능】")
    if 'exact' in results:
        m = results['exact']
        print(f"  Precision: {m['precision']:.3f}")
        print(f"  Recall: {m['recall']:.3f}")
        print(f"  F1 Score: {m['f1']:.3f}")
        print(f"  정확 일치: {m['matches']:,}개")
        print(f"  오탐(FP): {stats['false_positives']:,}개")
        print(f"  미탐(FN): {stats['false_negatives']:,}개")


def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("고전한문 구두점 위치 검증 프로그램")
    print("=" * 70)

    root = tk.Tk()
    root.withdraw()

    # 1. 체크포인트 선택
    print("\n1. 모델 체크포인트 선택...")
    checkpoint_path = filedialog.askopenfilename(
        title="체크포인트 파일 선택",
        filetypes=[("Checkpoint", "*.ckpt"), ("All", "*.*")]
    )

    if not checkpoint_path:
        print("취소되었습니다.")
        return
    print(f"   선택: {Path(checkpoint_path).name}")

    # 2. 입력 방식 선택 (2개로 축소)
    print("\n2. 입력 방식 선택")
    print("   [1] 단일 파일")
    print("   [2] 폴더 (하위 폴더 포함)")

    choice = input("   선택 (1/2): ").strip()

    if choice == "1":
        # 단일 파일
        print("\n3. 테스트 파일 선택...")
        test_path = filedialog.askopenfilename(
            title="구두점이 있는 한문 텍스트 파일",
            filetypes=[("Text", "*.txt"), ("All", "*.*")]
        )

        if not test_path:
            print("취소되었습니다.")
            return

        print(f"   선택: {Path(test_path).name}")
        results = validate_punctuation_positions(checkpoint_path, test_path, is_folder=False)

    elif choice == "2":
        # 폴더 (하위 폴더 포함)
        print("\n3. 폴더 선택...")
        folder_path = filedialog.askdirectory(
            title="테스트 파일들이 있는 폴더 선택"
        )

        if not folder_path:
            print("취소되었습니다.")
            return

        print(f"   선택: {folder_path}")
        print(f"   하위 폴더 포함: True")

        results = validate_punctuation_positions(checkpoint_path, folder_path, is_folder=True, recursive=True)

    else:
        print("잘못된 선택입니다.")
        return

    print("\n" + "=" * 70)
    print("검증 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()