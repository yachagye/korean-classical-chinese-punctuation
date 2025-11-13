"""
생성된 JSONL 데이터 샘플 확인 및 통계 분석
7개 구두점 버전
"""

import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from collections import Counter
import random
from tqdm import tqdm

# 7개 구두점 정의 (학습데이터 생성 코드와 일치)
punctuations = [
    ',', '。', '·', '?', '!', '《', '》'
]

def show_sample_reconstruction(jsonl_path: str, num_samples: int = 10) -> None:
    """JSONL 데이터에서 샘플을 복원하여 보여주기"""
    print("\n" + "=" * 80)
    print("📝 학습데이터 샘플 확인")
    print("=" * 80)

    # JSONL 파일에서 랜덤 샘플 추출
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # 전체 라인 수 계산
        total_lines = sum(1 for _ in f)

        # 다시 읽기
        f.seek(0)

        # 랜덤 인덱스 생성
        if total_lines > num_samples:
            sample_indices = sorted(random.sample(range(total_lines), num_samples))
        else:
            sample_indices = list(range(total_lines))

        # 샘플 수집
        for i, line in enumerate(f):
            if i in sample_indices:
                try:
                    samples.append((i, json.loads(line)))
                except json.JSONDecodeError as e:
                    print(f"경고: 줄 {i + 1} JSON 파싱 오류: {e}")
                    continue

                if len(samples) >= num_samples:
                    break

    print(f"\n총 {total_lines:,}개 중 {len(samples)}개 샘플 확인\n")

    # 전체 통계
    total_chars = 0
    total_puncts = 0
    all_punct_counter = Counter()

    # 각 샘플 복원 및 출력
    for sample_idx, (line_no, sample) in enumerate(samples, 1):
        print(f"\n{'=' * 60}")
        print(f"샘플 #{sample_idx} (줄 번호: {line_no + 1:,})")
        print(f"{'=' * 60}")

        # 데이터 추출
        chars = sample['c']
        compressed_labels = sample['l']
        length = sample['n']

        # 1. 원본 한자만 출력
        print(f"\n1) 원본 한자 ({length}자):")
        print(f"   {chars[:length]}")

        # 2. 라벨 정보 출력 (구두점이 있는 위치만)
        punct_positions = []
        for i, indices in enumerate(compressed_labels[:length]):
            if indices:  # 구두점이 있는 경우만
                punct_list = [punctuations[idx] for idx in indices if 0 <= idx < len(punctuations)]
                punct_positions.append((i, chars[i], punct_list))

        if punct_positions:
            print(f"\n2) 구두점 위치 (총 {len(punct_positions)}곳):")
            for pos, char, puncts in punct_positions[:10]:  # 처음 10개만 표시
                print(f"   위치 {pos}: {char} → {puncts}")
            if len(punct_positions) > 10:
                print(f"   ... 외 {len(punct_positions) - 10}곳")

        # 3. 복원된 텍스트 (처음 100자만)
        print(f"\n3) 복원된 텍스트:")
        reconstructed = ""
        display_length = min(100, length)
        for i in range(display_length):
            reconstructed += chars[i]
            if i < len(compressed_labels) and compressed_labels[i]:
                for idx in compressed_labels[i]:
                    if 0 <= idx < len(punctuations):
                        reconstructed += punctuations[idx]

        if length > 100:
            print(f"   {reconstructed}... (총 {length}자)")
        else:
            print(f"   {reconstructed}")

        # 4. 통계
        sample_puncts = sum(len(indices) for indices in compressed_labels[:length])
        total_chars += length
        total_puncts += sample_puncts

        print(f"\n4) 샘플 통계:")
        print(f"   - 한자 수: {length}")
        print(f"   - 구두점 수: {sample_puncts}")
        print(f"   - 구두점 비율: {sample_puncts / length:.2%}")
        print(f"   - 최대 길이: {len(chars)} (패딩 포함)")

        # 5. 구두점 분포
        punct_counter = Counter()
        for indices in compressed_labels[:length]:
            for idx in indices:
                if 0 <= idx < len(punctuations):
                    punct_counter[punctuations[idx]] += 1
                    all_punct_counter[punctuations[idx]] += 1

        if punct_counter:
            print(f"\n5) 이 샘플의 구두점 분포 (상위 5개):")
            for punct, count in punct_counter.most_common(5):
                print(f"   {punct}: {count}회")

    # 전체 통계 출력
    print(f"\n{'=' * 80}")
    print("📊 전체 샘플 통계")
    print(f"{'=' * 80}")
    print(f"총 {len(samples)}개 샘플 분석:")
    print(f"- 총 한자 수: {total_chars:,}")
    print(f"- 총 구두점 수: {total_puncts:,}")
    print(f"- 평균 구두점 비율: {total_puncts / total_chars:.2%}")

    if all_punct_counter:
        print(f"\n전체 구두점 분포:")
        for punct, count in all_punct_counter.most_common():
            print(f"  {punct}: {count:,}회 ({count / total_puncts * 100:.1f}%)")

    # 구두점 인덱스 확인
    print(f"\n구두점 인덱스 매핑:")
    for i, punct in enumerate(punctuations):
        print(f"  {i:2d}: {punct}")

    # 전처리 데이터와 비교 (샘플 기준)
    if len(samples) >= 100:  # 샘플이 충분히 많을 때만
        compare_with_preprocessing_stats(all_punct_counter, total_puncts)


def load_preprocessing_stats():
    """전처리 분석 파일 로드"""
    root = tk.Tk()
    root.withdraw()

    print("\n전처리 분석 파일을 선택하세요 (선택 안하면 하드코딩된 값 사용)...")
    stats_path = filedialog.askopenfilename(
        title="전처리 분석 txt 파일 선택",
        filetypes=[("Text files", "*.txt")]
    )

    if not stats_path:
        # 하드코딩된 기본값 사용
        print("전처리 분석 파일이 선택되지 않아 기본값을 사용합니다.")
        return {
            ',': 38732510,
            '。': 10277924,
            '·': 2053592,
            '?': 1874165,
            '》': 49977,
            '《': 49971,
            '!': 19575
        }

    print(f"전처리 분석 파일 읽는 중: {stats_path}")

    stats = {}
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # [구두점 목록] 섹션 찾기
        in_punct_section = False
        for line in lines:
            line = line.strip()

            if '[구두점 목록]' in line:
                in_punct_section = True
                continue
            elif in_punct_section and line.startswith('['):
                # 다른 섹션 시작
                break
            elif in_punct_section and line and '(' in line and ':' in line:
                # 구두점 정보 파싱
                # 예: , (U+002C): 38,732,510회
                try:
                    punct_part = line.split('(')[0].strip()
                    count_part = line.split(':')[1].strip()
                    count = int(count_part.replace(',', '').replace('회', ''))

                    # 구두점 문자 추출
                    if punct_part:
                        stats[punct_part] = count
                except Exception as e:
                    continue

        # 곡선 따옴표 처리 (파일에서는 실제 문자로 저장됨)
        if "'" in stats:
            stats[chr(0x2018)] = stats.pop("'")
        if "'" in stats:
            stats[chr(0x2019)] = stats.pop("'")

        print(f"전처리 분석 파일에서 {len(stats)}개 구두점 통계 로드 완료")

    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

    return stats


def compare_with_preprocessing_stats(all_punct_counter, total_puncts, preprocessing_stats=None):
    """전처리 통계와 비교"""
    print(f"\n{'=' * 80}")
    print("📊 전처리 데이터와 비교")
    print(f"{'=' * 80}")

    if preprocessing_stats is None:
        preprocessing_stats = load_preprocessing_stats()
        if preprocessing_stats is None:
            print("전처리 통계를 로드할 수 없습니다.")
            return

    total_preprocessing = sum(preprocessing_stats.values())

    print("\n구두점별 비교:")
    print(f"{'구두점':^6} | {'전처리 비율':>12} | {'학습데이터 비율':>15} | {'차이':>8}")
    print("-" * 60)

    for punct in punctuations:
        prep_count = preprocessing_stats.get(punct, 0)
        prep_ratio = prep_count / total_preprocessing * 100 if total_preprocessing > 0 else 0

        learn_count = all_punct_counter.get(punct, 0)
        learn_ratio = learn_count / total_puncts * 100 if total_puncts > 0 else 0

        diff = abs(prep_ratio - learn_ratio)

        # 차이가 큰 경우 강조
        flag = "⚠️" if diff > 1.0 else "✅"

        print(f"{punct:^6} | {prep_ratio:>11.2f}% | {learn_ratio:>14.2f}% | {diff:>7.2f}% {flag}")

    # 전체 통계 비교
    print(f"\n전체 통계:")
    print(f"- 전처리 총 구두점: {total_preprocessing:,}")
    print(f"- 학습데이터 샘플 구두점: {total_puncts:,}")


def main():
    """메인 함수"""
    # 파일 선택
    root = tk.Tk()
    root.withdraw()

    print("생성된 JSONL 파일을 선택하세요...")
    jsonl_path = filedialog.askopenfilename(
        title="train.jsonl 또는 val.jsonl 선택",
        filetypes=[("JSONL files", "*.jsonl")]
    )

    if not jsonl_path:
        print("파일이 선택되지 않았습니다.")
        return

    print(f"\n선택된 파일: {jsonl_path}")

    # 전체 파일 분석할지 샘플만 볼지 선택
    analysis_type = input("\n분석 유형 선택:\n1. 샘플만 확인 (빠름)\n2. 전체 파일 통계 분석 (느림)\n선택 [1/2]: ")

    if analysis_type == "2":
        # 전체 파일 통계 분석
        analyze_full_statistics(jsonl_path)
    else:
        # 샘플 개수 입력
        try:
            num_samples = int(input("\n확인할 샘플 개수 (기본값 10): ") or "10")
        except ValueError:
            num_samples = 10

        # 샘플 확인
        show_sample_reconstruction(jsonl_path, num_samples)

    print("\n\n✅ 분석 완료!")


def analyze_full_statistics(jsonl_path: str) -> None:
    """전체 파일의 구두점 통계 분석"""
    print(f"\n전체 파일 분석 중... (시간이 걸릴 수 있습니다)")

    total_chars = 0
    total_puncts = 0
    punct_counter = Counter()
    line_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="분석 중"):
            line_count += 1

            try:
                sample = json.loads(line)
                length = sample['n']
                total_chars += length

                # 구두점 카운트
                for indices in sample['l'][:length]:
                    for idx in indices:
                        if 0 <= idx < len(punctuations):
                            punct_counter[punctuations[idx]] += 1
                            total_puncts += 1

            except Exception as e:
                if line_count % 10000 == 0:  # 에러가 너무 많이 출력되지 않도록
                    print(f"경고: 줄 {line_count} 처리 오류: {e}")
                continue

    print(f"\n{'=' * 80}")
    print("📊 전체 파일 통계")
    print(f"{'=' * 80}")
    print(f"- 총 라인 수: {line_count:,}")
    print(f"- 총 한자 수: {total_chars:,}")
    print(f"- 총 구두점 수: {total_puncts:,}")
    print(f"- 평균 구두점 비율: {total_puncts / total_chars:.3f} (한자당)")

    print(f"\n구두점 분포:")
    for punct, count in punct_counter.most_common():
        print(f"  {punct}: {count:,}회 ({count / total_puncts * 100:.2f}%)")

    # 전처리 데이터와 비교
    compare_with_preprocessing_stats(punct_counter, total_puncts)

    # 구두점 수 차이 설명 추가
    print(f"\n{'=' * 80}")
    print("💡 구두점 수 차이 설명")
    print(f"{'=' * 80}")

    # 파일명에서 train/val 구분
    file_name = Path(jsonl_path).name
    if 'train' in file_name:
        print("\n⚠️  현재 train.jsonl만 분석 중입니다.")
        print("   전체 데이터의 약 90%만 포함되어 있습니다.")
        print("   정확한 비교를 위해서는 val.jsonl도 함께 분석해야 합니다.")
    elif 'val' in file_name:
        print("\n⚠️  현재 val.jsonl만 분석 중입니다.")
        print("   전체 데이터의 약 10%만 포함되어 있습니다.")

    print("\n전처리 데이터와 차이가 나는 이유:")
    print("1. 오버랩(overlap=50)으로 인한 데이터 증가")
    print("   - 512자보다 긴 텍스트가 여러 청크로 분할")
    print("   - 청크 간 50자씩 겹치면서 일부 구두점 중복 카운트")
    print("2. 예제 수: 약 8.86% 증가")
    print("3. 구두점 수: 약 3-4% 증가 예상")

if __name__ == "__main__":
    main()