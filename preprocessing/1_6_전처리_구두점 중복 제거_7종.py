"""
구두점 중복 제거 및 패턴 변환 스크립트
1. 7개 구두점(, 。 · ? ! 《 》)이 각각 연속으로 나타나면 하나로 줄임
2. 특정 패턴 변환: 。, -> 。  /  。· -> 。
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from collections import Counter


def remove_duplicate_and_convert_patterns():
    """구두점 중복 제거 및 패턴 변환 메인 함수"""

    print("=== 구두점 중복 제거 및 패턴 변환 ===\n")
    print("1단계: 각 구두점별 중복 제거")
    print("2단계: 패턴 변환 (。, → 。 / 。· → 。)\n")

    # tkinter 설정
    root = tk.Tk()
    root.withdraw()

    # 폴더 선택
    folder_path = filedialog.askdirectory(
        title="처리할 txt 파일이 있는 폴더 선택",
        initialdir=Path.home() / "Desktop"
    )

    if not folder_path:
        print("폴더 선택이 취소되었습니다.")
        return

    folder_path = Path(folder_path)
    print(f"선택된 폴더: {folder_path}\n")

    # txt 파일 찾기
    txt_files = list(folder_path.glob("*.txt"))
    print(f"발견된 파일: {len(txt_files)}개\n")

    if not txt_files:
        print("txt 파일이 없습니다!")
        return

    # 중복 제거할 구두점 7개
    target_puncts = [',', '。', '·', '?', '!', '《', '》']

    # 통계
    total_removed = 0
    total_lines = 0
    punct_stats = Counter()
    pattern_stats = Counter()

    print("처리 중...")

    for txt_file in tqdm(txt_files):
        try:
            # 파일 읽기
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_lines += len(lines)
            processed_lines = []

            for line in lines:
                # 1단계: 중복 제거
                chars = list(line)
                i = 0
                result = []

                while i < len(chars):
                    if chars[i] in target_puncts:
                        # 같은 구두점이 연속으로 나타나는지 확인
                        current_punct = chars[i]
                        j = i + 1

                        # 같은 구두점만 세기
                        while j < len(chars) and chars[j] == current_punct:
                            j += 1

                        # 중복된 개수
                        duplicate_count = j - i

                        if duplicate_count > 1:
                            total_removed += duplicate_count - 1
                            punct_stats[current_punct] += duplicate_count - 1

                        # 하나만 추가
                        result.append(current_punct)
                        i = j
                    else:
                        result.append(chars[i])
                        i += 1

                # 2단계: 패턴 변환
                # result를 문자열로 변환 후 패턴 치환
                result_str = ''.join(result)

                # 。, -> 。 변환
                count_1 = result_str.count('。,')
                if count_1 > 0:
                    result_str = result_str.replace('。,', '。')
                    pattern_stats['。, → 。'] += count_1

                # 。· -> 。 변환
                count_2 = result_str.count('。·')
                if count_2 > 0:
                    result_str = result_str.replace('。·', '。')
                    pattern_stats['。· → 。'] += count_2

                processed_lines.append(result_str)

            # 파일 덮어쓰기
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

        except Exception as e:
            print(f"\n파일 오류 {txt_file.name}: {e}")
            continue

    print(f"\n완료!")
    print(f"처리된 파일: {len(txt_files)}개")
    print(f"처리된 줄: {total_lines:,}줄\n")

    print("=" * 50)
    print("【1단계 결과: 중복 제거】")
    print(f"제거된 중복 구두점: {total_removed:,}개")

    if punct_stats:
        print("\n구두점별 중복 제거 통계:")
        for punct, count in punct_stats.most_common():
            print(f"  {punct}: {count:,}개")

    print("\n" + "=" * 50)
    print("【2단계 결과: 패턴 변환】")

    if pattern_stats:
        total_patterns = sum(pattern_stats.values())
        print(f"변환된 패턴: {total_patterns:,}개")
        print("\n패턴별 변환 통계:")
        for pattern, count in pattern_stats.items():
            print(f"  {pattern}: {count:,}개")
    else:
        print("변환된 패턴이 없습니다.")

    # 결과 저장
    save = input("\n처리 결과를 저장하시겠습니까? [y/n]: ")
    if save.lower() == 'y':
        save_path = filedialog.asksaveasfilename(
            title="처리 결과 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt")],
            initialfile="구두점중복제거_패턴변환_결과.txt",
            initialdir=Path.home() / "Desktop"
        )

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== 구두점 중복 제거 및 패턴 변환 결과 ===\n\n")
                f.write(f"처리 폴더: {folder_path}\n")
                f.write(f"처리 파일: {len(txt_files)}개\n")
                f.write(f"처리 줄수: {total_lines:,}줄\n\n")

                f.write("[1단계: 중복 제거]\n")
                f.write(f"제거된 중복: {total_removed:,}개\n")
                if punct_stats:
                    f.write("\n구두점별 중복 제거:\n")
                    for punct, count in punct_stats.most_common():
                        f.write(f"  {punct}: {count:,}개\n")

                f.write("\n[2단계: 패턴 변환]\n")
                if pattern_stats:
                    total_patterns = sum(pattern_stats.values())
                    f.write(f"변환된 패턴: {total_patterns:,}개\n")
                    f.write("\n패턴별 변환:\n")
                    for pattern, count in pattern_stats.items():
                        f.write(f"  {pattern}: {count:,}개\n")
                else:
                    f.write("변환된 패턴 없음\n")

            print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    remove_duplicate_and_convert_patterns()