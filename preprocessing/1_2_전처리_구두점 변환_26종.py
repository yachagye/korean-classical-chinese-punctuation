"""
전처리 2단계 - 구두점 변환
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from collections import Counter


def stage2_conversion_only():
    """2단계 구두점 변환 메인 함수"""

    print("=== 전처리 2단계: 구두점 변환 ===\n")

    # tkinter 설정
    root = tk.Tk()
    root.withdraw()

    # 폴더 선택
    folder_path = filedialog.askdirectory(
        title="1단계 처리된 txt 파일이 있는 폴더 선택",
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

    # 2단계 실행
    print("=" * 50)
    print("2단계: 구두점 변환")
    print("=" * 50)
    stage2_stats = stage2_convert_punctuations(txt_files)

    # 결과 출력
    print(f"\n[2단계 결과]")
    print(f"처리 전 줄수: {stage2_stats['total_lines']:,}줄")
    print(f"처리 후 줄수: {stage2_stats['processed_lines']:,}줄")
    print(f"문자 변환:")
    for replacement, count in stage2_stats['replacements'].items():
        if count > 0:
            print(f"  - {replacement}: {count:,}회")

    # 결과 저장
    save_results(folder_path, stage2_stats)


def stage2_convert_punctuations(txt_files):
    """2단계: 구두점 변환"""

    # 통계
    stats = {
        'total_lines': 0,
        'processed_lines': 0,
        'replacements': Counter()
    }

    print("2단계 처리 중...")

    for file_idx, txt_file in enumerate(txt_files):
        print(f"\r처리 중: {file_idx + 1}/{len(txt_files)}", end='', flush=True)

        try:
            # 파일 읽기
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            stats['total_lines'] += len(lines)

            # 처리된 줄 저장
            processed_lines = []

            for line in lines:
                original_line = line.strip()

                # 이미 빈 줄은 건너뛰기
                if not original_line:
                    continue

                # 문자 단위로 처리
                processed_chars = []

                for char in original_line:
                    # 1. 구두점 통합 변환
                    # 전각 구두점을 기본 형태로 변환
                    if char == '，':  # U+FF0C
                        processed_chars.append(',')  # U+002C
                        stats['replacements']['，→,'] += 1
                    elif char == '｡':  # U+FF61
                        processed_chars.append('。')  # U+3002
                        stats['replacements']['｡→。'] += 1
                    elif char == '？':  # U+FF1F
                        processed_chars.append('?')  # U+003F
                        stats['replacements']['？→?'] += 1
                    elif char == '！':  # U+FF01
                        processed_chars.append('!')  # U+0021
                        stats['replacements']['！→!'] += 1
                    elif char == '：':  # U+FF1A
                        processed_chars.append(':')  # U+003A
                        stats['replacements']['：→:'] += 1
                    elif char == '；':  # U+FF1B
                        processed_chars.append(';')  # U+003B
                        stats['replacements']['；→;'] += 1
                    elif char == '）':  # U+FF09
                        processed_chars.append(')')  # U+0029
                        stats['replacements']['）→)'] += 1
                    elif char == '［':  # U+FF3B
                        processed_chars.append('[')  # U+005B
                        stats['replacements']['［→['] += 1
                    elif char == '］':  # U+FF3D
                        processed_chars.append(']')  # U+005D
                        stats['replacements']['］→]'] += 1
                    elif char == '｢':  # U+FF62
                        processed_chars.append('「')  # U+300C
                        stats['replacements']['｢→「'] += 1
                    elif char == '｣':  # U+FF63
                        processed_chars.append('」')  # U+300D
                        stats['replacements']['｣→」'] += 1

                    # 2. 중점 통합
                    elif char == '\u318D':  # HANGUL LETTER ARAEA (ㆍ)
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['ㆍ→·'] += 1
                    elif char == '\uFF65':  # HALFWIDTH KATAKANA MIDDLE DOT (･)
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['･→·'] += 1
                    elif char == '\u30FB':  # KATAKANA MIDDLE DOT (・)
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['・→·'] += 1
                    elif char == '\u2024':  # ONE DOT LEADER (․)
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['․→·'] += 1
                    elif char == '\u2027':  # HYPHENATION POINT (‧)
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['‧→·'] += 1
                    elif char == '\u2022':  # BULLET (•)
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['•→·'] += 1
                    elif char == '\u2219':  # BULLET OPERATOR (∙) ← 추가!
                        processed_chars.append('\u00B7')  # MIDDLE DOT
                        stats['replacements']['∙→·'] += 1

                    # 3. 마침표 통합
                    elif char == ".":  # U+002E
                        processed_chars.append("。")  # U+3002
                        stats["replacements"][".→。"] += 1
                    elif char == "．":  # U+FF0E
                        processed_chars.append("。")  # U+3002
                        stats["replacements"]["．→。"] += 1

                    # 4. 꺽쇠 괄호 통합
                    elif char == "『":  # U+300E
                        processed_chars.append("《")  # U+300A
                        stats["replacements"]["『→《"] += 1
                    elif char == "』":  # U+300F
                        processed_chars.append("》")  # U+300B
                        stats["replacements"]["』→》"] += 1

                    # 5. 곡선 큰따옴표 변환
                    elif char == '\u201C':  # "
                        processed_chars.append('"')
                        stats['replacements']['"→"'] += 1
                    elif char == '\u201D':  # "
                        processed_chars.append('"')
                        stats['replacements']['"→"'] += 1

                    # 6. 특수 문자 변환
                    elif char == 'ː':  # U+02D0
                        processed_chars.append(':')
                        stats['replacements']['ː→:'] += 1
                    elif char == 'ⴰ':  # U+2D30
                        processed_chars.append('。')
                        stats['replacements']['ⴰ→。'] += 1
                    elif char == '､':  # U+FF64
                        processed_chars.append('、')
                        stats['replacements']['､→、'] += 1

                    # 7. 그 외 문자는 그대로
                    else:
                        processed_chars.append(char)

                # 처리된 줄 저장
                processed_line = "".join(processed_chars)
                if processed_line:
                    processed_lines.append(processed_line + "\n")
                    stats["processed_lines"] += 1

            # 파일 덮어쓰기
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

        except Exception as e:
            print(f"\n❌ 파일 오류 {txt_file.name}: {e}")
            continue

    print(f"\n2단계 완료!")
    return stats


def save_results(folder_path, stats):
    """처리 결과 저장"""
    save = input("\n\n처리 결과를 저장하시겠습니까? [y/n]: ")
    if save.lower() != 'y':
        return

    # tkinter 설정
    root = tk.Tk()
    root.withdraw()

    save_path = filedialog.asksaveasfilename(
        title="처리 결과 저장",
        defaultextension=".txt",
        filetypes=[("텍스트 파일", "*.txt")],
        initialfile="전처리_2단계_결과.txt",
        initialdir=Path.home() / "Desktop"
    )

    if not save_path:
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== 전처리 2단계 결과 ===\n\n")
        f.write(f"처리 폴더: {folder_path}\n")
        f.write("=" * 50 + "\n\n")

        f.write("[2단계: 구두점 변환]\n")
        f.write(f"처리 전 줄수: {stats['total_lines']:,}줄\n")
        f.write(f"처리 후 줄수: {stats['processed_lines']:,}줄\n\n")

        f.write("문자 변환:\n")
        for replacement, count in stats['replacements'].items():
            if count > 0:
                f.write(f"  - {replacement}: {count:,}회\n")

    print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    stage2_conversion_only()