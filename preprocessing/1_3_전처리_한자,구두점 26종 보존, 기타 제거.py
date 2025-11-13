"""
2단계 후반부 전처리:
구두점 변환 완료 후 → 한자 및 구두점 보존 → 나머지 기호 제거
3단계: 빈 괄호 제거
"""

import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from collections import Counter
import unicodedata


def stage2_after_conversion_and_stage3():
    """2단계 후반부(구두점 변환 이후) + 3단계 전처리"""

    print("=== 전처리 2단계 후반부 + 3단계 ===\n")

    # tkinter 설정
    root = tk.Tk()
    root.withdraw()

    # 폴더 선택
    folder_path = filedialog.askdirectory(
        title="구두점 변환까지 완료된 txt 파일이 있는 폴더 선택",
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

    # 2단계 후반부 실행
    print("\n" + "=" * 50)
    print("2단계 후반부: 한자/구두점 보존, 불필요한 문자 제거")
    print("=" * 50)
    stage2_stats = stage2_clean_characters(txt_files)

    # 3단계 실행
    print("\n" + "=" * 50)
    print("3단계: 빈 괄호 제거")
    print("=" * 50)
    stage3_stats = stage3_remove_empty_brackets(txt_files)

    # 전체 결과 출력
    print("\n" + "=" * 50)
    print("전체 처리 결과")
    print("=" * 50)

    print(f"\n[2단계 후반부 결과]")
    print(f"처리 전 줄수: {stage2_stats['total_lines']:,}줄")
    print(f"처리 후 줄수: {stage2_stats['processed_lines']:,}줄")
    print(f"\n제거된 문자 종류 (상위 10개):")
    for char_type, count in stage2_stats['chars_removed'].most_common(10):
        print(f"  - {char_type}: {count:,}개")

    # 기타 문자 상위 5개 추가
    if "기타_상세" in stage2_stats:
        print(f"\n'기타'로 분류된 문자 상위 5개:")
        for char, count in stage2_stats["기타_상세"].most_common(5):
            try:
                name = unicodedata.name(char)
            except:
                name = "Unknown"
            print(f"  - '{char}' (U+{ord(char):04X}, {name}): {count:,}회")

    print(f"\n[3단계 결과]")
    print(f"처리 전 줄수: {stage3_stats['total_lines']:,}줄")
    print(f"처리 후 줄수: {stage3_stats['processed_lines']:,}줄")
    print(f"제거된 빈 괄호:")
    for bracket, count in stage3_stats['brackets_removed'].most_common():
        print(f"  - {bracket}: {count:,}개")

    # 결과 저장
    save_results(folder_path, stage2_stats, stage3_stats)


def stage2_clean_characters(txt_files):
    """2단계 후반부: 한자/구두점 보존, 불필요한 문자 제거"""

    # 26개 구두점 정의 (･ 제거)
    valid_punctuations = [
        ',', '。', '·', '?', '"', ':', '、', '/',
        '\u2018',  # LEFT SINGLE QUOTATION MARK
        '\u2019',  # RIGHT SINGLE QUOTATION MARK
        '〉', '〈', ']', '[', '》', '《',
        ';', '(', ')', '【', '】', '〔', '〕', '!',
        '「', '」'
    ]
    punct_set = set(valid_punctuations)

    # 통계
    stats = {
        'total_lines': 0,
        'processed_lines': 0,
        'empty_lines_after_clean': 0,
        'chars_removed': Counter(),
    }

    print("2단계 후반부 처리 중...")

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
                    # 구두점 변환은 이미 완료되었으므로 보존/제거만 판단

                    # 1. 한자인지 확인 (확장된 범위) - 보존
                    if is_chinese_char(char):
                        processed_chars.append(char)
                    # 2. 유효한 구두점인지 확인 - 보존
                    elif char in punct_set:
                        processed_chars.append(char)
                    # 3. 그 외 모든 문자는 제거
                    else:
                        category = unicodedata.category(char)
                        if category == 'Co':  # Private Use
                            stats["chars_removed"]["Unknown문자"] += 1
                        elif char in " \t\u00A0\u3000\uFEFF":  # 모든 공백류
                            stats["chars_removed"]["공백"] += 1
                        elif char == "〇":
                            stats["chars_removed"]["한자숫자(〇)"] += 1
                        elif re.match(r"[a-zA-Z]", char):
                            stats["chars_removed"]["알파벳"] += 1
                        elif char in "○□●◆◯△■▲☐◎▩▼":
                            stats["chars_removed"]["도형기호"] += 1
                        elif char == "�":
                            stats["chars_removed"]["깨진문자"] += 1
                        elif char == "-":  # 하이픈도 별도 분류
                            stats["chars_removed"]["하이픈"] += 1
                        else:
                            stats["chars_removed"]["기타"] += 1
                            # 기타 문자 상세 수집
                            if "기타_상세" not in stats:
                                stats["기타_상세"] = Counter()
                            stats["기타_상세"][char] += 1

                # 처리된 줄이 비어있지 않으면 추가
                processed_line = "".join(processed_chars)
                if processed_line:
                    processed_lines.append(processed_line + "\n")
                    stats["processed_lines"] += 1
                else:
                    stats["empty_lines_after_clean"] += 1

            # 파일 덮어쓰기
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

        except Exception as e:
            print(f"\n❌ 파일 오류 {txt_file.name}: {e}")
            continue

    print(f"\n2단계 후반부 완료: {sum(stats['chars_removed'].values()):,}개 문자 제거")
    return stats


def stage3_remove_empty_brackets(txt_files):
    """3단계: 빈 괄호 제거"""

    # 제거할 빈 괄호 목록
    empty_brackets = [
        '【】', '〔〕', '[]', '()',
        '《》', '〈〉', '「」', "''", '""'
    ]

    # 통계
    stats = {
        'total_lines': 0,
        'processed_lines': 0,
        'brackets_removed': Counter()
    }

    print("3단계 처리 중...")

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
                processed_line = line.strip()

                # 빈 줄은 그대로
                if not processed_line:
                    processed_lines.append(line)
                    continue

                # 각 빈 괄호 제거
                for bracket in empty_brackets:
                    count_before = processed_line.count(bracket)
                    if count_before > 0:
                        processed_line = processed_line.replace(bracket, '')
                        stats['brackets_removed'][bracket] += count_before

                # 처리 후에도 내용이 있으면 추가
                if processed_line:
                    processed_lines.append(processed_line + '\n')
                    stats['processed_lines'] += 1

            # 파일 덮어쓰기
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

        except Exception as e:
            print(f"\n❌ 파일 오류 {txt_file.name}: {e}")
            continue

    print(f"\n3단계 완료: {sum(stats['brackets_removed'].values()):,}개 빈 괄호 제거")
    return stats


def is_chinese_char(char):
    """한자 판별 - 모든 CJK 영역 포함 (Compatibility 추가)"""
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
            0xF900 <= code <= 0xFAFF or  # CJK Compatibility Ideographs
            0xFA00 <= code <= 0xFA6F or  # CJK Compatibility Ideographs
            0xFA70 <= code <= 0xFADF or  # CJK Compatibility Ideographs
            0x2F00 <= code <= 0x2FDF or  # Kangxi Radicals
            0x2E80 <= code <= 0x2EFF)  # CJK Radicals Supplement


def save_results(folder_path, stage2_stats, stage3_stats):
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
        initialfile="2단계후반_3단계_전처리_결과.txt",
        initialdir=Path.home() / "Desktop"
    )

    if not save_path:
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== 2단계 후반부 + 3단계 전처리 결과 ===\n\n")
        f.write(f"처리 폴더: {folder_path}\n")
        f.write("=" * 50 + "\n\n")

        f.write("[2단계 후반부: 한자/구두점 보존, 불필요한 문자 제거]\n")
        f.write(f"처리 전 줄수: {stage2_stats['total_lines']:,}줄\n")
        f.write(f"처리 후 줄수: {stage2_stats['processed_lines']:,}줄\n")
        f.write(f"빈 줄로 변환: {stage2_stats['empty_lines_after_clean']:,}줄\n\n")

        f.write(f"제거된 문자: 총 {sum(stage2_stats['chars_removed'].values()):,}개\n")
        for char_type, count in stage2_stats['chars_removed'].most_common():
            f.write(f"  - {char_type}: {count:,}개\n")

        if "기타_상세" in stage2_stats:
            f.write("\n'기타'로 분류된 문자 상위 5개:\n")
            for char, count in stage2_stats["기타_상세"].most_common(5):
                try:
                    name = unicodedata.name(char)
                except:
                    name = "Unknown"
                f.write(f"  - '{char}' (U+{ord(char):04X}, {name}): {count:,}회\n")

        f.write("\n[3단계: 빈 괄호 제거]\n")
        f.write(f"처리 전 줄수: {stage3_stats['total_lines']:,}줄\n")
        f.write(f"처리 후 줄수: {stage3_stats['processed_lines']:,}줄\n")
        f.write(f"\n제거된 빈 괄호: 총 {sum(stage3_stats['brackets_removed'].values()):,}개\n")
        for bracket, count in stage3_stats['brackets_removed'].most_common():
            f.write(f"  - {bracket}: {count:,}개\n")

    print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    stage2_after_conversion_and_stage3()