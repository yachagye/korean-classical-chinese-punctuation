"""
구두점 변환 스크립트 - 7개 구두점 명시 버전
- 유지: , 。 · ? ! 《 》 (7개)
- 변환: : ; → , / → 。 、 → · (4개)
- 삭제: 나머지 15개 구두점
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from collections import Counter


def process_punctuation():
    """구두점 변환 메인 함수"""

    print("=== 구두점 변환 (7개 구두점 명시) ===\n")

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

    # 26개 구두점 전체 정의 및 처리 방법
    punctuation_rules = {
        # 유지 (7개)
        ',': 'keep',  # U+002C
        '。': 'keep',  # U+3002
        '·': 'keep',  # U+00B7
        '?': 'keep',  # U+003F
        '!': 'keep',  # U+0021
        '《': 'keep',  # U+300A
        '》': 'keep',  # U+300B

        # 변환 (4개)
        ':': ',',  # U+003A → U+002C
        ';': ',',  # U+003B → U+002C
        '、': '·',  # U+3001 → U+00B7
        '/': '。',  # U+002F → U+3002

        # 삭제 (15개)
        '"': 'delete',  # U+0022
        chr(0x2018): 'delete',  # U+2018 '
        chr(0x2019): 'delete',  # U+2019 '
        ']': 'delete',  # U+005D
        '[': 'delete',  # U+005B
        '(': 'delete',  # U+0028
        ')': 'delete',  # U+0029
        '【': 'delete',  # U+3010
        '】': 'delete',  # U+3011
        '〉': 'delete',  # U+3009
        '〈': 'delete',  # U+3008
        '〔': 'delete',  # U+3014
        '〕': 'delete',  # U+3015
        '「': 'delete',  # U+300C
        '」': 'delete',  # U+300D
    }

    # 26개 확인
    print(f"정의된 구두점: {len(punctuation_rules)}개")
    print(f"- 유지: {sum(1 for v in punctuation_rules.values() if v == 'keep')}개")
    print(f"- 변환: {sum(1 for v in punctuation_rules.values() if v not in ['keep', 'delete'])}개")
    print(f"- 삭제: {sum(1 for v in punctuation_rules.values() if v == 'delete')}개\n")

    # 통계
    stats = Counter()
    total_lines = 0

    print("처리 중...")

    for txt_file in tqdm(txt_files):
        try:
            # 파일 읽기
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_lines += len(lines)
            processed_lines = []

            for line in lines:
                processed_chars = []

                for char in line:
                    # 26개 구두점에 포함되는지 확인
                    if char in punctuation_rules:
                        action = punctuation_rules[char]

                        if action == 'keep':
                            # 유지
                            processed_chars.append(char)
                            stats[f'유지_{char}'] += 1
                        elif action == 'delete':
                            # 삭제 (아무것도 추가하지 않음)
                            stats[f'삭제_{char}'] += 1
                        else:
                            # 변환
                            processed_chars.append(action)
                            stats[f'변환_{char}→{action}'] += 1
                    else:
                        # 26개 구두점이 아니면 그대로 유지
                        processed_chars.append(char)

                processed_lines.append(''.join(processed_chars))

            # 파일 덮어쓰기
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)

        except Exception as e:
            print(f"\n파일 오류 {txt_file.name}: {e}")
            continue

    # 결과 출력
    print(f"\n완료!")
    print(f"처리된 파일: {len(txt_files)}개")
    print(f"처리된 줄: {total_lines:,}줄\n")

    print("처리 통계:")
    print("-" * 40)

    # 유지 통계
    keep_total = sum(v for k, v in stats.items() if k.startswith('유지_'))
    if keep_total:
        print(f"[유지] 총 {keep_total:,}개")
        for k, v in sorted(stats.items()):
            if k.startswith('유지_'):
                punct = k.replace('유지_', '')
                print(f"  {punct}: {v:,}개")

    # 변환 통계
    convert_total = sum(v for k, v in stats.items() if k.startswith('변환_'))
    if convert_total:
        print(f"\n[변환] 총 {convert_total:,}개")
        for k, v in sorted(stats.items()):
            if k.startswith('변환_'):
                conversion = k.replace('변환_', '')
                print(f"  {conversion}: {v:,}개")

    # 삭제 통계
    delete_total = sum(v for k, v in stats.items() if k.startswith('삭제_'))
    if delete_total:
        print(f"\n[삭제] 총 {delete_total:,}개")
        for k, v in sorted(stats.items()):
            if k.startswith('삭제_'):
                punct = k.replace('삭제_', '')
                print(f"  {punct}: {v:,}개")

    # 결과 저장
    save = input("\n처리 결과를 저장하시겠습니까? [y/n]: ")
    if save.lower() == 'y':
        save_path = filedialog.asksaveasfilename(
            title="처리 결과 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt")],
            initialfile="구두점변환_결과.txt",
            initialdir=Path.home() / "Desktop"
        )

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== 구두점 변환 결과 (26개 구두점) ===\n\n")
                f.write(f"처리 폴더: {folder_path}\n")
                f.write(f"처리 파일: {len(txt_files)}개\n")
                f.write(f"처리 줄수: {total_lines:,}줄\n\n")

                f.write("[처리 규칙]\n")
                f.write("유지 (7개): , 。 · ? ! 《 》\n")
                f.write("변환 (4개): : ; → ,  / → 。  、 → ·\n")
                f.write("삭제 (15개): \" ' ' ] [ ( ) 【 】 〉 〈 〔 〕 「 」\n\n")

                f.write("[처리 통계]\n")
                f.write(f"유지: {keep_total:,}개\n")
                f.write(f"변환: {convert_total:,}개\n")
                f.write(f"삭제: {delete_total:,}개\n\n")

                f.write("[상세 통계]\n")
                for k, v in sorted(stats.items()):
                    f.write(f"{k}: {v:,}개\n")

            print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    process_punctuation()