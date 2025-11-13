"""
7종 구두점이 하나도 없는 행 제거 스크립트
7개 구두점(, 。 · ? ! 《 》)이 하나도 없는 행을 삭제
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def remove_lines_without_target_punctuation():
    """7종 구두점이 없는 행 제거 메인 함수"""

    print("=== 7종 구두점이 없는 행 제거 ===\n")
    print("대상 구두점: , 。 · ? ! 《 》\n")

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

    # 7종 구두점 정의
    target_puncts = [',', '。', '·', '?', '!', '《', '》']

    # 통계
    total_original_lines = 0
    total_removed_lines = 0
    total_kept_lines = 0
    files_with_changes = []

    print("처리 중...")

    for txt_file in tqdm(txt_files):
        try:
            # 파일 읽기
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            original_count = len(lines)
            total_original_lines += original_count

            # 유지할 줄 선택
            kept_lines = []
            file_removed = 0

            for line in lines:
                # 7종 구두점이 하나라도 있는지 확인
                has_target_punct = any(punct in line for punct in target_puncts)

                if has_target_punct:
                    # 7종 구두점이 있으면 유지
                    kept_lines.append(line)
                else:
                    # 7종 구두점이 없으면 제거
                    file_removed += 1

            total_removed_lines += file_removed
            total_kept_lines += len(kept_lines)

            # 변경사항이 있는 경우만 파일 저장
            if file_removed > 0:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.writelines(kept_lines)

                files_with_changes.append((txt_file.name, file_removed, original_count))

        except Exception as e:
            print(f"\n파일 오류 {txt_file.name}: {e}")
            continue

    print(f"\n완료!")
    print("=" * 50)
    print(f"처리된 파일: {len(txt_files)}개")
    print(f"변경된 파일: {len(files_with_changes)}개")
    print(f"\n원본 총 줄 수: {total_original_lines:,}줄")
    print(f"제거된 줄 수: {total_removed_lines:,}줄 ({total_removed_lines / total_original_lines * 100:.1f}%)")
    print(f"유지된 줄 수: {total_kept_lines:,}줄")

    # 많이 변경된 파일 상위 10개 표시
    if files_with_changes:
        print("\n[변경 내역 상위 10개 파일]")
        files_with_changes.sort(key=lambda x: x[1], reverse=True)
        for i, (filename, removed, original) in enumerate(files_with_changes[:10], 1):
            removal_rate = removed / original * 100
            print(f"  {i}. {filename}: {removed:,}줄 제거 ({removal_rate:.1f}%)")

    # 결과 저장
    save = input("\n처리 결과를 저장하시겠습니까? [y/n]: ")
    if save.lower() == 'y':
        save_path = filedialog.asksaveasfilename(
            title="처리 결과 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt")],
            initialfile="구두점 없는 행 제거_결과.txt",
            initialdir=Path.home() / "Desktop"
        )

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== 7종 구두점이 없는 행 제거 결과 ===\n\n")
                f.write(f"처리 폴더: {folder_path}\n")
                f.write(f"처리 시각: {Path(save_path).stem.split('_')[-1] if '_' in Path(save_path).stem else ''}\n\n")

                f.write("[대상 구두점]\n")
                f.write(", 。 · ? ! 《 》 (7종)\n\n")

                f.write("[전체 통계]\n")
                f.write(f"처리 파일: {len(txt_files)}개\n")
                f.write(f"변경 파일: {len(files_with_changes)}개\n")
                f.write(f"원본 줄 수: {total_original_lines:,}줄\n")
                f.write(f"제거 줄 수: {total_removed_lines:,}줄 ({total_removed_lines / total_original_lines * 100:.1f}%)\n")
                f.write(f"유지 줄 수: {total_kept_lines:,}줄\n\n")

                if files_with_changes:
                    f.write("[파일별 변경 내역]\n")
                    for filename, removed, original in files_with_changes:
                        removal_rate = removed / original * 100
                        f.write(f"  {filename}: {removed:,}/{original:,}줄 제거 ({removal_rate:.1f}%)\n")

                f.write("\n[처리 규칙]\n")
                f.write("- 7종 구두점(, 。 · ? ! 《 》) 중 하나라도 있는 행: 유지\n")
                f.write("- 7종 구두점이 하나도 없는 행: 제거\n")

            print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    remove_lines_without_target_punctuation()