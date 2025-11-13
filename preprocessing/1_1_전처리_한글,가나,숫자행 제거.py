"""
1단계 전처리: 한글/일본어/아라비아숫자가 포함된 줄 전체 삭제
- 선택한 폴더와 모든 하위 폴더의 txt 파일 처리 (os.walk 사용)
"""

import os
import re
import tkinter as tk
from tkinter import filedialog


def stage1_remove_lines():
    """1단계: 한글/일본어/아라비아숫자가 포함된 줄 삭제"""

    print("=== 1단계 전처리: 한글/일본어/아라비아숫자 포함 줄 삭제 ===")
    print("(하위 폴더 포함)\n")

    # tkinter 설정
    root = tk.Tk()
    root.withdraw()

    # 폴더 선택
    root_folder = filedialog.askdirectory(
        title="원본 txt 파일이 있는 폴더 선택 (하위 폴더 포함 처리)"
    )

    if not root_folder:
        print("폴더 선택이 취소되었습니다.")
        return

    print(f"선택된 폴더: {root_folder}")
    print("하위 폴더 검색 중...\n")

    # 모든 txt 파일 찾기 (os.walk 사용)
    txt_files = []
    folder_structure = {}

    for dirpath, dirnames, filenames in os.walk(root_folder):
        txt_in_folder = [f for f in filenames if f.endswith('.txt')]
        if txt_in_folder:
            # 상대 경로 계산
            rel_path = os.path.relpath(dirpath, root_folder)
            if rel_path == '.':
                rel_path = '루트'

            folder_structure[dirpath] = {
                'rel_path': rel_path,
                'files': txt_in_folder,
                'file_paths': [os.path.join(dirpath, f) for f in txt_in_folder]
            }
            txt_files.extend(folder_structure[dirpath]['file_paths'])

    if not txt_files:
        print("txt 파일이 없습니다!")
        return

    print(f"발견된 폴더: {len(folder_structure)}개")
    print(f"발견된 파일: {len(txt_files)}개")

    # 폴더 구조 출력
    print("\n폴더 구조:")
    for dirpath in sorted(folder_structure.keys()):
        info = folder_structure[dirpath]
        print(f"  📁 {info['rel_path']}/ ({len(info['files'])}개 파일)")
    print()

    # 패턴 정의
    hangul_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')  # 히라가나, 가타카나
    digit_pattern = re.compile(r'[0-9]')  # 아라비아 숫자만

    # 전체 통계
    stats = {
        'total_folders': len(folder_structure),
        'total_files': len(txt_files),
        'original_lines': 0,
        'removed_lines': 0,
        'removed_hangul': 0,
        'removed_japanese': 0,
        'removed_digit': 0,
        'removed_mixed': 0,
        'removed_empty': 0,
        'remaining_lines': 0,
        'folder_stats': {}  # 폴더별 통계
    }

    print("처리 중...")

    # 폴더별로 처리
    folder_idx = 0
    for dirpath in sorted(folder_structure.keys()):
        folder_idx += 1
        info = folder_structure[dirpath]

        # 폴더별 통계 초기화
        folder_stat = {
            'file_count': len(info['files']),
            'original_lines': 0,
            'removed_lines': 0,
            'remaining_lines': 0
        }

        print(f"\n폴더 처리 중 [{folder_idx}/{len(folder_structure)}]: {info['rel_path']}/")

        for file_idx, file_path in enumerate(info['file_paths']):
            file_name = os.path.basename(file_path)
            print(f"  파일 [{file_idx + 1}/{len(info['files'])}]: {file_name}", end=' ... ', flush=True)

            try:
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                original_count = len(lines)
                stats['original_lines'] += original_count
                folder_stat['original_lines'] += original_count

                # 필터링
                filtered_lines = []
                file_removed = 0

                for line in lines:
                    # 빈 줄 제거
                    if not line.strip():
                        stats['removed_empty'] += 1
                        stats['removed_lines'] += 1
                        file_removed += 1
                        continue

                    # 한글, 일본어, 아라비아 숫자 확인
                    has_hangul = bool(hangul_pattern.search(line))
                    has_japanese = bool(japanese_pattern.search(line))
                    has_digit = bool(digit_pattern.search(line))

                    # 제거 대상 판별
                    remove_count = sum([has_hangul, has_japanese, has_digit])

                    if remove_count == 0:
                        # 한글, 일본어, 숫자 모두 없는 경우만 유지
                        filtered_lines.append(line)
                        stats['remaining_lines'] += 1
                        folder_stat['remaining_lines'] += 1
                    else:
                        # 제거 대상 통계 업데이트
                        stats['removed_lines'] += 1
                        folder_stat['removed_lines'] += 1
                        file_removed += 1

                        if remove_count >= 2:
                            stats['removed_mixed'] += 1
                        elif has_hangul:
                            stats['removed_hangul'] += 1
                        elif has_japanese:
                            stats['removed_japanese'] += 1
                        elif has_digit:
                            stats['removed_digit'] += 1

                # 파일 덮어쓰기
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(filtered_lines)

                print(f"{file_removed}줄 삭제")

            except Exception as e:
                print(f"❌ 오류: {e}")
                continue

        # 폴더별 통계 저장
        stats['folder_stats'][info['rel_path']] = folder_stat

    print(f"\n\n✅ 처리 완료!")

    # 결과 출력
    print("\n" + "=" * 60)
    print("전체 처리 결과")
    print("=" * 60)
    print(f"처리된 폴더: {stats['total_folders']}개")
    print(f"처리된 파일: {stats['total_files']}개")
    print(f"원본 총 줄수: {stats['original_lines']:,}줄")

    if stats['original_lines'] > 0:
        print(f"삭제된 줄: {stats['removed_lines']:,}줄 ({stats['removed_lines'] / stats['original_lines'] * 100:.1f}%)")
        print(f"  - 한글만: {stats['removed_hangul']:,}줄")
        print(f"  - 일본어만: {stats['removed_japanese']:,}줄")
        print(f"  - 숫자만: {stats['removed_digit']:,}줄")
        print(f"  - 혼합(2개 이상): {stats['removed_mixed']:,}줄")
        print(f"  - 빈 줄: {stats['removed_empty']:,}줄")
        print(f"남은 줄수: {stats['remaining_lines']:,}줄 ({stats['remaining_lines'] / stats['original_lines'] * 100:.1f}%)")

    # 폴더별 요약
    if len(stats['folder_stats']) > 1:
        print("\n" + "=" * 60)
        print("폴더별 요약")
        print("=" * 60)
        for folder_name, folder_stat in sorted(stats['folder_stats'].items()):
            if folder_stat['original_lines'] > 0:
                removal_rate = folder_stat['removed_lines'] / folder_stat['original_lines'] * 100
                print(f"📁 {folder_name}/")
                print(f"   파일: {folder_stat['file_count']}개 | "
                      f"원본: {folder_stat['original_lines']:,}줄 | "
                      f"삭제: {folder_stat['removed_lines']:,}줄 ({removal_rate:.1f}%) | "
                      f"남음: {folder_stat['remaining_lines']:,}줄")

    # 결과 저장 옵션
    save = input("\n\n처리 결과를 저장하시겠습니까? [y/n]: ")
    if save.lower() == 'y':
        save_results(root_folder, stats)


def save_results(root_folder, stats):
    """처리 결과 저장"""
    # 저장 경로 선택
    root = tk.Tk()
    root.withdraw()

    save_path = filedialog.asksaveasfilename(
        title="처리 결과 저장",
        defaultextension=".txt",
        filetypes=[("텍스트 파일", "*.txt")],
        initialfile="1단계_전처리_결과.txt"
    )

    if not save_path:
        print("저장이 취소되었습니다.")
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== 1단계 전처리 결과 (하위 폴더 포함) ===\n\n")
        f.write(f"처리 루트 폴더: {root_folder}\n")
        f.write("=" * 50 + "\n\n")

        f.write("[전체 처리 결과]\n")
        f.write(f"처리된 폴더: {stats['total_folders']}개\n")
        f.write(f"처리된 파일: {stats['total_files']}개\n")
        f.write(f"원본 총 줄수: {stats['original_lines']:,}줄\n")

        if stats['original_lines'] > 0:
            f.write(
                f"삭제된 줄: {stats['removed_lines']:,}줄 ({stats['removed_lines'] / stats['original_lines'] * 100:.1f}%)\n")
            f.write(f"  - 한글만: {stats['removed_hangul']:,}줄\n")
            f.write(f"  - 일본어만: {stats['removed_japanese']:,}줄\n")
            f.write(f"  - 숫자만: {stats['removed_digit']:,}줄\n")
            f.write(f"  - 혼합(2개 이상): {stats['removed_mixed']:,}줄\n")
            f.write(f"  - 빈 줄: {stats['removed_empty']:,}줄\n")
            f.write(
                f"남은 줄수: {stats['remaining_lines']:,}줄 ({stats['remaining_lines'] / stats['original_lines'] * 100:.1f}%)\n")

        # 폴더별 상세 결과
        if len(stats['folder_stats']) > 1:
            f.write("\n[폴더별 상세]\n")
            f.write("-" * 50 + "\n")
            for folder_name, folder_stat in sorted(stats['folder_stats'].items()):
                if folder_stat['original_lines'] > 0:
                    removal_rate = folder_stat['removed_lines'] / folder_stat['original_lines'] * 100
                    f.write(f"\n📁 {folder_name}/\n")
                    f.write(f"  - 파일 수: {folder_stat['file_count']}개\n")
                    f.write(f"  - 원본 줄수: {folder_stat['original_lines']:,}줄\n")
                    f.write(f"  - 삭제 줄수: {folder_stat['removed_lines']:,}줄 ({removal_rate:.1f}%)\n")
                    f.write(f"  - 남은 줄수: {folder_stat['remaining_lines']:,}줄\n")

        # 예상 효과
        f.write("\n[예상 효과]\n")
        f.write("- 한글이 포함된 현대 텍스트 제거\n")
        f.write("- 일본어(히라가나, 가타카나)가 포함된 텍스트 제거\n")
        f.write("- 아라비아 숫자가 포함된 주석/번호 제거\n")
        f.write("- 순수 한문 텍스트만 남음\n")
        f.write("\n※ 주의: 원본 파일이 덮어쓰여졌습니다.\n")
        f.write("※ 필요시 백업본에서 복원하세요.\n")

    print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    stage1_remove_lines()