"""
build_with_checkpoint.py - 체크포인트 포함 EXE 빌드
"""

import PyInstaller.__main__
import os
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def prepare_checkpoint():
    """체크포인트 파일 선택 및 준비"""

    print("\n1. 체크포인트 파일(.ckpt)을 선택하세요...")

    # tkinter 파일 다이얼로그
    root = tk.Tk()
    root.withdraw()

    checkpoint_path = filedialog.askopenfilename(
        title="체크포인트 파일 선택",
        filetypes=[
            ("Checkpoint files", "*.ckpt"),
            ("All files", "*.*")
        ],
        initialdir=Path.home() / "Desktop"
    )
    root.destroy()

    if not checkpoint_path:
        print("❌ 파일이 선택되지 않았습니다.")
        return None

    if not os.path.exists(checkpoint_path):
        print(f"❌ 오류: {checkpoint_path} 파일을 찾을 수 없습니다")
        return None

    print(f"✅ 선택된 파일: {Path(checkpoint_path).name}")

    # model 폴더 생성
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    # 체크포인트 복사
    target_path = model_dir / "checkpoint.ckpt"
    file_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"   파일 크기: {file_size_mb:.1f} MB")
    print(f"   복사 중...")

    shutil.copy2(checkpoint_path, target_path)
    print(f"✅ 복사 완료!")

    return str(target_path)


def select_output_path():
    """EXE 파일 저장 경로 선택"""

    print("\n2. EXE 파일을 저장할 폴더를 선택하세요...")

    root = tk.Tk()
    root.withdraw()

    output_dir = filedialog.askdirectory(
        title="EXE 파일 저장 폴더 선택",
        initialdir=Path.home() / "Desktop"
    )
    root.destroy()

    if not output_dir:
        # 선택하지 않으면 현재 폴더의 dist 사용
        print("   기본 경로 사용: ./dist")
        return None

    print(f"✅ 저장 폴더: {output_dir}")
    return Path(output_dir)


def check_required_files():
    """필수 파일 확인"""
    required_files = [
        "구두점_지정_실행파일_GUI.py",
        "구두점7_추론모델.py",
        "구두점7_지정_txt.py",
        "구두점7_지정_csv.py"
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("\n❌ 필수 파일이 없습니다:")
        for file in missing:
            print(f"   - {file}")
        return False

    print("✅ 필수 파일 확인 완료")
    return True


def build_exe_with_checkpoint():
    """체크포인트 포함 EXE 빌드"""

    # 빌드 설정
    app_name = "한문구두점추론"
    main_script = "구두점_지정_실행파일_GUI.py"

    print("=" * 60)
    print("체크포인트 통합 EXE 빌드")
    print("=" * 60)

    # 필수 파일 확인
    if not check_required_files():
        return

    # 체크포인트 준비
    checkpoint = prepare_checkpoint()
    if not checkpoint:
        return

    # 출력 경로 선택
    custom_output_dir = select_output_path()

    # 빌드 폴더 정리
    print("\n이전 빌드 정리 중...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # PyInstaller spec 파일 생성 - scipy 포함 버전
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['{main_script}'],
    pathex=[],
    binaries=[],
    datas=[
        ('model/checkpoint.ckpt', 'model'),
        ('구두점7_추론모델.py', '.'),
        ('구두점7_지정_txt.py', '.'),
        ('구두점7_지정_csv.py', '.'),
    ],
    hiddenimports=[
        'torch',
        'torch._C',
        'torch._C._dynamo',
        'transformers',
        'transformers.generation',
        'transformers.generation.utils',
        'pandas',
        'numpy',
        'scipy',  # scipy 추가
        'scipy.sparse',
        'scipy.spatial',
        'scipy.special',
        'scipy.stats',
        'sklearn',
        'sklearn.metrics',
        'sklearn.utils',
        'tqdm',
        'regex',
        'sacremoses',
        'sentencepiece',
        'tiktoken',
        'tkinter',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'notebook',
        'jupyter',
        'ipython',
        # scipy 제외 목록에서 삭제
        'PIL',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
"""

    # spec 파일 저장
    spec_file = f"{app_name}.spec"
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(spec_content)

    print("\n🔨 EXE 빌드 중...")
    print(f"  - 앱 이름: {app_name}")
    print(f"  - 체크포인트 포함: {os.path.getsize('model/checkpoint.ckpt') / 1024 / 1024:.1f} MB")

    # PyInstaller 실행
    PyInstaller.__main__.run([
        spec_file,
        '--noconfirm',
        '--clean'
    ])

    # 기본 빌드 결과 확인
    default_exe_path = Path('dist') / f"{app_name}.exe"

    if default_exe_path.exists():
        # 사용자가 지정한 경로로 이동
        if custom_output_dir:
            final_exe_path = custom_output_dir / f"{app_name}.exe"

            # 이미 있으면 덮어쓸지 확인
            if final_exe_path.exists():
                overwrite = input(f"\n'{final_exe_path.name}'이 이미 존재합니다. 덮어쓰시겠습니까? (y/n) [y]: ").strip().lower()
                if overwrite == 'n':
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_exe_path = custom_output_dir / f"{app_name}_{timestamp}.exe"
                    print(f"새 이름으로 저장: {final_exe_path.name}")

            print(f"파일 이동 중: {custom_output_dir}")
            shutil.move(str(default_exe_path), str(final_exe_path))
            exe_path = final_exe_path
        else:
            exe_path = default_exe_path

        exe_size = exe_path.stat().st_size / 1024 / 1024

        print("\n" + "=" * 60)
        print("✅ 빌드 성공!")
        print("=" * 60)
        print(f"실행 파일: {exe_path.absolute()}")
        print(f"파일 크기: {exe_size:.1f} MB")
        print("\n특징:")
        print("  - 체크포인트 파일 내장")
        print("  - Python 설치 불필요")
        print("  - 단일 실행 파일")
        print("\n사용법:")
        print("  1. exe 파일 실행")
        print("  2. 처리할 파일/폴더 선택")
        print("  3. 처리 시작 클릭")

        # 임시 파일만 정리
        print("\n" + "-" * 40)
        clean = input("빌드 임시 파일을 삭제하시겠습니까? (y/n) [y]: ").strip().lower()
        if clean != 'n':
            print("빌드 임시 파일 삭제 중...")

            # build 폴더 삭제
            if Path('build').exists():
                shutil.rmtree('build')
                print("  ✓ build 폴더 삭제")

            # dist 폴더 삭제 (사용자 지정 경로로 이동한 경우만)
            if custom_output_dir and Path('dist').exists():
                shutil.rmtree('dist')
                print("  ✓ dist 폴더 삭제")

            # model 폴더 삭제 (임시 복사본)
            if Path('model').exists():
                shutil.rmtree('model')
                print("  ✓ model 폴더 삭제")

            # spec 파일 삭제
            if os.path.exists(spec_file):
                os.remove(spec_file)
                print(f"  ✓ {spec_file} 삭제")

            print("✅ 임시 파일 정리 완료")
            print("\n[유지된 파일]")
            print("  - 모든 Python 소스 파일 (.py)")
            print("  - 원본 체크포인트 파일 (.ckpt)")
            if not custom_output_dir:
                print("  - dist 폴더의 EXE 파일")

        # 폴더 열기 옵션
        print("\n" + "-" * 40)
        open_folder = input("저장 폴더를 열어보시겠습니까? (y/n) [y]: ").strip().lower()
        if open_folder != 'n':
            import platform
            if platform.system() == 'Windows':
                os.startfile(exe_path.parent)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{exe_path.parent}"')
            else:  # Linux
                os.system(f'xdg-open "{exe_path.parent}"')

        print("\n" + "=" * 60)
        print("빌드가 완료되었습니다!")
        print(f"실행 파일 위치: {exe_path.absolute()}")
        print("=" * 60)

    else:
        print("\n❌ 빌드 실패")
        print("오류를 확인하고 다시 시도하세요")

        # 실패시에도 임시 파일만 정리
        print("\n임시 파일 정리 중...")
        if Path('build').exists():
            shutil.rmtree('build')
        if Path('model').exists():
            shutil.rmtree('model')
        if os.path.exists(spec_file):
            os.remove(spec_file)


if __name__ == "__main__":
    try:
        build_exe_with_checkpoint()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 취소되었습니다.")
        # 취소시 임시 파일만 정리
        print("임시 파일 정리 중...")
        if Path('build').exists():
            shutil.rmtree('build', ignore_errors=True)
        if Path('dist').exists():
            shutil.rmtree('dist', ignore_errors=True)
        if Path('model').exists():
            shutil.rmtree('model', ignore_errors=True)
        # spec 파일 삭제
        for spec in Path('.').glob('*.spec'):
            try:
                spec.unlink()
            except:
                pass  # 오류 무시
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    input("\n엔터를 누르면 종료합니다...")