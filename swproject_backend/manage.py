#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. manage.py가 있는 폴더(Backend/Django)를 경로에 추가
sys.path.insert(0, BASE_DIR) 
# 2. 프로젝트 설정 파일이 있는 폴더(swproject_backend)를 경로에 추가
sys.path.insert(0, os.path.join(BASE_DIR, 'swproject_backend'))
# ⬆️ ⚠️ 이 로직을 추가합니다.

def main():
    """Run administrative tasks."""
    # 이제 이 경로는 확실히 찾을 수 있어야 합니다.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'swproject_backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
