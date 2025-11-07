"""
Streamlit Cloud 진입점 파일

Streamlit Cloud는 기본적으로 루트의 streamlit_app.py를 찾지만,
이 프로젝트는 app/main.py를 사용합니다.

Streamlit Cloud 설정에서 Main file path를 'app/main.py'로 지정하거나,
이 파일을 진입점으로 사용할 수 있습니다.
"""

from app.main import main

if __name__ == "__main__":
    main()

