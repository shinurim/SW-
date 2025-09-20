# SW-

# SW- 프로젝트

## Development Environment
- Python 3.9.13
- Virtual Environment: venv
- Dependencies: requirements.txt 참조

---

## Setup Guide

### 1. Clone repository
```bash
git clone https://github.com/shinurim/SW-.git
cd SW-

```
### 2. Create & Activate Virtual Environment + Install Dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
### 3. REACT
```bash
1. 프로젝트 환경설정(vite를 활용한 React 설치): npm install vite@latest

2. React 중앙집중식 상태관리 라이브러리 Recoil 설치: npm install recoil

3. 외부 오픈 API 통신을 위한 라이브러리 Axios 설치: npm install axios

4. CSS 스타일링을 위한 SASS/SCSS 설치: npm install -D sass

5. React Router 설치: npm install react-router-dom localforage match-sorter sort-by

6. TypeScript에서 Node.js 모듈을 쓸 수 있는 환경 구축 : npm i @types/node

7. React Toast Popup 모듈 설치 : npm install react-simple-toasts
```

### 3. Django
```bash
cd Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
