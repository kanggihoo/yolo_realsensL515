yolov5 intelrealsense L515

# 모델 구동을 위해 꼭 필요한 것
## 디렉토리 
    1. data
    2. models
    3. runs (가중치 파일 들어가 있음)
    4. utils
## 파일
    1. eport.py
    2. box인식후 좌표 반환 코드
    3. requirements.txt

## requirements.txt 파일 생성
    - pip freeze > requirements.txt
## requirements.txt 설치 방법
    - pip install -r requirements.txt

## 가상환경 만들기 (venv 패키지 활용)
   - 가상환경 만들고 싶은 디렉토리로 이동
   - python -m venv (가상환경 파일 이름)
## 가상환경 활성화
   - 윈도우
     - 가상환경폴더/Scripts/activate 입력, python interpreter를 가상환경 이름으로 설정
   - 우분투
     - source 가상환경폴더/bin/activate 입력 , python interpreter를 가상환경 이름으로 설정
## 가상환경 비활성화 
   - deactivate


## 파이썬 코드 ROS로 옮길 때 참조사항
1. __init__() 부분은 건드리지 말것
2. def Model_config() 에서는 weights 파일 이름은 메일로 보낼 때 best.pt변경 및 모델 가중치가 저장된 config 디렉토리의 가중치 파일 이름 변경
3. def Aruco_detect() 부분의 depth_point[2] 부분 변경사항 있으면 변경
4. def Aruco_detect_reset() 부분은 없어도됨
5. 모델 반환되는 명칭 주의 "box", "pallet"
6. 반환되는 각도 float형태인지 확인
7. 현재 윈도우 환경에서의 def Run() 부분은 다름
