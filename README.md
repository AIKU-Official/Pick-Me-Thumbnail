<br />
<div align="center">
  <a href="https://github.com/AIKU-Official/Pick-Me-Thumbnail">
    <img src="project_logo/logo.png" alt="Logo" width="300" height="300">
  </a>

  <h3 align="center">GGAMZI-AI</h3>

  <p align="center">
    Generate Short-Form/Thumbnail with your own title!
    <!--<br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>-->
  </p>
</div>

## Project Info
* 고려대학교 딥러닝 학회 AIKU 2024년 겨울방학 프로젝트
* 개발 기간 : 2024.01 ~ 2023.03.07

## About The Project
* ThumNailed_It은 비디오의 제목만을 참고하여 비디오 썸네일/숏폼 영상을 자동으로 추출 및 생성하는 Moment Reitrieval 모델입니다.

<details>
  <summary>📌 Moment Retreival </summary>
  <ol>
    Moment retrieval can de defined as the task of "localizing moments in a video given a user query".
  </ol>
</details>

* 제목만 주어졌을 때 연관성이 높은 특정 frame 또는 특정 구간의 moment를 자동 추출하는 모델을 만들어보자는 동기에서 시작한 프로젝트입니다.
* 기존의 MR(Moment Retrieval)에서 사용한 text query는 특정 moment를 묘사하는 description의 성격이 강했지만, 본 프로젝트에서 압축성/주관성이 더 강한 '제목'을 text query로 사용합니다.
## Installation

## Modeling
* CG-DETR (CVPR 2023)을 차용하였습니다. <a href="https://github.com/wjun0830/CGDETR">Link</a>
    
* **Inference**
  * 입력 : 비디오(.mp4, .avi), 비디오 제목 text
 
## Contribution
 * **Frame-Text Score 분석 및 개선 시도**
   <br />
   - 모델에서 사용한 attention weights를 시각화하여 텍스트 토큰과 비디오 토큰 간의 연관성 분석
      - 명사나 앞쪽에 위치한 동사 위주로 상대적으로 높은 attention score를 반영하는 경향성을 보이기는 하나, 토큰 간 유의미한 차이는 확인할 수 없음
      - 특정 dummy token들만 attention score가 낮음 -> 어떠한 기준으로 query-excluding context를 담아내는 지 모호함
  - 시도 방법 :
    - Dummy token 모델에서 제외
    - Video token을 추출할 때 사용한 frame들로부터 image caption 생성
        - Frame 별 Image caption과 제목 간의 CSIM 계산하여  attention weight에 가중치 부여
        - Image caption의  text embedding과  video embedding을  fuse 하여 새로운 feature 추출
   
``` * **한글 데이터로 훈련하였을 때 발생한 문제점 및 해결 방식**
   <br />
   - 한글에 존재하지 않는 component 생성
     - 원인 분석
       - 모델 구조 중 최대 이분 매칭 알고리즘을 한글에 적용하여 원인을 분석 → overfitting으로 결론
       - 훈련 데이터에서 사용한 글꼴마다 미세하게  다른 스타일
         - 자음 모음을 이어 쓰거나 붙여 쓰는 style이 동시에 섞이는 경우
         - 글꼴 스타일에 따라 다른 ㅎ,ㅊ 작성 방식
     - 해결 방식 
       - 여러 component로 분류될 여지가 있는 feature를 포함하는 특이한 데이터셋(out of distribution dataset)을 과감히 배제
       - 훈련 데이터의 다양성을 낮추어 학습
     <br />
    - 스타일 반영도가 떨어짐
      - 원인 분석
        - 여러 ablation study와 experiment를 통해 확인하는 방식으로 진행
        - 여러 기준으로 데이터를 분류하여 experiment 진행 : 실제 손 글씨와 유사한 스타일, 곡선이 많은 스타일, 굵기, 기울기 등
      - 해결 방식 : 실제 손 글씨와 유사하고 곡선 위주의 데이터셋으로 훈련 시 style 반영도가 증가한다는 점을 발견 후 데이터셋을 재구성
       
     - 배경에 노이즈 포함
       - 해결 방식
         - 후처리 방식 고안 → morphological transformation, Alpha blending 기법 활용하여 해결

## Results
<div align="center">
    <img src="images/result.png" alt="Result" width="1000" height="500">
</div>
👉왼쪽이 reference images, 오른쪽이 생성 결과

## Team Members & Roles
- 박서현 : 프로젝트 주제 발제, 모델 조사 및 학습, 데이터셋 조사 및 분류,  후처리 등
- 오윤진 : 모델 조사 및 학습, 훈련 결과 시각화, 웹 데모 구성, 후처리 등
- 김지영 : 모델 조사 및 학습, 모델 조사 및 학습, 모델 추가 훈련, 데이터셋 조사 등
- 민재원 : 방향성 제시 및 자문`````````````
