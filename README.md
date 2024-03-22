<br />
<div align="center">
  <a href="https://github.com/AIKU-Official/Pick-Me-Thumbnail">
    <img src="project_logo/logo.png" alt="Logo" width="1000" height="300">
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

