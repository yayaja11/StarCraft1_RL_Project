# Starcraft1 RL Project "Random Bunker Defense" 

##### 시작에 앞서, 본 프로젝트는 [TorchCraft][torch], [BWAPI][bwapi], [SAIDA][saida]의 코드를 참고하여 만들어졌습니다. 

##
##
## 목차
### [환경 설정 및 구조](#환경-설정-및-구조)
### [설계 이슈](#설계-이슈)
### [모델 이슈](#)
##
####
####
####
####
####
####
####
####
####
####
####
####
####
####
####
##

####  환경 설정 및 구조
- BWAPI
- Torchcraft
- Agent

<img src="https://user-images.githubusercontent.com/19571027/159266080-844e7d50-e479-4fa2-adbe-f26aa9cd9aa9.png" width="700" height="300"/>

#### 환경 설계 
- 무엇을 풀 것인가? 
  - 랜덤 벙커 디펜스6 
    - 35 스테이지로 이루어진 디펜스 게임
    - 플레이어는 벙커를 적재적소에 건설하고 업그레이드를 통해 마지막까지 살아남는 것이 목적
    - 벙커는 영웅벙커와 일반벙커가 있음
    - 가끔씩 벙커를 건설하면 추가 금액(운)을 줌. 
    - 일반적으로 운이라는 요소가 게임 마지막을 클리어하는데 중요함!
 
- 평범한 스타 유저들은 게임을 잘 할까?

![Brood-War-2022-03-21-22-43-56](https://user-images.githubusercontent.com/19571027/159280610-f2e81cc5-50de-44ec-93e3-c890538f4ef3.gif)
#### -> 26스테이지 사망

![2](https://user-images.githubusercontent.com/19571027/159283649-370b0cab-39a2-41cf-9af1-7fa236118888.gif)
#### -> 21스테이지 사망

##### 
