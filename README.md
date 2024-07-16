# Information_Retrieval_Study
- Information Retrieval 스터디 자료들을 담고 있습니다.
- 여러 Sparse/Dense/Generative Retrieval의 모델들을 공부한 뒤, 실제 inference/evaluation 하는 과정을 포함하고 있습니다.
- 메모리 상의 이유로, NQ320K dataset 하의 실험을 진행하였습니다.

## Used dataset
### NQ320K dataset을 사용하였습니다. 구체적인 사항은 다음과 같습니다.
- [GLEN](https://github.com/skleee/GLEN?tab=readme-ov-file) 다음 링크에 있는 NQ320k dataset을 사용하였습니다.
- DOC_NQ_first64.tsv, GTQ_NQ_dev.tsv, GTQ_NQ_train.tsv 파일을 다운로드하여 data 폴더 아래에 위치하세요.
- 메모리 상의 이유로 문서의 첫 64token만 존재하는 DOC_NQ_first64.tsv만 사용하여 검색을 진행하였습니다.

## Code Format
### 기본적으로 다음의 과정으로 코드가 구성됩니다.
- sparse / dense 여부에 따라 폴더가 구분됩니다.
- 각 모델 및 방법론을 하나의 ipynb 파일로 실험합니다. 만약 복잡한 코드가 있을 경우, 추가적으로 폴더를 만든 뒤 이를 사용합니다.
- ex) BM25.ipynb 형태를 기본적으로 사용하며, 해당 ipynb에 필요한 코드들은 BM25/* 폴더에 저장합니다.
- 최종적으로 BM25.ipynb 파일은 각 모델의 예측 결과를 data/inference/BM25.json 형태로 저장해야 합니다.
- Summary.ipynb 파일은 data/inference 내부의 모든 모델 성능을 종합적으로 확인할 수 있는 코드입니다.

## Update Timeline
### 2024.07.16. (Jonghyo)
- Information Retrieval의 기본 모델들 및 baseline code들을 업데이트 하였습니다.
- BM25 (Sparse)
- SPLADE (Sparse)
- DPR (Dense)