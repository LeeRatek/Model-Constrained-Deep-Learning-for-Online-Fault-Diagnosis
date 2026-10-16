"""mcdl 패키지: 여러 실행 스크립트가 함께 쓰는 공통 코드 모음.

이름의 뜻
    mcdl은 논문 제목 "Model-Constrained Deep Learning for Online Fault Diagnosis"의 앞 글자(Model-Constrained
    Deep Learning)를 딴 것이다. 폴더 하나에 .py 파일을 모아 두고 `from mcdl.run import Run`처럼 불러 쓰는
    파이썬 "패키지"이며, 이 __init__.py 파일이 있어야 폴더가 패키지로 인식된다.

왜 따로 모았나
    Train_.py, Test_.py, Test_multi_models.py, Test_tsne.py, data_filter.py 같은 실행 스크립트가 예전에는
    같은 계산(데이터 읽기, 재구성 오차, 진단 특징, PCA 관리도, ROC)을 각자 복사해 갖고 있었다. 한쪽만 고치면
    스크립트끼리 결과가 달라지는 문제가 있어서, 공통 계산을 여기에 한 벌만 두고 모든 스크립트가 불러 쓴다.

파일별 역할 (데이터가 흐르는 순서)
    preprocessing.py   원본 데이터 파일(.pkl)을 읽고 전처리(구간 자르기, 정규화)한다.
    vehicles.py        분할 목록(.npy)에서 차량 ID를 읽고, 차량 한 대 또는 여러 대의 전처리된 텐서를 돌려준다.
    run.py             학습 결과 폴더(models/<RUN_ID>/)를 읽고 쓴다. 설정 파일, 체크포인트, 모델 생성을 맡는다.
    reconstruction.py  학습된 모델로 차량 데이터를 재구성해 재구성 오차와 진단 특징을 만든다.
    diagnosis.py       재구성 오차를 진단 특징 6개로 요약하고, PCA 관리도로 T²·SPE·CI를 계산한다.
    chart_source.py    평가할 모델 쌍의 관리도를 어디서 가져올지(저장된 파일, 캐시, 다시 계산) 정한다.
    roc.py             차량별 최대 CI와 고장 여부로 ROC 곡선과 AUC를 계산한다.
    plotting.py        그림 함수 모음 (loss 곡선, ROC, 진단 시계열, t-SNE, AUC 히트맵 등).

자주 나오는 용어는 저장소 루트의 GLOSSARY.md에 풀어서 정리되어 있다.
"""
