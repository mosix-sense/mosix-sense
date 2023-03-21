싸이킷런패키지의 load_digits, load_wine, load_breath_cancer 데이터셋에 대한
의사결정트리, 로지스틱회귀, SGD분류기, SVM 모델들을 각각 적용하여 평가한결과
다음과 같이 판단하였습니다.


<손글씨>
가장 중요한 지표는 정확성(Accuracy)
이유:모든 클래스가 동등한 중요도를 가지고 있기 때문
전체샘플대비 예측성공샘플의 비율을 나타내는 정확성 지표가 재현율(Recall)보다 중요하다.

가장 높은 정확도를 가진 모델은 SVM
따라서 SVM이 load_digits를 분류하는데 있어 적용한모델 중 가장 효과적인 모델이다.



<와인>
가장 중요한 지표는 
가장 중요한 지표는 정확성(Accuracy)
모든 클래스(와인 종류)가 동등한 중요도를 가지고 있기 때문
전체샘플대비 예측성공샘플의 비율을 나타내는 정확성 지표가 재현율(Recall)보다 중요하다.

의사결정트리를 제외한 모든 분류모델에서 높은 학습율이 나옴
과적합을 의심해볼만함


<유방암>
가장중요한 지표는 재현율(Recall)
이유: 분류기가 악성 종양을 놓치면 치명적인 상황이 발생할 수 있으므로

모든 분류모델들이 높은 지표값을 기록했으나 그중에서 의사결정트리는 상대적으로
지표값이 가장 밀림


- README.md
- code/
  - decision_tree/
    - decision_tree_digits.py
    - decision_tree_wine.py
    - decision_tree_breast_cancer.py
  - logistic_regression/
    - logistic_regression_digits.py
    - logistic_regression_wine.py
    - logistic_regression_breast_cancer.py
  - sgd_classifier/
    - sgd_classifier_digits.py
    - sgd_classifier_wine.py
    - sgd_classifier_breast_cancer.py
  - svm/
    - svm_digits.py
    - svm_wine.py
    - svm_breast_cancer.py
- data/
  - digits/
    - digits.csv
  - wine/
    - wine.csv
  - breast_cancer/
    - breast_cancer.csv
- results/
  - decision_tree/
    - decision_tree_digits_results.csv
    - decision_tree_wine_results.csv
    - decision_tree_breast_cancer_results.csv
  - logistic_regression/
    - logistic_regression_digits_results.csv
    - logistic_regression_wine_results.csv
    - logistic_regression_breast_cancer_results.csv
  - sgd_classifier/
    - sgd_classifier_digits_results.csv
    - sgd_classifier_wine_results.csv
    - sgd_classifier_breast_cancer_results.csv
  - svm/
    - svm_digits_results.csv
    - svm_wine_results.csv
    - svm_breast_cancer_results.csv