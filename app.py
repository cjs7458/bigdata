
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    st.sidebar.success('나눔고딕 폰트 설정 완료')
else:
    st.sidebar.warning(f'나눔고딕 폰트 파일을 찾을 수 없습니다. 한글이 깨질 수 있습니다. 경로: {font_path}')


# Streamlit 앱 제목
st.title('생애 기대 수명 예측 및 모델 비교 대시보드')

# --- 데이터 및 모델 로드 ---
st.sidebar.header('데이터 및 모델 로드 상태')
try:
    linear_model = joblib.load('linear_regression_model.pkl')
    poly_model = joblib.load('polynomial_regression_model.pkl')
    ridge_model = joblib.load('ridge_regression_model.pkl')
    st.sidebar.success('모델 로드 성공!')

    X_train_50 = joblib.load('X_train_50.pkl')
    y_train_50 = joblib.load('y_train_50.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    st.sidebar.success('데이터 로드 성공!')

except Exception as e:
    st.sidebar.error(f'모델 또는 데이터 로드 실패: {e}')
    st.stop() # 로드 실패 시 앱 중단

models = {
    'Linear Model': linear_model,
    'Polynomial Model': poly_model,
    'Ridge Model': ridge_model
}

# --- 1. 모델 성능 비교 섹션 ([조건 3]) ---
st.header('모델 성능 비교')

performance_metrics = []

for name, model in models.items():
    # 훈련 데이터 예측 및 평가
    y_train_pred = model.predict(X_train_50)
    train_r2 = r2_score(y_train_50, y_train_pred)
    train_mse = mean_squared_error(y_train_50, y_train_pred)

    # 테스트 데이터 예측 및 평가
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # 모델 복잡성 (변수 변환 후 특성 개수)
    complexity = X_train_50.shape[1] # 기본 특성 개수
    if 'poly_features' in model.named_steps:
        # PolynomialFeatures를 거친 후의 특성 개수를 계산
        # StandardScaler가 적용된 후의 데이터에 대해 계산해야 정확
        temp_scaler = model.named_steps['scaler']
        X_scaled_temp = temp_scaler.transform(X_train_50)

        poly_transformer = model.named_steps['poly_features']
        complexity = poly_transformer.transform(X_scaled_temp).shape[1]

    performance_metrics.append({
        'Model': name,
        'Train R^2': train_r2,
        'Test R^2': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Complexity': complexity
    })

performance_df = pd.DataFrame(performance_metrics)

st.subheader('모델 성능 평가지표 테이블')
st.dataframe(performance_df.round(3))

# 테스트 R^2 점수 비교 시각화
st.subheader('테스트 R^2 점수 비교 시각화')
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ['skyblue', 'lightcoral', 'lightgreen']
ax.bar(performance_df['Model'], performance_df['Test R^2'], color=bar_colors)
ax.set_xlabel('모델', fontsize=12)
ax.set_ylabel('R^2 점수', fontsize=12)
ax.set_title('각 모델의 테스트 R^2 점수 비교', fontsize=14)
ax.set_ylim(min(0, performance_df['Test R^2'].min() - 0.1), max(1, performance_df['Test R^2'].max() + 0.1))

for i, v in enumerate(performance_df['Test R^2']):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10) # 값 표시

st.pyplot(fig)


# --- 2. 실시간 예측 UI 구성 섹션 ([조건 4]) ---
st.header('실시간 기대 수명 예측')

st.sidebar.header('예측 변수 입력')

# 슬라이더를 위한 입력값
input_adult_mortality = st.sidebar.slider('성인 사망률 (Adult mortality)', 1.0, 750.0, 160.0)
input_bmi = st.sidebar.slider('BMI', 1.0, 80.0, 38.0)
input_gdp = st.sidebar.slider('GDP', 1.0, 120000.0, 7000.0)
input_alcohol = st.sidebar.slider('알코올 소비량 (Alcohol)', 0.0, 18.0, 5.0)
input_polio = st.sidebar.slider('소아마비 예방접종률 (Polio)', 0.0, 100.0, 80.0)

# 입력값을 DataFrame으로 변환
input_data = pd.DataFrame([{
    'Adult mortality': input_adult_mortality,
    'BMI': input_bmi,
    'GDP': input_gdp,
    'Alcohol': input_alcohol,
    'Polio': input_polio
}])

# 모델 선택 인터페이스
selected_model_name = st.selectbox(
    '예측에 사용할 모델을 선택하세요:',
    ('Linear Model', 'Polynomial Model', 'Ridge Model')
)

selected_model = models[selected_model_name]

# 예측 수행
prediction = selected_model.predict(input_data)

st.subheader('예측된 기대 수명')
st.markdown(f"<h1 style='text-align: center; color: #007BFF;'>{prediction[0]:.2f}년</h1>", unsafe_allow_html=True)

st.info("사이드바의 슬라이더를 조절하여 기대 수명을 실시간으로 예측하고, 모델을 변경하여 비교해 보세요.")
