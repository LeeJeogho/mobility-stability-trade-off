import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 우리가 만든 모듈 불러오기
from data_loader import load_and_preprocess_data
from model_builder import get_rf_pipeline, get_xgboost_pipeline

# ==========================================
# 1. 데이터 로드 및 모델 훈련
# ==========================================
# 단일 엑셀 파일 경로 지정
excel_file = 'Supplementary material 2(pvd, ald data).xlsx'

# 데이터 로드 및 전처리
X, y, num_feats, cat_feats = load_and_preprocess_data(excel_file)

# 훈련 / 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 파이프라인 가져오기 및 훈련 (XGBoost 사용)
model = get_xgboost_pipeline(num_feats, cat_feats)
model.fit(X_train, y_train)

# 예측 및 최종 평가
y_pred = model.predict(X_test)

print("\n=== 로그 변환 적용 후 모델 성능 평가 ===")
print(f"Mobility R2 : {r2_score(y_test.iloc[:, 0], y_pred[:, 0]):.4f}")
print(f"Stability (Log) R2: {r2_score(y_test.iloc[:, 1], y_pred[:, 1]):.4f}")

# ==========================================
# 2. 피처 중요도 (Feature Importance) 확인
# ==========================================
# 파이프라인에서 훈련된 XGBoost 모델과 전처리기를 꺼냅니다.
rf_model = model.named_steps['regressor']
preprocessor = model.named_steps['preprocessor']

# 범주형 변수(One-Hot Encoding 적용됨)의 변환된 이름들을 가져옵니다.
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
encoded_cat_features = cat_encoder.get_feature_names_out(cat_feats)

# 수치형 변수 이름과 합쳐서 전체 피처 이름을 완성합니다.
all_feature_names = np.concatenate([num_feats, encoded_cat_features])

# 다중 출력 모델이므로 각각 중요도를 가져옵니다.
importance_mobility = rf_model.estimators_[0].feature_importances_
importance_stability = rf_model.estimators_[1].feature_importances_

# 두 중요도의 평균을 구해서 종합 랭킹을 매깁니다.
importance_avg = (importance_mobility + importance_stability) / 2

# 보기 좋게 정렬하여 데이터프레임으로 만듭니다.
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Mobility_Imp': importance_mobility,
    'Stability_Imp': importance_stability,
    'Average_Imp': importance_avg
}).sort_values(by='Average_Imp', ascending=False)

print("\n🔥 [Top 15] 모델이 가장 중요하게 생각한 핵심 변수 🔥")
print(feature_importance_df.head(15).round(4).to_string(index=False))

# ==========================================
# 3. 대망의 역설계: 몬테카를로 (Monte Carlo) 시뮬레이션
# ==========================================
print("\n[가상 실험실 가동] 몬테카를로 시뮬레이션으로 30,000개의 무작위 공정을 테스트합니다...")

n_samples = 30000
np.random.seed(42) # 결과 재현성을 위한 시드 고정

# 연속적인 난수(Uniform)와 무작위 추출(Choice)을 통해 촘촘한 가상 공간 생성
virtual_X = pd.DataFrame({
    'channel_material_name': np.random.choice(['IGZO', 'IGZTO', 'ITZO', 'GD-AZO', 'IZO', 'ZNON'], n_samples),
    'gate_insulator_material': np.random.choice(['SIO2', 'AL2O3', 'ZRO2', 'PEALD AL2O3'], n_samples),
    'Deposition_Method': np.random.choice(['PVD', 'ALD'], n_samples),
    'process_temperature_°C': np.random.uniform(100, 400, n_samples), # 100~400도 사이의 소수점 포함 랜덤 온도
    'annealing_temperature_°C': np.random.uniform(150, 500, n_samples), 
    'annealing_atmosphere': np.random.choice(['AIR', 'O2', 'DRY AIR (O2 21%/N2 79%)', 'VACUUM', 'N2'], n_samples),
    'semiconductor_thickness_nm': np.random.uniform(5, 50, n_samples),
    'gate_insulator_thickness_nm': np.random.uniform(5, 200, n_samples),
    
    # 평가를 위한 가혹 스트레스 조건 고정 (PBTS, 20V, 60도, 1시간)
    'Stress_Type': ['PBTS'] * n_samples,
    'Stress_Voltage_V': [20.0] * n_samples,
    'Stress_Temp_C': [60.0] * n_samples,
    'Stress_Time_s': [3600.0] * n_samples
})

# 훈련 데이터와 완벽하게 구조 맞추기
virtual_X = virtual_X[X.columns] 

for col in cat_feats:
    virtual_X.loc[:, col] = virtual_X[col].astype(str)
for col in num_feats:
    virtual_X.loc[:, col] = virtual_X[col].astype(float)

# 모델 예측
virtual_predictions = model.predict(virtual_X)

virtual_X['Pred_Mobility'] = virtual_predictions[:, 0]
virtual_X['Pred_Stability_Score'] = np.expm1(virtual_predictions[:, 1])

# ==========================================
# 🌟 도메인 지식(Domain Knowledge) 주입: 절연 파괴(Breakdown) 필터링
# ==========================================
virtual_X['Electric_Field_MV_cm'] = (virtual_X['Stress_Voltage_V'] / virtual_X['gate_insulator_thickness_nm']) * 10

# 물리적으로 버틸 수 있는 한계치(8 MV/cm)를 넘는 레시피는 폐기
physically_valid_recipes = virtual_X[virtual_X['Electric_Field_MV_cm'] <= 8.0]

# 🚨 수정된 부분: 물질 이름이 아니라 '예측된 점수'가 완전히 똑같은(같은 트리 방에 빠진) 중복만 제거합니다.
# 이렇게 하면 물질이 같아도 두께나 온도가 최적화되어 다른 성능이 나온 레시피들은 모두 살아남습니다.
unique_recipes = physically_valid_recipes.drop_duplicates(
    subset=['Pred_Mobility', 'Pred_Stability_Score']
)

# 안정성 점수 기준으로 내림차순 정렬
sorted_recipes = unique_recipes.sort_values(by='Pred_Stability_Score', ascending=False)

# 이동도가 40 이상인 것들 중에서 Top 5를 추출합니다.
# (만약 여기서도 5개가 다 안 채워지면, > 40을 > 35나 > 30으로 살짝 낮춰보세요!)
top_recipes = sorted_recipes[sorted_recipes['Pred_Mobility'] > 40].head(5)

print(f"\n[물리법칙 검증] 생존한 소자 중 성능이 중복되지 않은 {len(unique_recipes)}개의 고유한 레시피를 확보했습니다.")
print("🏆 [Top 5] 물리적 한계 돌파 & 이동도(>40) 조건을 만족하는 최적 레시피 🏆")
print(top_recipes[['channel_material_name', 'gate_insulator_material', 'gate_insulator_thickness_nm',
                   'Deposition_Method', 'process_temperature_°C',  
                   'Pred_Mobility', 'Pred_Stability_Score']].to_string(index=False))
# Pred_Stability_Score = 1/(|ΔVth| + 0.00001)

# ==========================================
# 4. 시각화: 논문과 동일한 Y축(전압 변화량)으로 변환 및 하단 파레토 프론티어(L자) 추출
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

# 💡 목표: 이동도(X)는 '최대화'하면서, 전압 변화량(Y)은 '최소화'하는 하단 한계선을 찾습니다.
def get_lower_pareto_frontier(xs, ys):
    # 이동도(x)를 내림차순(가장 높은 것부터)으로 정렬
    sorted_indices = np.argsort(xs)[::-1] 
    pareto_x = []
    pareto_y = []
    
    # Y축(전압 변화량)은 낮을수록 좋으므로, 최소값을 추적합니다.
    min_y = np.inf 
    for idx in sorted_indices:
        if ys[idx] < min_y: # 기존에 발견된 가장 낮은 V-shift보다 더 낮다면 한계선으로 인정!
            pareto_x.append(xs[idx])
            pareto_y.append(ys[idx])
            min_y = ys[idx]
    return pareto_x, pareto_y

plt.figure(figsize=(10, 6))

# 1. 실제 원본 데이터(y) 변환: 안정성 점수 -> 원본 문턱전압 변화량(dV)으로 복구
actual_mobility = y.iloc[:, 0].values
actual_stability_score = np.expm1(y.iloc[:, 1]).values
actual_v_shift = 1 / actual_stability_score # 역수를 취해 V-shift로 되돌림!

# 회색 배경 점으로 실제 논문 데이터 깔기
plt.scatter(actual_mobility, actual_v_shift, 
            color='lightgray', alpha=0.5, s=20, label='Actual Literature Data')

# 실제 데이터의 L자 한계선(하단) 그리기
real_px, real_py = get_lower_pareto_frontier(actual_mobility, actual_v_shift)
plt.plot(real_px, real_py, color='gray', linestyle='-', linewidth=2, label='Actual Pareto Frontier')

# 2. AI 가상 데이터 변환 및 시각화
unique_recipes['Pred_V_shift'] = 1 / unique_recipes['Pred_Stability_Score'] # AI 예측값도 V-shift로 변환

for method, color in zip(['ALD', 'PVD'], ['#1f77b4', '#ff7f0e']):
    subset = unique_recipes[unique_recipes['Deposition_Method'] == method]
    plt.scatter(subset['Pred_Mobility'], subset['Pred_V_shift'], 
                alpha=0.3, color=color, edgecolors='none', s=20)

# 3. AI 가상 데이터의 하단 한계선 (새롭게 뚫은 한계 돌파선)
ai_px, ai_py = get_lower_pareto_frontier(unique_recipes['Pred_Mobility'].values, unique_recipes['Pred_V_shift'].values)
plt.plot(ai_px, ai_py, color='red', linestyle='-', linewidth=3, label='AI Virtual Pareto Frontier')

plt.title('Mobility vs. Threshold Voltage Shift (Actual vs AI Virtual)', fontsize=15, fontweight='bold')
plt.xlabel('Mobility (cm²/V⋅s)', fontsize=12)
# 💡 Y축 라벨을 논문과 완벽하게 동일하게 변경!
plt.ylabel('Threshold voltage shifts (V)', fontsize=12) 
plt.yscale('log') # 논문처럼 Y축을 로그 스케일로 적용

# 타겟 성능 십자선 (이동도 40 이상, V-shift 0.1 이하 타겟)
plt.axvline(x=40, color='black', linestyle=':', alpha=0.5)
plt.axhline(y=0.1, color='black', linestyle=':', alpha=0.5)

# 축 범위 세팅 (극단적 아웃라이어로 인해 그래프가 찌그러지는 것 방지)
plt.xlim(0, 140)
plt.ylim(0.01, 100) # 논문과 유사하게 10^-2 ~ 10^2 범위로 세팅

plt.legend(loc='upper left') # 범례 위치 조정
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()