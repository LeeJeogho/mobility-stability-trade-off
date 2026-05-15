import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 우리가 만든 모듈 불러오기
from data_loader import load_and_preprocess_data
from model_builder import get_xgboost_pipeline

warnings.filterwarnings('ignore')

# ==========================================
# 1. 데이터 로드 및 모델 훈련
# ==========================================
excel_file = 'Supplementary material 2(pvd, ald data).xlsx'
X, y, num_feats, cat_feats = load_and_preprocess_data(excel_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = get_xgboost_pipeline(num_feats, cat_feats)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== 로그 변환 적용 후 모델 성능 평가 ===")
print(f"Mobility R2 : {r2_score(y_test.iloc[:, 0], y_pred[:, 0]):.4f}")
print(f"Stability (Log) R2: {r2_score(y_test.iloc[:, 1], y_pred[:, 1]):.4f}")

# ==========================================
# 1.5 변수 중요도 (Feature Importance) 추출
# ==========================================
try:
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    # 원핫 인코딩된 범주형 변수 이름 가져오기
    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
    cat_feat_names = cat_encoder.get_feature_names_out(cat_feats)
    all_feat_names = num_feats + list(cat_feat_names)
    
    # Mobility와 Stability 각각의 중요도 추출
    mob_imp = regressor.estimators_[0].feature_importances_
    stab_imp = regressor.estimators_[1].feature_importances_
    avg_imp = (mob_imp + stab_imp) / 2
    
    imp_df = pd.DataFrame({
        'Feature': all_feat_names,
        'Mobility_Imp': mob_imp,
        'Stability_Imp': stab_imp,
        'Average_Imp': avg_imp
    }).sort_values(by='Average_Imp', ascending=False)
    
    print("\n🔥 [Top 15] 모델이 가장 중요하게 생각한 핵심 변수 🔥")
    print(imp_df.head(15).to_string(index=False))
except Exception as e:
    print("\n[알림] 파이프라인 구조 차이로 변수 중요도를 출력할 수 없습니다.", e)

# ==========================================
# 2. 가상 실험실 (Monte Carlo Simulation) 준비
# ==========================================
print("\n[가상 실험실 가동] 30,000개의 무작위 공정 레시피를 탐색합니다...")
n_samples = 30000
np.random.seed(42)

available_channels = X['channel_material_name'].unique()
available_gi_materials = X['gate_insulator_material'].unique()
available_atmospheres = X['annealing_atmosphere'].unique()
available_passi_mat = [str(m) for m in X['passivation_layer_material'].unique() 
                       if str(m).upper() not in ['NAN', 'NONE', '']]
available_passi_proc = [str(p) for p in X['passivation_process'].unique() 
                        if str(p).upper() not in ['NAN', 'NONE', 'MISSING', '']]
available_structures = [str(s) for s in X['device_structure_type'].unique() 
                        if 'OTHER' not in str(s) and 'UNKNOWN' not in str(s) and str(s).lower() != 'nan']

# 가상 데이터 생성 (Stress_Type 제외하고 생성)
virtual_X = pd.DataFrame({
    'channel_material_name': np.random.choice(available_channels, n_samples),
    'gate_insulator_material': np.random.choice(available_gi_materials, n_samples),
    'Deposition_Method': np.random.choice(['PVD', 'ALD'], n_samples),
    'device_structure_type': np.random.choice(available_structures, n_samples),
    'annealing_atmosphere': np.random.choice(available_atmospheres, n_samples),
    'passivation_layer_material': np.random.choice(available_passi_mat, n_samples),
    'passivation_process': np.random.choice(available_passi_proc, n_samples),
    
    # 수치형 변수 난수 생성
    'process_temperature_°C': np.random.uniform(150, 450, n_samples),
    'annealing_temperature_°C': np.random.uniform(150, 450, n_samples),
    'semiconductor_thickness_nm': np.random.uniform(10, 50, n_samples),
    'gate_insulator_thickness_nm': np.random.uniform(5, 100, n_samples),
    'passivation_layer_thickness': np.random.choice(np.arange(5, 201, 5), n_samples),
    'channel_length_nm': np.random.choice(np.arange(2000, 51000, 1000), n_samples), # 2~50㎛
    'channel_width_nm': np.random.choice(np.arange(2000, 51000, 1000), n_samples),  # 2~50㎛
    
    # 스트레스 기본 조건 고정
    'Stress_Temp_C': 60.0,
    'Stress_Time_s': 3600.0,
    'Stress_Voltage_V': 20.0
})

# 물리 법칙(Cox, E-field) 주입
dielectric_constants = {'SIO2': 3.9, 'AL2O3': 9.0, 'ZRO2': 22.0, 'HFO2': 20.0, 'Y2O3': 15.0, 'SI3N4': 7.5}
def get_k_v(m):
    m = str(m).upper().strip()
    if '/' in m: return 6.5
    return dielectric_constants.get(m, 3.9)

virtual_X['k_value'] = virtual_X['gate_insulator_material'].apply(get_k_v)
virtual_X['Cox_nF_cm2'] = (885.4 * virtual_X['k_value']) / virtual_X['gate_insulator_thickness_nm']
virtual_X['Stress_E_field_MV_cm'] = (virtual_X['Stress_Voltage_V'] / virtual_X['gate_insulator_thickness_nm']) * 10

# ==========================================
# 3. PBTS & NBTS 동시 평가 루프 
# ==========================================
# 동일한 소자에 대해 PBTS와 NBTS를 번갈아 가며 시뮬레이션
virtual_X_pbts = virtual_X.copy()
virtual_X_pbts['Stress_Type'] = 'PBTS'

virtual_X_nbts = virtual_X.copy()
virtual_X_nbts['Stress_Type'] = 'NBTS'

preds_pbts = model.predict(virtual_X_pbts[num_feats + cat_feats])
preds_nbts = model.predict(virtual_X_nbts[num_feats + cat_feats])

# 이동도는 구조 고유의 값이므로 PBTS 런의 결과를 사용
virtual_X['Pred_Mobility'] = preds_pbts[:, 0]
# 로그 역변환으로 델타 Vth 계산
virtual_X['PBTS'] = 1 / np.expm1(preds_pbts[:, 1])
virtual_X['NBTS'] = 1 / np.expm1(preds_nbts[:, 1])

# 통합 안정성 점수 (두 값이 모두 낮아야 유리)
virtual_X['Total_Stability_Score'] = virtual_X['PBTS'] + virtual_X['NBTS']

# ==========================================
# 4. 결과 분석 및 최종 레시피 출력 (Full-Spec)
# ==========================================
mask = virtual_X['Pred_Mobility'] > 40
top_recipes = virtual_X[mask].sort_values(by='Total_Stability_Score', ascending=True).head(5)

display_cols = [
    'channel_material_name', 'device_structure_type', 
    'channel_length_nm', 'channel_width_nm',
    'semiconductor_thickness_nm', 
    'gate_insulator_material', 'gate_insulator_thickness_nm', 'Deposition_Method', 
    'passivation_layer_material', 'passivation_layer_thickness',
    'Pred_Mobility', 'PBTS', 'NBTS'
]

formatted_recipes = top_recipes[display_cols].copy()

int_cols = [
    'channel_length_nm', 'channel_width_nm', 
    'semiconductor_thickness_nm', 'gate_insulator_thickness_nm', 'passivation_layer_thickness'
]
for col in int_cols:
    formatted_recipes[col] = formatted_recipes[col].astype(int)

formatted_recipes['Pred_Mobility'] = formatted_recipes['Pred_Mobility'].round(2)
formatted_recipes['PBTS'] = formatted_recipes['PBTS'].round(4)
formatted_recipes['NBTS'] = formatted_recipes['NBTS'].round(4)

print(f"\n[평가 완료] 30,000개의 가상 공정 중 최적의 풀-스펙 레시피를 도출했습니다.")
print("🏆 [Top 5] 이동도(>40) & 신뢰성(PBTS/NBTS) 극대화 레시피 (Full-Spec) 🏆")
print(formatted_recipes.to_string(index=False))

# ==========================================
# 5. 파레토 프론티어 시각화 (별 모양 마킹)
# ==========================================
def get_pareto_frontier(xs, ys):
    sorted_idx = np.argsort(xs)
    xs, ys = xs[sorted_idx], ys[sorted_idx]
    pareto_x, pareto_y = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        if ys[i] < pareto_y[-1]:
            pareto_x.append(xs[i])
            pareto_y.append(ys[i])
    return np.array(pareto_x), np.array(pareto_y)

plt.figure(figsize=(10, 6))

# 문헌 데이터 (역변환된 문턱전압 변화량 사용)
actual_mob = y.iloc[:, 0].values
actual_dv = 1 / np.expm1(y.iloc[:, 1]).values
plt.scatter(actual_mob, actual_dv, color='lightgray', alpha=0.5, label='Literature Data')

# AI 가상 실험 데이터 (통합 신뢰성 평균 사용)
virtual_avg_dv = (virtual_X['PBTS'] + virtual_X['NBTS']) / 2
plt.scatter(virtual_X['Pred_Mobility'], virtual_avg_dv, color='skyblue', alpha=0.1, s=1, label='AI Simulated')

# 우리가 찾은 Top 5 (Red Star ★)
top_avg_dv = (top_recipes['PBTS'] + top_recipes['NBTS']) / 2
plt.scatter(top_recipes['Pred_Mobility'], top_avg_dv, 
            color='red', marker='*', s=200, edgecolors='black', label='AI Optimized (Top 5)')

# 파레토 한계선
px, py = get_pareto_frontier(actual_mob, actual_dv)
plt.plot(px, py, color='black', linestyle='--', linewidth=2, label='Pareto Limit')

plt.yscale('log')
plt.xlabel('Mobility (cm²/V·s)')
plt.ylabel('Average Threshold Voltage Shift (V)')
plt.title('Breaking the Mobility-Stability Trade-off with Full-Spec Recipe')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()