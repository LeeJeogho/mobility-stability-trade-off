import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 텍스트에서 테스트 조건(스트레스) 추출 함수
# ==========================================
def extract_stress_features(row):
    cond = str(row).upper()
    
    s_type = 'Other'
    s_volt = 20.0
    s_temp = 25.0
    s_time = 3600.0
    
    if 'PBTS' in cond: s_type = 'PBTS'
    elif 'NBTS' in cond: s_type = 'NBTS'
    elif 'NBIS' in cond: s_type = 'NBIS'
    elif 'PBS' in cond: s_type = 'PBS'
    elif 'NBS' in cond: s_type = 'NBS'
    
    v_match = re.search(r'([-+]?\d+(\.\d+)?)\s*V', cond)
    if v_match: s_volt = abs(float(v_match.group(1)))
    
    t_match = re.search(r'(\d+(\.\d+)?)\s*(?:°C|℃|C\b)', cond)
    if t_match: 
        s_temp = float(t_match.group(1))
    elif 'RT' in cond or 'ROOM' in cond: 
        s_temp = 25.0
        
    time_s = re.search(r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*S', cond)
    time_h = re.search(r'(\d+(\.\d+)?)\s*HOUR', cond)
    if time_s: s_time = float(time_s.group(1))
    elif time_h: s_time = float(time_h.group(1)) * 3600
        
    return pd.Series([s_type, s_volt, s_temp, s_time])

# ==========================================
# 2. 아웃라이어 제거 함수 (IQR)
# ==========================================
def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

# ==========================================
# 🚀 메인 데이터 전처리 파이프라인
# ==========================================
def load_and_preprocess_data(excel_path):
    print("엑셀 파일에서 데이터를 불러오는 중입니다...")
    
    # 1. 데이터 로드
    ald_df = pd.read_excel(excel_path, sheet_name="ALD data")
    pvd_df = pd.read_excel(excel_path, sheet_name="PVD data")
    ald_df['Deposition_Method'] = 'ALD'
    pvd_df['Deposition_Method'] = 'PVD'
    df = pd.concat([pvd_df, ald_df], ignore_index=True)

    # 2. 타겟 변수 세팅
    df['threshold_voltage_shifts_V'] = pd.to_numeric(df['threshold_voltage_shifts_V'], errors='coerce')
    df['field_effect_mobility_cm²/V⋅s'] = pd.to_numeric(df['field_effect_mobility_cm²/V⋅s'], errors='coerce')

    df['Stability_1_over_dV'] = 1 / (df['threshold_voltage_shifts_V'].abs() + 1e-5)
    df['Stability_log'] = np.log1p(df['Stability_1_over_dV'])

    df = df.dropna(subset=['field_effect_mobility_cm²/V⋅s', 'Stability_1_over_dV'])

    # 3. 기본 텍스트 추출 및 대문자화
    df[['Stress_Type', 'Stress_Voltage_V', 'Stress_Temp_C', 'Stress_Time_s']] = \
        df['stability_measurement_conditions'].apply(extract_stress_features)

    df['annealing_atmosphere'] = df['annealing_atmosphere'].astype(str).str.upper()
    df['channel_material_name'] = df['channel_material_name'].astype(str).str.upper()
    df['gate_insulator_material'] = df['gate_insulator_material'].astype(str).str.upper()

    # ==========================================
    # 🌟 추가: 패시베이션 범주형 결측치 방어 (안 적혀있으면 'NONE'으로 처리)
    # ==========================================
    passi_cat_cols = ['passivation_layer_material', 'passivation_process']
    for col in passi_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().replace('NAN', 'NONE')

    # ==========================================
    # 🌟 추가: 패시베이션 & 반도체 수치형 변수 결측치 방어
    # (안 적혀있으면 두께/시간은 0, 온도는 상온(25)으로 처리하여 데이터 삭제 방지)
    # ==========================================
    if 'passivation_layer_thickness' in df.columns:
        df['passivation_layer_thickness'] = pd.to_numeric(df['passivation_layer_thickness'], errors='coerce').fillna(0)
    
    if 'semiconductor_thickness_nm' in df.columns:
        median_thick = pd.to_numeric(df['semiconductor_thickness_nm'], errors='coerce').median()
        df['semiconductor_thickness_nm'] = pd.to_numeric(df['semiconductor_thickness_nm'], errors='coerce').fillna(median_thick)

    # ==========================================
    # 🌟 4. 도메인 지식 기반 소자 구조 정제 로직 (_ASSUMED 태그 포함)
    # ==========================================
    def standardize_structure(text):
        if pd.isna(text): return 'UNKNOWN'
        
        s = str(text).upper().replace(',', '').replace('-', ' ').strip()
        s = re.sub(r'\s+', ' ', s)

        if 'DOUBLE' in s or 'DUAL' in s: return 'DOUBLE_GATE'

        # 명확한 표준 약어 우선 처리
        if 'BGTC' in s: return 'BGTC'
        if 'BGBC' in s: return 'BGBC'
        if 'TGTC' in s: return 'TGTC'
        if 'TGBC' in s: return 'TGBC'

        # 재료공학적 공정 추론 (_ASSUMED)
        if 'INVERTED STAGGERED' in s: return 'BGTC'
        if 'ETCH STOPPER' in s or 'ESL' in s: return 'BGTC_ESL' # 독립 공정
        if 'BACK CHANNEL ETCH' in s or 'BCE' in s: return 'BGTC_BCE' # 독립 공정
        if 'SELF ALIGNED' in s and 'COPLANAR' in s: return 'TG_SELF_ALIGNED' # 이동도 박살나는 자가정렬 구조 분리!

        if 'COPLANAR' in s:
            if 'TOP GATE' in s: return 'TGTC'
            if 'BOTTOM GATE' in s: return 'BGBC'
            return 'COPLANAR_UNKNOWN'

        if 'BOTTOM GATE' in s and 'TOP CONTACT' in s: return 'BGTC'
        if 'BOTTOM GATE' in s and 'BOTTOM CONTACT' in s: return 'BGBC'
        if 'TOP GATE' in s and 'TOP CONTACT' in s: return 'TGTC'
        if 'TOP GATE' in s and 'BOTTOM CONTACT' in s: return 'TGBC'

        # 불명확한 보류 데이터
        if 'BOTTOM GATE' in s: return 'BG_OTHER'
        if 'TOP GATE' in s: return 'TG_OTHER'

        return 'UNKNOWN'
    
    df['device_structure_type'] = df['device_structure_type'].apply(standardize_structure)

    # 5. 다중 값('10, 20') 정제 후 숫자형 변환
    def clean_multiple_values(x):
        s = str(x)
        if ',' in s: return s.split(',')[0].strip()
        if '/' in s: return s.split('/')[0].strip()
        return x

    num_cols_to_clean = [
        'gate_insulator_thickness_nm', 'Stress_Voltage_V', 'process_temperature_°C', 
        'annealing_temperature_°C', 'semiconductor_thickness_nm', 'Stress_Temp_C', 
        'Stress_Time_s', 'channel_length_nm', 'channel_width_nm'
    ]

    for col in num_cols_to_clean:
        df[col] = df[col].apply(clean_multiple_values)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ==========================================
    # 6. 유전율 매핑 및 물리 지표(Cox, E-field) 계산
    # ==========================================
    dielectric_constants = {
        'SIO2': 3.9, 'AL2O3': 9.0, 'PEALD AL2O3': 9.0,
        'ZRO2': 22.0, 'HFO2': 20.0, 'Y2O3': 15.0, 'SI3N4': 7.5
    }
    
    def get_k_value(mat_name):
        mat = str(mat_name).upper().strip()
        if mat in dielectric_constants: return dielectric_constants[mat]
        # 하이브리드(이중층) 유전율 평균 계산
        if '/' in mat:
            parts = mat.split('/')
            k_sum, count = 0, 0
            for p in parts:
                p_clean = p.strip()
                if p_clean in dielectric_constants:
                    k_sum += dielectric_constants[p_clean]
                    count += 1
            if count > 0: return k_sum / count
            else: return 6.0 
        return 3.9 

    df['k_value'] = df['gate_insulator_material'].apply(get_k_value)
    
    df['EOT_nm'] = df['gate_insulator_thickness_nm'] * (3.9 / df['k_value'])
    df['Stress_E_field_MV_cm'] = (df['Stress_Voltage_V'] / df['gate_insulator_thickness_nm']) * 10
    df['Cox_nF_cm2'] = (885.4 * df['k_value']) / df['gate_insulator_thickness_nm']
    
    df = df.dropna(subset=['EOT_nm', 'Stress_E_field_MV_cm', 'Cox_nF_cm2'])

    # 7. 아웃라이어 제거
    print(f"아웃라이어 제거 전 데이터 개수: {len(df)}")
    df = remove_outliers_iqr(df, 'field_effect_mobility_cm²/V⋅s', multiplier=3.0)
    df = remove_outliers_iqr(df, 'Stability_log', multiplier=3.0)
    print(f"아웃라이어 제거 후 데이터 개수: {len(df)}\n")

    # ==========================================
    # 🔍 8. 데이터 검증 (EDA): 도메인 가설 확인
    # ==========================================
    print("📊 [데이터 검증] 소자 구조별 평균 이동도 & 신뢰성 비교")
    eda_df = df.groupby('device_structure_type')[['field_effect_mobility_cm²/V⋅s', 'Stability_log']].mean().round(2)
    print(eda_df.to_string())
    print("-" * 55)
    print("💡 분석 팁: 본 구조와 '_ASSUMED' 수치가 비슷하다면 추론 로직이 완벽한 것입니다!")
    print("-" * 55 + "\n")

    # ==========================================
    # 🛠️ 9. 데이터 파편화 방지: 학습 직전 꼬리표 병합
    # ==========================================
    # 검증이 끝났으므로 모델이 하나로 인식하도록 꼬리표를 떼어줍니다.
    # df['device_structure_type'] = df['device_structure_type'].str.replace('_ASSUMED', '')

    # 10. 모델 입력 피처 세팅
    y = df[['field_effect_mobility_cm²/V⋅s', 'Stability_log']]
    
    numeric_features = [
        'process_temperature_°C', 'annealing_temperature_°C', 
        'semiconductor_thickness_nm', 
        'Cox_nF_cm2', 'Stress_E_field_MV_cm', 
        'channel_length_nm', 'channel_width_nm',
        'Stress_Temp_C', 'Stress_Time_s',
        'passivation_layer_thickness'
    ]
    
    categorical_features = [
        'channel_material_name', 'gate_insulator_material', 
        'annealing_atmosphere', 'Stress_Type', 'Deposition_Method',
        'device_structure_type',
        'passivation_layer_material', 'passivation_process' 
    ]

    X = df[numeric_features + categorical_features].copy()

    print(f"✅ 데이터 준비 완료! 총 {len(df)}개의 유효한 데이터 포인트를 학습합니다.")
    return X, y, numeric_features, categorical_features