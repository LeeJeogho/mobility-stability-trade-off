import pandas as pd
import numpy as np
import re

# 텍스트에서 테스트 조건(종류, 전압, 온도, 시간)을 추출하는 함수
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
# 🌟 추가된 함수: IQR 기반 아웃라이어 제거
# ==========================================
def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    # 정상 범위에 있는 데이터만 필터링하여 반환
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]


def load_and_preprocess_data(excel_path):
    print("엑셀 파일에서 데이터를 불러오는 중입니다...")
    
    # 1. 데이터 로드 및 병합
    ald_df = pd.read_excel(excel_path, sheet_name="ALD data")
    pvd_df = pd.read_excel(excel_path, sheet_name="PVD data")
    ald_df['Deposition_Method'] = 'ALD'
    pvd_df['Deposition_Method'] = 'PVD'
    df = pd.concat([pvd_df, ald_df], ignore_index=True)

    # 2. 타겟 변수 숫자형 변환 및 생성
    df['threshold_voltage_shifts_V'] = pd.to_numeric(df['threshold_voltage_shifts_V'], errors='coerce')
    df['field_effect_mobility_cm²/V⋅s'] = pd.to_numeric(df['field_effect_mobility_cm²/V⋅s'], errors='coerce')

    df['Stability_1_over_dV'] = 1 / (df['threshold_voltage_shifts_V'].abs() + 1e-5)
    df['Stability_log'] = np.log1p(df['Stability_1_over_dV'])

    df = df.dropna(subset=['field_effect_mobility_cm²/V⋅s', 'Stability_1_over_dV'])

    # 3. 피처 엔지니어링 (텍스트 추출 및 대소문자 통일)
    df[['Stress_Type', 'Stress_Voltage_V', 'Stress_Temp_C', 'Stress_Time_s']] = \
        df['stability_measurement_conditions'].apply(extract_stress_features)

    df['annealing_atmosphere'] = df['annealing_atmosphere'].astype(str).str.upper()
    df['channel_material_name'] = df['channel_material_name'].astype(str).str.upper()

    numeric_features = ['process_temperature_°C', 'annealing_temperature_°C', 
                        'semiconductor_thickness_nm', 'gate_insulator_thickness_nm',
                        'Stress_Voltage_V', 'Stress_Temp_C', 'Stress_Time_s']
    
    categorical_features = ['channel_material_name', 'gate_insulator_material', 
                            'annealing_atmosphere', 'Stress_Type', 'Deposition_Method']
    
    # 4. 수치형 피처의 숫자형 강제 변환
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ==========================================
    # 🌟 아웃라이어 제거 실행 (X, y 분리 직전)
    # ==========================================
    print(f"아웃라이어 제거 전 데이터 개수: {len(df)}")
    
    # 이동도와 안정성 각각에 대해 아웃라이어 제거 (multiplier=1.5 기준)
    df = remove_outliers_iqr(df, 'field_effect_mobility_cm²/V⋅s', multiplier=3.0)
    df = remove_outliers_iqr(df, 'Stability_log', multiplier=3.0)
    
    print(f"아웃라이어 제거 후 데이터 개수: {len(df)}")
    # ==========================================

    # 5. 최종 X, y 분리
    y = df[['field_effect_mobility_cm²/V⋅s', 'Stability_log']]
    X = df[numeric_features + categorical_features].copy()

    print(f"데이터 준비 완료! 총 {len(df)}개의 유효한 데이터 포인트를 학습합니다.")
    return X, y, numeric_features, categorical_features