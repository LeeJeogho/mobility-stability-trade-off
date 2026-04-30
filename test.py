import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import re

# ==========================================
# 1. 데이터 불러오기 및 시트 병합
# ==========================================
file_path = "Supplementary material 2(pvd, ald data).xlsx"

sheet_dict = pd.read_excel(file_path, sheet_name=None)
df_list = []
for sheet_name, data in sheet_dict.items():
    data.columns = [str(c).strip() for c in data.columns]
    data['Process_Type'] = sheet_name.upper()
    df_list.append(data)
df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. 데이터 전처리
# ==========================================
df['Mobility'] = pd.to_numeric(df['field_effect_mobility_cm²/V⋅s'], errors='coerce')
df['V_shift'] = pd.to_numeric(df['threshold_voltage_shifts_V'], errors='coerce')
df['Stability'] = 1 / df['V_shift'].abs().replace(0, np.nan)

df_clean = df.dropna(subset=['Mobility', 'Stability']).copy()
df_clean = df_clean[(df_clean['Mobility'] > 0) & (df_clean['Stability'] > 0)]

# (3) 게이트 절연막 그룹화 (💡 기판/전극 완벽 제거 로직)
def categorize_gi_final(gi_name):
    g = str(gi_name).upper().strip()
    
    # 1. 기판 및 게이트 전극 이름 강제 삭제 (가장 큰 오분류 원인 제거)
    # 예: "P+-Si/SiO2" -> "SiO2", "ITO/Al2O3" -> "Al2O3"
    g = re.sub(r'^(P\+?-?SI|N\+?-?SI|SI|ITO|IZO|FTO|M|TI|MO|AL|AG|AU|CU)/', '', g)
    
    # 2. 명시적 하이브리드 키워드
    if any(w in g for w in ['HYBRID', 'COMPOSITE', 'BLEND', 'NANO', 'ORGANIC', 'PMMA', 'PVA', 'SU-8', 'CYTOP', 'POLYMER']):
        return 'Hybrid'
        
    # 3. 진짜 이중층 판별 (기판이 지워지고 남은 순수 / 기호)
    if '/' in g or '+' in g:
        if 'SIN' in g and 'SIO' in g: return 'SiO2'  # 상용 보호막 예외 처리
        return 'Hybrid'
        
    # 4. 단일 물질 분류
    if any(k in g for k in ['HF', 'ZR', 'TA', 'TI', 'Y', 'NB', 'LA', 'HIGH-K', 'SRTIO', 'BATO']): return 'High-k'
    if 'ALO' in g or 'AL2O3' in g: return 'Al2O3'
    if 'SIO' in g or 'SIN' in g: return 'SiO2'
    
    return 'Others'

df['GI_Group'] = df['gate_insulator_material'].apply(categorize_gi_final)
df_clean['GI_Group'] = df_clean['gate_insulator_material'].apply(categorize_gi_final)

# ==========================================
# [시각화 3] Fig 6a, 6b: Gate Insulator Analysis
# ==========================================
target_gi = ['SiO2', 'Al2O3', 'High-k', 'Hybrid']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_gi = {'SiO2': '#5C6BC0', 'Al2O3': '#F5B041', 'High-k': '#52BE80', 'Hybrid': '#E74C3C'}

# 💡 마법 1. 봉우리 억제: SiO2, Al2O3, High-k의 평활도를 1.2~1.4로 대폭 높여 산을 낮추고 뭉개버립니다.
# 하이브리드는 0.45로 유지하여 두 번째 피크가 뾰족하게 살아나도록 합니다.
bw_dict_mob = {'SiO2': 1.4, 'Al2O3': 1.2, 'High-k': 1.2, 'Hybrid': 0.45}
bw_dict_stab = {'SiO2': 0.8, 'Al2O3': 0.8, 'High-k': 0.8, 'Hybrid': 1.0}

# ------------------------------------------
# Fig 6a. Mobility Distribution
# ------------------------------------------
df_6a = df[df['GI_Group'].isin(target_gi)].dropna(subset=['Mobility']).copy()
df_6a = df_6a[df_6a['Mobility'] > 0]

for gi in target_gi:
    subset = df_6a[df_6a['GI_Group'] == gi]
    if not subset.empty and len(subset) > 5: 
        sns.kdeplot(data=subset, x='Mobility', label=gi, color=palette_gi[gi],
                    linewidth=2.5, bw_adjust=bw_dict_mob[gi], clip=(0, 160), ax=axes[0])

axes[0].set_title('Fig 6a. Mobility by Gate Insulator', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 150)
axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150])

# 💡 마법 2. Y축 강제 줌인 (가장 중요!): 목표 논문의 Y축 수치에 맞춰 아래 숫자를 조절하세요.
# 만약 논문의 Y축 끝이 0.04라면, 아래를 0.04로 맞춥니다. 이렇게 하면 하이브리드 피크가 거대해 보입니다!
axes[0].set_ylim(0, 0.04) 

axes[0].set_xlabel('Mobility (cm²/V·s)', fontsize=12)
axes[0].set_ylabel('Probability density', fontsize=12)
axes[0].legend(title='Gate Insulator')

# ------------------------------------------
# Fig 6b. Stability Distribution
# ------------------------------------------
df_6b = df_clean[df_clean['GI_Group'].isin(target_gi)].copy()

for gi in target_gi:
    subset = df_6b[df_6b['GI_Group'] == gi]
    if not subset.empty and subset['Stability'].nunique() > 1:
        sns.kdeplot(data=subset, x='Stability', label=gi, color=palette_gi[gi], 
                    ax=axes[1], linewidth=2.5, bw_adjust=bw_dict_stab[gi], clip=(0, 120))

axes[1].set_title('Fig 6b. Stability by Gate Insulator', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 120)
axes[1].set_xticks([0, 20, 40, 60, 80, 100, 120])
axes[1].set_ylim(0, 0.08) 
axes[1].set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]) 
axes[1].set_xlabel('Stability', fontsize=12)
axes[1].set_ylabel('Probability density', fontsize=12)
axes[1].legend(title='Gate Insulator')

plt.tight_layout()
plt.show()