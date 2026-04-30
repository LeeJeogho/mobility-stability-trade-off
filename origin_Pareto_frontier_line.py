import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# ==========================================
# 1. 데이터 불러오기 및 시트 병합
# ==========================================
file_path = "Supplementary material 2(pvd, ald data).xlsx"

sheet_dict = pd.read_excel(file_path, sheet_name=None)
df_list = []
for sheet_name, data in sheet_dict.items():
    data['Process_Type'] = sheet_name.upper()
    df_list.append(data)
df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. 데이터 전처리 (💡 필터 없는 순수 데이터)
# ==========================================
df['Mobility'] = pd.to_numeric(df['field_effect_mobility_cm²/V⋅s'], errors='coerce')
df['V_shift'] = pd.to_numeric(df['threshold_voltage_shifts_V'], errors='coerce')
df['Stability'] = 1 / df['V_shift'].abs().replace(0, np.nan)

# 상한선(Outlier 커트라인) 없이 필수 결측치 및 음수만 제거
df_clean = df.dropna(subset=['Mobility', 'Stability']).copy()
df_clean = df_clean[(df_clean['Mobility'] > 0) & (df_clean['Stability'] > 0)]

print(f"추출된 순수 로우 데이터 개수: {len(df_clean)}")

# (2) 채널 물질 그룹화 (💡 ITO 등 예외 처리 없는 원본 로직)
def get_channel_groups_raw(mat_name):
    m = str(mat_name).upper().strip()
    m_clean = m.replace('TIN', 'SN').replace('ZINC', 'ZN').replace('INDIUM', 'IN').replace('GALLIUM', 'GA').replace('ALUMINUM', 'AL')
    
    has_in = 'IN' in m_clean or any(x in m_clean for x in ['ITO', 'IGO', 'IZO', 'IGZO', 'ITZO', 'IWO'])
    has_zn = 'ZN' in m_clean or any(x in m_clean for x in ['ZTO', 'GZO', 'IZO', 'IGZO', 'ITZO', 'AZO', 'ZNO', 'ZNON'])
    has_ga = 'GA' in m_clean or any(x in m_clean for x in ['GZO', 'IGO', 'IGZO', 'GA2O3', 'INGAO'])
    
    groups = []
    if not (has_in or has_zn or has_ga): return groups
        
    if not has_in: 
        groups.append('In free')
    elif has_in and not has_zn: 
        groups.append('Zn free')
    elif has_in and has_zn and not has_ga:
        # 혹(Bump) 생성 방지를 위한 투명전극 예외 처리 없이 무조건 포함
        groups.append('Ga free') 
                
    return groups

df_clean['Channel_Group_List'] = df_clean['channel_material_name'].apply(get_channel_groups_raw)
df_channel = df_clean.explode('Channel_Group_List').rename(columns={'Channel_Group_List': 'Channel_Group'})
df_channel = df_channel[df_channel['Channel_Group'].notna()]

# (3) 게이트 절연막 그룹화
def categorize_gi(mat):
    mat_str = str(mat).upper()
    if '/' in mat_str or '+' in mat_str or 'STACK' in mat_str: return 'Hybrid'
    elif 'SIO' in mat_str: return 'SiO2'
    elif 'ALO' in mat_str: return 'Al2O3'
    elif any(k in mat_str for k in ['HF', 'ZR', 'TA', 'Y2O3', 'TIO']): return 'High-k'
    else: return 'Others'

df_clean['GI_Group'] = df_clean['gate_insulator_material'].apply(categorize_gi)

sns.set_theme(style="whitegrid")

# ==========================================
# [시각화 1] Fig 2c: Mobility-Stability Trade-off
# ==========================================
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='Mobility', y='Stability', alpha=0.6, color='royalblue', edgecolor='k', s=70)

df_fit = df_clean[~((df_clean['Mobility'] > 25) & (df_clean['Stability'] > 25))].copy()
if not df_fit.empty and df_fit['Mobility'].nunique() > 1:
    bins = np.linspace(df_fit['Mobility'].min(), df_fit['Mobility'].max(), 20)
    df_fit['Mobility_Bin'] = pd.cut(df_fit['Mobility'], bins)
    frontier_points = df_fit.groupby('Mobility_Bin', observed=False)['Stability'].max().dropna().reset_index()
    frontier_points['Mobility_Mid'] = frontier_points['Mobility_Bin'].apply(lambda x: x.mid).astype(float)

    def extreme_l_shape(x, a, b, c, d): return a / (x + b)**c + d

    if len(frontier_points) > 3:
        try:
            popt, _ = curve_fit(extreme_l_shape, frontier_points['Mobility_Mid'], frontier_points['Stability'], 
                                p0=(1000, 1.0, 3.0, 0.0), bounds=([0, 0.01, 2.5, -5], [np.inf, 5, 6, 5]), maxfev=50000)
            x_line = np.linspace(0.1, 160, 1000)
            y_line = extreme_l_shape(x_line, *popt)
            y_line = np.clip(y_line, 0, 150)
            plt.plot(x_line, y_line, "-", color='#d62728', linewidth=3.5, label='Frontier line')
        except RuntimeError:
            print("Frontier line fitting failed.")

plt.title('Fig 2c. Original Data (No Filters)', fontsize=15, fontweight='bold')
plt.xlabel('Mobility (cm²/V·s)', fontsize=12); plt.ylabel('1/|ΔVth| (1/V)', fontsize=12)
plt.xlim(0, 160); plt.ylim(0, 150)
plt.legend(); plt.tight_layout(); plt.show()

# ==========================================
# [시각화 2] Fig 4a, 4b: Channel Material Analysis (Pure KDE)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_colors = {'In free': '#5C6BC0', 'Zn free': '#F5B041', 'Ga free': '#52BE80'}

# 💡 상한선 절단(clip, limit) 및 bw_adjust 조작이 전혀 없는 순수 KDE
sns.kdeplot(data=df_channel, x='Mobility', hue='Channel_Group', fill=False, linewidth=2.5, 
            common_norm=False, palette=palette_colors, ax=axes[0])

sns.kdeplot(data=df_channel, x='Stability', hue='Channel_Group', fill=False, linewidth=2.5, 
            common_norm=False, palette=palette_colors, ax=axes[1])

# 시각적 비교를 위해 화면 표시 영역(축)만 논문과 동일하게 세팅
axes[0].set_title('Fig 4a. Pure Mobility Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 160); axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150])
axes[0].set_ylabel('Probability density', fontsize=12)

axes[1].set_title('Fig 4b. Pure Stability Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 60); axes[1].set_yticks(np.arange(0, 0.19, 0.02)); axes[1].set_ylim(0, 0.18)
axes[1].set_ylabel('Probability density', fontsize=12)

plt.tight_layout(); plt.show()

# ==========================================
# [시각화 3] Fig 6a, 6b: Gate Insulator Analysis (Pure KDE)
# ==========================================
target_gi = ['SiO2', 'Al2O3', 'High-k', 'Hybrid']
df_gi = df_clean[df_clean['GI_Group'].isin(target_gi)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_gi = {'SiO2': '#5C6BC0', 'Al2O3': '#F5B041', 'High-k': '#52BE80', 'Hybrid': '#E74C3C'}

sns.kdeplot(data=df_gi, x='Mobility', hue='GI_Group', fill=False, linewidth=2.5, common_norm=False, 
            palette=palette_gi, ax=axes[0])
sns.kdeplot(data=df_gi, x='Stability', hue='GI_Group', fill=False, linewidth=2.5, common_norm=False, 
            palette=palette_gi, ax=axes[1])

axes[0].set_title('Fig 6a. Pure GI Mobility', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 160)
axes[1].set_title('Fig 6b. Pure GI Stability', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 120)
plt.tight_layout(); plt.show()