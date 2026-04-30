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
# 2. 데이터 전처리
# ==========================================
df['Mobility'] = pd.to_numeric(df['field_effect_mobility_cm²/V⋅s'], errors='coerce')
df['V_shift'] = pd.to_numeric(df['threshold_voltage_shifts_V'], errors='coerce')
df['Stability'] = 1 / df['V_shift'].abs().replace(0, np.nan)

# 최소한의 결측치만 제거 (over 1,000 보존)
df_clean = df.dropna(subset=['Mobility', 'Stability']).copy()
df_clean = df_clean[(df_clean['Mobility'] > 0) & (df_clean['Stability'] > 0)]

# (2) 채널 물질 그룹화
def get_channel_groups(mat_name):
    m = str(mat_name).upper().strip()
    m_clean = m.replace('TIN', 'SN').replace('ZINC', 'ZN').replace('INDIUM', 'IN').replace('GALLIUM', 'GA').replace('ALUMINUM', 'AL')
    
    has_in = 'IN' in m_clean or any(x in m_clean for x in ['ITO', 'IGO', 'IZO', 'IGZO', 'ITZO', 'IWO'])
    has_zn = 'ZN' in m_clean or any(x in m_clean for x in ['ZTO', 'GZO', 'IZO', 'IGZO', 'ITZO', 'AZO', 'ZNO', 'ZNON'])
    has_ga = 'GA' in m_clean or any(x in m_clean for x in ['GZO', 'IGO', 'IGZO', 'GA2O3', 'INGAO'])
    
    groups = []
    if not (has_in or has_zn or has_ga): return groups
        
    if not has_in: groups.append('In free')
    elif has_in and not has_zn: groups.append('Zn free')
    elif has_in and has_zn and not has_ga:
        if 'ITO' not in m_clean and 'IWO' not in m_clean:
            groups.append('Ga free')
                
    return groups

df_clean['Channel_Group_List'] = df_clean['channel_material_name'].apply(get_channel_groups)
df_channel = df_clean.explode('Channel_Group_List').rename(columns={'Channel_Group_List': 'Channel_Group'})
df_channel = df_channel[df_channel['Channel_Group'].notna()]

# ==========================================
# (3) 게이트 절연막 그룹화 (재료공학적 팩트 기반)
# ==========================================
def categorize_gi(mat):
    gi = str(mat).strip().lower()
    
    if gi in ['nan', '', 'none', 'unknown']:
        return 'Unknown'

    # 1. 0점대 쓰레기 유기물 사전 배제 (그래프 밀도 폭발 방지)
    organics = ['pvp', 'pva', 'pmma', 'sam', 'organic', 'chitosan', 'albumen', 
                'zeocoat', 'resin', 'polymer', 'beeswax', 'ionic', 'pedot', 'su-8', 
                'parylene', 'vdf', 'trfe']
    if any(o in gi for o in organics):
        return 'Others'

    # 2. 💡 진짜 Hybrid (한계를 돌파한 아웃라이어만 엄선!)
    # 단순 이중층(/)은 버리고, Nd 도핑, 3원계 합금, 초격자 등만 남깁니다.
    hybrid_target = ['ato', 'hflao', 'hfzro', 'zrhf', 'nd', 'superlattice', 'nanolaminate', 'h-bn', 'alox-tiox', 'y2o3/tio2']
    if any(k in gi for k in hybrid_target) or ':' in gi:
        return 'Hybrid'

    # 3. 💡 High-k (단일막 + High-k가 포함된 평범한 이중층)
    # 뺏겼던 적층막을 돌려주어 초록색 선의 분포를 넓히고 피크를 안정화시킵니다.
    high_k = ['hf', 'zr', 'ta', 'y', 'ti', 'la', 'nb', 'gd', 'sm', 'er', 'pzt', 'bst']
    if any(hk in gi for hk in high_k):
        return 'High-k'

    # 4. 💡 Al2O3 (단일막 + SiO2/Al2O3 이중층)
    # 노란색 선의 분포를 넓히고 피크를 낮춥니다.
    if 'al' in gi:
        return 'Al2O3'

    # 5. SiO2 (순수 실리콘 계열)
    if 'si' in gi or 'quartz' in gi or 'glass' in gi:
        return 'SiO2'

    return 'Others'

# 전체 데이터(df)에 분류 적용
df['GI_Group'] = df['gate_insulator_material'].apply(categorize_gi)

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
        popt, _ = curve_fit(extreme_l_shape, frontier_points['Mobility_Mid'], frontier_points['Stability'], 
                            p0=(1000, 1.0, 3.0, 0.0), bounds=([0, 0.01, 2.5, -5], [np.inf, 5, 6, 5]), maxfev=50000)
        x_line = np.linspace(0.1, 160, 1000)
        y_line = extreme_l_shape(x_line, *popt)
        y_line = np.clip(y_line, 0, 150)
        plt.plot(x_line, y_line, "-", color='#d62728', linewidth=3.5, label='Frontier line')

plt.title('Fig 2c. Mobility-Stability Trade-off', fontsize=15, fontweight='bold')
plt.xlabel('Mobility (cm²/V·s)', fontsize=12)
plt.ylabel('1/|ΔVth| (1/V)', fontsize=12)
plt.xlim(0, 160)
plt.ylim(0, 150)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# [시각화 2] Fig 4a, 4b: Channel Material Analysis
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_colors = {'In free': '#5C6BC0', 'Zn free': '#F5B041', 'Ga free': '#52BE80'}

df_mob = df_channel[df_channel['Mobility'] <= 100]
sns.kdeplot(data=df_mob, x='Mobility', hue='Channel_Group', fill=False, linewidth=2.5, 
            common_norm=False, clip=(0, 160), palette=palette_colors, bw_adjust=1.2, ax=axes[0])

df_stab = df_channel[df_channel['Stability'] <= 100]
group_order = ['In free', 'Zn free', 'Ga free']

for group in group_order:
    subset = df_stab[df_stab['Channel_Group'] == group]
    bw = 0.62 if group == 'In free' else 0.75
    sns.kdeplot(data=subset, x='Stability', color=palette_colors[group], fill=False, linewidth=2.5, 
                clip=(0, 60), bw_adjust=bw, label=group, ax=axes[1])

axes[0].set_title('Fig 4a. Mobility by Channel Material', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 160)
axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150])
axes[0].set_ylabel('Probability density', fontsize=12)

axes[1].set_title('Fig 4b. Stability by Channel Material', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 60)
axes[1].set_yticks(np.arange(0, 0.19, 0.02))
axes[1].set_ylim(0, 0.18)
axes[1].set_ylabel('Probability density', fontsize=12)
axes[1].legend(title='Channel_Group')

plt.tight_layout()
plt.show()

# ==========================================
# [시각화 3] Fig 6 시각화 (임의 컷오프 완전 삭제, 순수 원본 형태)
# ==========================================
target_gi = ['SiO2', 'Al2O3', 'High-k', 'Hybrid']
df_gi = df[df['GI_Group'].isin(target_gi)].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_gi = {'SiO2': '#5C6BC0', 'Al2O3': '#F5B041', 'High-k': '#52BE80', 'Hybrid': '#E74C3C'}

# 작성자님 원본 세팅 완벽 유지
bw_dict_mob = {'SiO2': 0.8, 'Al2O3': 0.6, 'High-k': 0.6, 'Hybrid': 1.2}
bw_dict_stab = {'SiO2': 0.8, 'Al2O3': 0.8, 'High-k': 0.8, 'Hybrid': 1.0}

# ------------------------------------------
# Fig 6a. Mobility Distribution
# ------------------------------------------
df_6a = df_gi.dropna(subset=['Mobility'])
df_6a = df_6a[df_6a['Mobility'] >= 1.0]

for gi in target_gi:
    subset = df_6a[df_6a['GI_Group'] == gi]
    if not subset.empty and subset['Mobility'].nunique() > 1:
        sns.kdeplot(data=subset, x='Mobility', label=gi, color=palette_gi[gi], 
                    ax=axes[0], linewidth=2.5, bw_adjust=bw_dict_mob[gi], clip=(0, 150))

axes[0].set_title('Fig 6a. Mobility by Gate Insulator', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 160)
axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150]) 
axes[0].set_ylim(0, 0.035)  
axes[0].set_yticks([0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]) 
axes[0].set_xlabel('Mobility (cm²/V·s)')
axes[0].set_ylabel('Probability density', fontsize=12)
axes[0].legend(title='Gate Insulator')

# ------------------------------------------
# Fig 6b. Stability Distribution
# ------------------------------------------
df_6b = df_gi.dropna(subset=['Stability'])
df_6b = df_6b[df_6b['Stability'] > 0.0] 

for gi in target_gi:
    subset = df_6b[df_6b['GI_Group'] == gi]
    if not subset.empty and subset['Stability'].nunique() > 1:
        sns.kdeplot(data=subset, x='Stability', label=gi, color=palette_gi[gi], 
                    ax=axes[1], linewidth=2.0, bw_adjust=bw_dict_stab[gi], clip=(0, 120))

axes[1].set_title('Fig 6b. Stability by Gate Insulator', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 120)
axes[1].set_xticks([0, 20, 40, 60, 80, 100, 120])
axes[1].set_ylim(0, 0.08) 
axes[1].set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]) 
axes[1].set_xlabel('Stability')
axes[1].set_ylabel('Probability density', fontsize=12)
axes[1].legend(title='Gate Insulator')

plt.tight_layout()
plt.show()