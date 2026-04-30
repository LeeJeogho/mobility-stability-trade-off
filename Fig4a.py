import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

#ITZO를 강제로 노란색(Zn-free)으로 옮긴 그래프
#x축 25~45구간의 초록색 그래프의 혹이 사라지고 논문의 데이터와 더 유사해 짐을 볼 수 있음
#주석(Sn/Tin)이 들어갔다는 이유만으로 ITZO를 단순 투명전극(ITO) 부류로 착각한 것으로 보임

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

# (2) 채널 물질 그룹화 (저자의 오분류 가설 유지)
def get_channel_groups(mat_name):
    m = str(mat_name).upper().strip()
    m_clean = m.replace('TIN', 'SN').replace('ZINC', 'ZN').replace('INDIUM', 'IN').replace('GALLIUM', 'GA').replace('ALUMINUM', 'AL')
    
    # ITZO, InSnZnO를 강제로 노란색(Zn-free)으로 보냄
    if 'ITZO' in m_clean or 'INSNZNO' in m_clean:
        return ['Zn free']
    
    has_in = 'IN' in m_clean or any(x in m_clean for x in ['ITO', 'IGO', 'IZO', 'IGZO', 'IWO'])
    has_zn = 'ZN' in m_clean or any(x in m_clean for x in ['ZTO', 'GZO', 'IZO', 'IGZO', 'AZO', 'ZNO', 'ZNON'])
    has_ga = 'GA' in m_clean or any(x in m_clean for x in ['GZO', 'IGO', 'IGZO', 'GA2O3', 'INGAO'])
    
    groups = []
    if not (has_in or has_zn or has_ga): return groups
        
    if not has_in: groups.append('In free')
    elif not has_zn: groups.append('Zn free')
    elif not has_ga:
        if 'ITO' not in m_clean and 'IWO' not in m_clean:
            groups.append('Ga free')
                
    return groups

df_clean['Channel_Group_List'] = df_clean['channel_material_name'].apply(get_channel_groups)
df_channel = df_clean.explode('Channel_Group_List').rename(columns={'Channel_Group_List': 'Channel_Group'})
df_channel = df_channel[df_channel['Channel_Group'].notna()]

# (3) 게이트 절연막 그룹화
def categorize_gi(mat):
    gi_str = str(mat).strip()
    gi_lower = gi_str.lower()
    
    if gi_lower in ['nan', '', 'none', 'unknown']: return 'Unknown'
    if '/' in gi_str or '&' in gi_str or 'hybrid' in gi_lower or 'stack' in gi_lower or 'bilayer' in gi_lower: return 'Hybrid'
    if ':' in gi_str:
        base_mat = gi_lower.split(':')[0]
        if 'alo' in base_mat: return 'Al2O3'
        elif 'sio' in base_mat: return 'SiO2'
        elif any(hk in base_mat for hk in ['hf', 'zr', 'ta', 'y', 'ti', 'nb']): return 'High-k'
    if 'sio' in gi_lower: return 'SiO2'
    elif 'al' in gi_lower and 'o' in gi_lower: return 'Al2O3'
    elif any(hk in gi_lower for hk in ['hf', 'zr', 'ta', 'y', 'ti', 'nb', 'high-k']): return 'High-k'
        
    return 'Others'

df_clean['GI_Group'] = df_clean['gate_insulator_material'].apply(categorize_gi)

sns.set_theme(style="whitegrid")
# ==========================================
# [시각화 2] Fig 4a, 4b: Channel Material Analysis
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_colors = {'In free': '#5C6BC0', 'Zn free': '#F5B041', 'Ga free': '#52BE80'}
group_order = ['In free', 'Zn free', 'Ga free']

# 💡 a 그래프 (Mobility): 완벽한 세팅 영구 고정
df_mob = df_channel[df_channel['Mobility'] <= 160]

for group in group_order:
    subset_mob = df_mob[df_mob['Channel_Group'] == group]
    
    if group == 'Ga free': bw_a = 1.4
    elif group == 'In free': bw_a = 0.8
    else: bw_a = 0.85
        
    sns.kdeplot(data=subset_mob, x='Mobility', color=palette_colors[group], fill=False, linewidth=2.5, 
                clip=(0, 160), bw_adjust=bw_a, label=group, ax=axes[0])

# 💡 b 그래프 (Stability): 3단 정밀 타격
df_stab = df_channel[df_channel['Stability'] <= 100]

for group in group_order:
    subset_stab = df_stab[df_stab['Channel_Group'] == group]
    
    # 파란색(0.60) 올리고, 노란색(0.85) 내리고, 초록색(0.70) 올림!
    if group == 'In free': bw_b = 0.60
    elif group == 'Zn free': bw_b = 0.85
    else: bw_b = 0.70 # Ga free
        
    sns.kdeplot(data=subset_stab, x='Stability', color=palette_colors[group], fill=False, linewidth=2.5, 
                clip=(0, 60), bw_adjust=bw_b, label=group, ax=axes[1])

axes[0].set_title('Fig 4a. Mobility by Channel Material', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 160)
axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150])
axes[0].set_ylabel('Probability density', fontsize=12)
axes[0].legend(title='Channel_Group')

axes[1].set_title('Fig 4b. Stability by Channel Material', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 60)
axes[1].set_yticks(np.arange(0, 0.19, 0.02))
axes[1].set_ylim(0, 0.18)
axes[1].set_ylabel('Probability density', fontsize=12)
axes[1].legend(title='Channel_Group')

plt.tight_layout()
plt.show()