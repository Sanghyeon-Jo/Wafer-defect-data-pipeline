%%capture
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
import seaborn as sns
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import mannwhitneyu, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer


# 한글 폰트 설정
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'NanumGothic'

# 데이터 로드
df = pd.read_csv('/content/반도체 데이터셋.csv')

print(f"총 데이터 포인트: {len(df)}")
print(f"실제 결함: {len(df[df['IS_DEFECT'] == 'REAL'])}")
print(f"거짓 결함: {len(df[df['IS_DEFECT'] == 'FALSE'])}")
print(f"고유 클래스: {sorted(df['Class'].unique())}")
print(f"반지름 범위: {df['RADIUS'].min():.2f} - {df['RADIUS'].max():.2f}")
print(f"각도 범위: {df['ANGLE'].min():.2f} - {df['ANGLE'].max():.2f}")

df.groupby(['Step_desc', 'Lot Name', 'Slot No'])['SIZE_X'].sum()

import pandas as pd
import numpy as np

# (df는 전체 데이터 프레임이라고 가정합니다)

# 1. 정제할 크기 관련 컬럼들
size_columns = ['SIZE_X', 'SIZE_Y', 'DEFECT_AREA']

# 2. 정제된 데이터프레임을 저장할 빈 리스트 생성
cleaned_dfs_list = []

# 3. 모든 'Class' ID를 하나씩 순회
print("--- 이상치 정제 시작 ---")
for class_id in df['Class'].unique():

    # 3-1. 해당 Class의 데이터만 분리
    df_class = df[df['Class'] == class_id].copy()
    original_count = len(df_class)
    print(f"\nProcessing [Class {class_id}] (원본 {original_count}개)")

    # 3-2. 해당 Class 데이터의 '정상 범위'를 IQR로 계산
    for col in size_columns:
        # 데이터가 너무 적으면(e.g., 1개) quantile 계산이 안되므로 스킵
        if len(df_class) < 2:
            continue

        Q1 = df_class[col].quantile(0.25)
        Q3 = df_class[col].quantile(0.75)
        IQR = Q3 - Q1

        # IQR이 0이면 (모든 값이 같으면) 이상치 계산이 무의미하므로 스킵
        if IQR == 0:
            continue

        # '정상 범위'의 상한선 정의 (Q3 + 1.5 * IQR)
        upper_bound = Q3 + 1.5 * IQR

        # 3-3. 상한선을 벗어나는 데이터(이상치)를 제거
        initial_rows = len(df_class)
        df_class = df_class[df_class[col] <= upper_bound]
        removed_count = initial_rows - len(df_class)

        if removed_count > 0:
            print(f" -> [{col}] 상한선({upper_bound:.2f}) 초과: {removed_count}개 이상치 행 제거.")

    # 3-4. 정제가 완료된 df_class를 리스트에 추가
    cleaned_dfs_list.append(df_class)

# 4. 리스트에 저장된 모든 정제된 데이터프레임을 하나로 다시 합치기
df_cleaned = pd.concat(cleaned_dfs_list)

print(f"\n--- 최종 정제 결과 ---")
print(f"원본 데이터 행 수: {len(df)}")
print(f"정제 후 데이터 행 수: {len(df_cleaned)}")
print(f"제거된 이상치 행 수: {len(df) - len(df_cleaned)}")

import pandas as pd
import numpy as np

print("--- 파생 변수 생성 시작 ---")

# 1. 신호 대 잡음비 (SNR) 관련
# SNR_OFFSET_GL: MDAT_OFFSET / MDAT_GL
# Add a small value (1e-6) to the denominator to avoid division by zero
df_cleaned['SNR_OFFSET_GL'] = df_cleaned['MDAT_OFFSET'] / (df_cleaned['MDAT_GL'] + 1e-6)
print("파생 변수 'SNR_OFFSET_GL' 생성 완료.")

# SNR_INTENSITY_NOISE: INTENSITY / PATCHNOISE (또는 MDAT_NOISE)
# Using PATCHNOISE as the denominator as it's a more direct noise measurement for the patch
# Add a small value (1e-6) to the denominator to avoid division by zero
df_cleaned['SNR_INTENSITY_NOISE'] = df_cleaned['INTENSITY'] / (df_cleaned['PATCHNOISE'] + 1e-6)
print("파생 변수 'SNR_INTENSITY_NOISE' 생성 완료.")


# 2. 형태 및 밀도 관련
# ASPECT_RATIO (종횡비): SIZE_X / SIZE_Y
# Add a small value (1e-6) to the denominator to avoid division by zero
df_cleaned['ASPECT_RATIO'] = df_cleaned['SIZE_X'] / (df_cleaned['SIZE_Y'] + 1e-6)
# Handle potential infinite values that might result from division by near zero
df_cleaned['ASPECT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
print("파생 변수 'ASPECT_RATIO' 생성 완료.")

# DENSITY_SIGNAL (신호 밀도): INTENSITY / DEFECT_AREA
# Add a small value (1e-6) to the denominator to avoid division by zero
df_cleaned['DENSITY_SIGNAL'] = df_cleaned['INTENSITY'] / (df_cleaned['DEFECT_AREA'] + 1e-6)
# Handle potential infinite values
df_cleaned['DENSITY_SIGNAL'].replace([np.inf, -np.inf], np.nan, inplace=True)
print("파생 변수 'DENSITY_SIGNAL' 생성 완료.")

print("\n--- 파생 변수 생성 완료. 데이터 미리보기: ---")
# Display the new columns and a few relevant original columns
display(df_cleaned[['MDAT_OFFSET', 'MDAT_GL', 'SNR_OFFSET_GL',
            'INTENSITY', 'PATCHNOISE', 'SNR_INTENSITY_NOISE',
            'SIZE_X', 'SIZE_Y', 'ASPECT_RATIO',
            'INTENSITY', 'DEFECT_AREA', 'DENSITY_SIGNAL']].head())

print(f"\n업데이트된 데이터 형태: {df_cleaned.shape}")

df_real_defect = df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL']
df_false_defect = df_cleaned[df_cleaned['IS_DEFECT'] == 'FALSE']

print(f"실제 결함 데이터 포인트: {len(df_real_defect)}")
print(f"거짓 결함 데이터 포인트: {len(df_false_defect)}")

print(df_cleaned.columns)

print(f"반지름 범위: {df_cleaned['RADIUS'].min():.2f} - {df_cleaned['RADIUS'].max():.2f}")
print(f"각도 범위: {df_cleaned['ANGLE'].min():.2f} - {df_cleaned['ANGLE'].max():.2f}")

# Task
Generate separate wafer maps for 'Real' and 'False' defects from the `df_cleaned` DataFrame, using a wafer radius of 150000. Represent each defect as an ellipse with dimensions controlled by the 'SIZE_X' and 'SIZE_Y' columns, and adjust the transparency of the ellipses based on the 'SNR_OFFSET_GL' column.

## Update the plotting function

### Subtask:
Modify the `plot_wafer_map` function to accept parameters for controlling ellipse dimensions (e.g., `size_x_col`, `size_y_col`).


**Reasoning**:
The subtask requires modifying the existing `plot_wafer_map` function to accept parameters for controlling ellipse dimensions and returning the separated dataframes. This involves updating the function definition, calculating X and Y coordinates, splitting the data, handling alpha values, and returning the dataframes.



## Create ellipse patches

### Subtask:
Inside the updated function, iterate through each defect in the filtered DataFrame. For each defect, create a `matplotlib.patches.Ellipse` object using its calculated X and Y coordinates, and the values from `size_x_col` and `size_y_col` for width and height. Consider adding a rotation angle if available or necessary (though typically not available in this dataset).


**Reasoning**:
Iterate through the real and false defect dataframes to create ellipse patches for plotting.



def plot_wafer_map(df, title="Wafer Map", defect_color='red', false_color='blue', alpha_col=None, size_x_col=None, size_y_col=None, real_alpha=0.8, false_alpha=0.3):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 웨이퍼 배경 (원)
    wafer_radius = 150000
    wafer_circle = Circle((0, 0), wafer_radius, edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(wafer_circle)

    # Calculate X and Y coordinates from RADIUS and ANGLE
    df['X'] = df['RADIUS'] * np.cos(np.deg2rad(df['ANGLE']))
    df['Y'] = df['RADIUS'] * np.sin(np.deg2rad(df['ANGLE']))

    # 실제 결함
    df_real = df[df['IS_DEFECT'] == 'REAL'].copy()
    # 거짓 결함
    df_false = df[df['IS_DEFECT'] == 'FALSE'].copy()

    # Determine alpha values based on alpha_col
    if alpha_col and alpha_col in df.columns:
        # Normalize the column values to a 0-1 range for alpha
        # Add a small value to avoid division by zero if all values are the same
        min_val = df[alpha_col].min()
        max_val = df[alpha_col].max()
        alpha_values_real = (df_real[alpha_col] - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) != 0 else real_alpha
        alpha_values_false = (df_false[alpha_col] - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) != 0 else false_alpha
    else:
        alpha_values_real = real_alpha
        alpha_values_false = false_alpha

    # Determine size values based on size_col
    # Scale the column values for ellipse dimensions. May require experimentation for good visual scale.
    # Add a small value to avoid division by zero if all values are the same
    size_range = 10000 # Adjust this range as needed for visual size
    min_size = 10 # Minimum size to avoid invisible ellipses

    size_x_real = min_size
    size_y_real = min_size
    size_x_false = min_size
    size_y_false = min_size

    if size_x_col and size_x_col in df.columns and size_y_col and size_y_col in df.columns:
        min_val_x = df[size_x_col].min()
        max_val_x = df[size_x_col].max()
        min_val_y = df[size_y_col].min()
        max_val_y = df[size_y_col].max()

        if (max_val_x - min_val_x) != 0:
             size_x_real = min_size + size_range * (df_real[size_x_col] - min_val_x) / (max_val_x - min_val_x)
             size_x_false = min_size + size_range * (df_false[size_x_col] - min_val_x) / (max_val_x - min_val_x)
        else:
            size_x_real = min_size
            size_x_false = min_size

        if (max_val_y - min_val_y) != 0:
            size_y_real = min_size + size_range * (df_real[size_y_col] - min_val_y) / (max_val_y - min_val_y)
            size_y_false = min_size + size_range * (df_false[size_y_col] - min_val_y) / (max_val_y - min_val_y)
        else:
            size_y_real = min_size
            size_y_false = min_size


    # Create ellipse patches for real defects
    patches_real = []
    for i, row in df_real.iterrows():
        ellipse = Ellipse((row['X'], row['Y']), width=size_x_real.loc[i] if isinstance(size_x_real, pd.Series) else size_x_real,
                          height=size_y_real.loc[i] if isinstance(size_y_real, pd.Series) else size_y_real,
                          angle=0, alpha=alpha_values_real.loc[i] if isinstance(alpha_values_real, pd.Series) else alpha_values_real,
                          color=defect_color)
        patches_real.append(ellipse)

    # Create ellipse patches for false defects
    patches_false = []
    for i, row in df_false.iterrows():
         ellipse = Ellipse((row['X'], row['Y']), width=size_x_false.loc[i] if isinstance(size_x_false, pd.Series) else size_x_false,
                          height=size_y_false.loc[i] if isinstance(size_y_false, pd.Series) else size_y_false,
                          angle=0, alpha=alpha_values_false.loc[i] if isinstance(alpha_values_false, pd.Series) else alpha_values_false,
                          color=false_color)
         patches_false.append(ellipse)


    # Add patches to the plot
    if patches_real:
        collection_real = PatchCollection(patches_real, match_original=True)
        ax.add_collection(collection_real)

    if patches_false:
        collection_false = PatchCollection(patches_false, match_original=True)
        ax.add_collection(collection_false)


    # 웨이퍼 중심 설정 및 축 범위 조정
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_ylim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, linestyle='--', alpha=0.6)
    # Create proxy artists for the legend
    proxy_real = Circle((0,0), 1, fc=defect_color, alpha=real_alpha, label='Real Defect')
    proxy_false = Circle((0,0), 1, fc=false_color, alpha=false_alpha, label='False Defect')
    ax.legend(handles=[proxy_real, proxy_false])


    plt.show()

# 'Real' 결함만 웨이퍼 맵 그리기 (SNR_OFFSET_GL로 투명도, SIZE_X, SIZE_Y로 크기 조절)
plot_wafer_map(df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL'], title="Wafer Map (Real Defects) with SNR_OFFSET_GL Transparency and SIZE_X, SIZE_Y Size", defect_color='red', false_color='gray', alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y')

# 'False' 결함만 웨이퍼 맵 그리기 (SNR_OFFSET_GL로 투명도, SIZE_X, SIZE_Y로 크기 조절)
plot_wafer_map(df_cleaned[df_cleaned['IS_DEFECT'] == 'FALSE'], title="Wafer Map (False Defects) with SNR_OFFSET_GL Transparency and SIZE_X, SIZE_Y Size", defect_color='gray', false_color='blue', alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y')

## Collect and add patches

### Subtask:
Collect all created Ellipse patches into a `matplotlib.collections.PatchCollection`.


## Adjust plot limits and aspect ratio

### Subtask:
Ensure the plot limits and aspect ratio are correctly set to display the wafer circle and the ellipses within it.


**Reasoning**:
Ensure the plot limits, aspect ratio, title, labels, legend, and grid are correctly set within the plot_wafer_map function.



def plot_wafer_map(df, class_id=None, title="Wafer Map", defect_color='red', false_color='blue', alpha_col=None, size_x_col=None, size_y_col=None, real_alpha=0.8, false_alpha=0.3, size_x_scale=10000, size_y_scale=10000, min_size_x=10, min_size_y=10):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 웨이퍼 배경 (원)
    wafer_radius = 150000
    wafer_circle = Circle((0, 0), wafer_radius, edgecolor='lightgray', facecolor='lightgray', linewidth=0.5)
    ax.add_patch(wafer_circle)

    # Filter data by class if class_id is provided
    if class_id is not None:
        df_filtered = df[df['Class'] == class_id].copy()
        if title == "Wafer Map": # Update title if default and class_id is provided
            title = f"Wafer Map (Class {class_id})"
    else:
        df_filtered = df.copy()


    # Calculate X and Y coordinates from RADIUS and ANGLE
    df_filtered['X'] = df_filtered['RADIUS'] * np.cos(np.deg2rad(df_filtered['ANGLE']))
    df_filtered['Y'] = df_filtered['RADIUS'] * np.sin(np.deg2rad(df_filtered['ANGLE']))

    # 실제 결함
    df_real = df_filtered[df_filtered['IS_DEFECT'] == 'REAL'].copy()
    # 거짓 결함
    df_false = df_filtered[df_filtered['IS_DEFECT'] == 'FALSE'].copy()

    # Determine alpha values based on alpha_col
    if alpha_col and alpha_col in df_filtered.columns:
        # Normalize the column values to a 0-1 range for alpha
        # Add a small value to avoid division by zero if all values are the same
        min_val = df_filtered[alpha_col].min()
        max_val = df_filtered[alpha_col].max()
        alpha_values_real = (df_real[alpha_col] - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) != 0 else real_alpha
        alpha_values_false = (df_false[alpha_col] - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) != 0 else false_alpha
    else:
        alpha_values_real = real_alpha
        alpha_values_false = false_alpha

    # Determine size values based on size_col and new scaling parameters
    size_x_real = min_size_x
    size_y_real = min_size_y
    size_x_false = min_size_x
    size_y_false = min_size_y


    if size_x_col and size_x_col in df_filtered.columns and size_y_col and size_y_col in df_filtered.columns:
        min_val_x = df_filtered[size_x_col].min()
        max_val_x = df_filtered[size_x_col].max()
        min_val_y = df_filtered[size_y_col].min()
        max_val_y = df_filtered[size_y_col].max()

        if (max_val_x - min_val_x) != 0:
             size_x_real = min_size_x + size_x_scale * (df_real[size_x_col] - min_val_x) / (max_val_x - min_val_x)
             size_x_false = min_size_x + size_x_scale * (df_false[size_x_col] - min_val_x) / (max_val_x - min_val_x)
        else:
            size_x_real = min_size_x
            size_x_false = min_size_x


        if (max_val_y - min_val_y) != 0:
            size_y_real = min_size_y + size_y_scale * (df_real[size_y_col] - min_val_y) / (max_val_y - min_val_y)
            size_y_false = min_size_y + size_y_scale * (df_false[size_y_col] - min_val_y) / (max_val_y - min_val_y)
        else:
            size_y_real = min_size_y
            size_y_false = min_size_y


    # Create ellipse patches for real defects
    patches_real = []
    for i, row in df_real.iterrows():
        ellipse = Ellipse((row['X'], row['Y']), width=size_x_real.loc[i] if isinstance(size_x_real, pd.Series) else size_x_real,
                          height=size_y_real.loc[i] if isinstance(size_y_real, pd.Series) else size_y_real,
                          angle=0, alpha=alpha_values_real.loc[i] if isinstance(alpha_values_real, pd.Series) else alpha_values_real,
                          color=defect_color)
        patches_real.append(ellipse)

    # Create ellipse patches for false defects
    patches_false = []
    for i, row in df_false.iterrows():
         ellipse = Ellipse((row['X'], row['Y']), width=size_x_false.loc[i] if isinstance(size_x_false, pd.Series) else size_x_false,
                          height=size_y_false.loc[i] if isinstance(size_y_false, pd.Series) else size_y_false,
                          angle=0, alpha=alpha_values_false.loc[i] if isinstance(alpha_values_false, pd.Series) else alpha_values_false,
                          color=false_color)
         patches_false.append(ellipse)


    # Add patches to the plot
    if patches_real:
        collection_real = PatchCollection(patches_real, match_original=True)
        ax.add_collection(collection_real)

    if patches_false:
        collection_false = PatchCollection(patches_false, match_original=True)
        ax.add_collection(collection_false)


    # 웨이퍼 중심 설정 및 축 범위 조정
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_ylim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("X Position", color='white')
    ax.set_ylabel("Y Position", color='white')
    ax.grid(True, linestyle='--', alpha=0.6)
    # Create proxy artists for the legend
    proxy_real = Circle((0,0), 1, fc=defect_color, alpha=real_alpha, label='Real Defect')
    proxy_false = Circle((0,0), 1, fc=false_color, alpha=false_alpha, label='False Defect')
    plt.xticks(color='white')
    plt.yticks(color='white')
    # Add legend only if both real and false defects are present
    if patches_real and patches_false:
        ax.legend(handles=[proxy_real, proxy_false])
    elif patches_real:
        ax.legend(handles=[proxy_real])
    elif patches_false:
        ax.legend(handles=[proxy_false])


    plt.show()

## Generate separate plots for real and false defects

### Subtask:
Generate separate plots for real and false defects.


**Reasoning**:
Generate the wafer map for 'Real' defects and then for 'False' defects using the updated plotting function with ellipse representation, controlled size and transparency, and correct plot settings.



for i in df_cleaned['Class'].unique():
    print(f"Class {i}")
    print(df_cleaned[df_cleaned['Class']==i]["ACTIVERATIO"].value_counts(), "\n")

def plot_wafer_map(df, class_id=None, step_desc=None, lot_name=None, title="Wafer Map", defect_color='red', false_color='blue', alpha_col=None, size_x_col=None, size_y_col=None, real_alpha=0.8, false_alpha=0.3, size_x_scale=10000, size_y_scale=10000, min_size_x=10, min_size_y=10):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 웨이퍼 배경 (원)
    wafer_radius = 150000
    wafer_circle = Circle((0, 0), wafer_radius, edgecolor='lightgray', facecolor='lightgray', linewidth=0.5)
    ax.add_patch(wafer_circle)

    # Filter data by class, step description, and lot name if provided
    df_filtered = df.copy()
    if class_id is not None:
        df_filtered = df_filtered[df_filtered['Class'] == class_id].copy()
        if title == "Wafer Map":
            title = f"Wafer Map (Class {class_id})"
    if step_desc is not None:
        df_filtered = df_filtered[df_filtered['Step_desc'] == step_desc].copy()
        if title == "Wafer Map" or title.startswith("Wafer Map ("):
             title = f"{title}, {step_desc}" if class_id is not None else f"Wafer Map ({step_desc})"
    if lot_name is not None:
        df_filtered = df_filtered[df_filtered['Lot Name'] == lot_name].copy()
        if title == "Wafer Map":
            title = f"Wafer Map (Lot Name: {lot_name})"
        elif title.startswith("Wafer Map ("):
            title = f"{title}, Lot Name: {lot_name}"
        else:
            title = f"Wafer Map (Lot Name: {lot_name})"


    # Calculate X and Y coordinates from RADIUS and ANGLE
    df_filtered['X'] = df_filtered['RADIUS'] * np.cos(np.deg2rad(df_filtered['ANGLE']))
    df_filtered['Y'] = df_filtered['RADIUS'] * np.sin(np.deg2rad(df_filtered['ANGLE']))

    # 실제 결함
    df_real = df_filtered[df_filtered['IS_DEFECT'] == 'REAL'].copy()
    # 거짓 결함
    df_false = df_filtered[df_filtered['IS_DEFECT'] == 'FALSE'].copy()

    # Determine alpha values based on alpha_col
    if alpha_col and alpha_col in df_filtered.columns:
        # Normalize the column values to a 0-1 range for alpha
        # Add a small value to avoid division by zero if all values are the same
        min_val = df_filtered[alpha_col].min()
        max_val = df_filtered[alpha_col].max()
        alpha_values_real = (df_real[alpha_col] - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) != 0 else real_alpha
        alpha_values_false = (df_false[alpha_col] - min_val) / (max_val - min_val + 1e-6) if (max_val - min_val) != 0 else false_alpha
    else:
        alpha_values_real = real_alpha
        alpha_values_false = false_alpha

    # Determine size values based on size_col and new scaling parameters
    size_x_real = min_size_x
    size_y_real = min_size_y
    size_x_false = min_size_x
    size_y_false = min_size_y


    if size_x_col and size_x_col in df_filtered.columns and size_y_col and size_y_col in df_filtered.columns:
        min_val_x = df_filtered[size_x_col].min()
        max_val_x = df_filtered[size_x_col].max()
        min_val_y = df_filtered[size_y_col].min()
        max_val_y = df_filtered[size_y_col].max()

        if (max_val_x - min_val_x) != 0:
             size_x_real = min_size_x + size_x_scale * (df_real[size_x_col] - min_val_x) / (max_val_x - min_val_x)
             size_x_false = min_size_x + size_x_scale * (df_false[size_x_col] - min_val_x) / (max_val_x - min_val_x)
        else:
            size_x_real = min_size_x
            size_x_false = min_size_x


        if (max_val_y - min_val_y) != 0:
            size_y_real = min_size_y + size_y_scale * (df_real[size_y_col] - min_val_y) / (max_val_y - min_val_y)
            size_y_false = min_size_y + size_y_scale * (df_false[size_y_col] - min_val_y) / (max_val_y - min_val_y)
        else:
            size_y_real = min_size_y
            size_y_false = min_size_y


    # Create ellipse patches for real defects
    patches_real = []
    for i, row in df_real.iterrows():
        ellipse = Ellipse((row['X'], row['Y']), width=size_x_real.loc[i] if isinstance(size_x_real, pd.Series) else size_x_real,
                          height=size_y_real.loc[i] if isinstance(size_y_real, pd.Series) else size_y_real,
                          angle=0, alpha=alpha_values_real.loc[i] if isinstance(alpha_values_real, pd.Series) else alpha_values_real,
                          color=defect_color)
        patches_real.append(ellipse)

    # Create ellipse patches for false defects
    patches_false = []
    for i, row in df_false.iterrows():
         ellipse = Ellipse((row['X'], row['Y']), width=size_x_false.loc[i] if isinstance(size_x_false, pd.Series) else size_x_false,
                          height=size_y_false.loc[i] if isinstance(size_y_false, pd.Series) else size_y_false,
                          angle=0, alpha=alpha_values_false.loc[i] if isinstance(alpha_values_false, pd.Series) else alpha_values_false,
                          color=false_color)
         patches_false.append(ellipse)


    # Add patches to the plot
    if patches_real:
        collection_real = PatchCollection(patches_real, match_original=True)
        ax.add_collection(collection_real)

    if patches_false:
        collection_false = PatchCollection(patches_false, match_original=True)
        ax.add_collection(collection_false)


    # 웨이퍼 중심 설정 및 축 범위 조정
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_ylim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("X Position", color='white')
    ax.set_ylabel("Y Position", color='white')
    ax.grid(True, linestyle='--', alpha=0.6)
    # Create proxy artists for the legend
    proxy_real = Circle((0,0), 1, fc=defect_color, alpha=real_alpha, label='Real Defect')
    proxy_false = Circle((0,0), 1, fc=false_color, alpha=false_alpha, label='False Defect')
    plt.xticks(color='white')
    plt.yticks(color='white')
    # Add legend only if both real and false defects are present
    if patches_real and patches_false:
        ax.legend(handles=[proxy_real, proxy_false])
    elif patches_real:
        ax.legend(handles=[proxy_real])
    elif patches_false:
        ax.legend(handles=[proxy_false])


    plt.show()

# Specify the classes and step descriptions to plot
classes_to_plot = [9]
step_descs_to_plot = ['PC', 'CBCMP', 'RMG'] # Assuming these are the relevant step descriptions

for class_id in classes_to_plot:
    for step_desc in step_descs_to_plot:
        # Filter df_cleaned for the specific class and step description
        df_subset = df_cleaned[(df_cleaned['Class'] == class_id) & (df_cleaned['Step_desc'] == step_desc)]

        # Only plot if the subset is not empty
        if not df_subset.empty:
             plot_wafer_map(df_subset, class_id=class_id, step_desc=step_desc,
                            title=f"Class {class_id} ({step_desc})",
                            defect_color='red', false_color='blue',
                            alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y',
                            size_x_scale=2500, size_y_scale=2500) # Using the same scale as previous class plots

for class_id in df_cleaned['Class'].unique():
    plot_wafer_map(df_cleaned, class_id=class_id,
                                title=f"Class {class_id}",
                                defect_color='red', false_color='blue',
                                alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y',
                                size_x_scale=2500, size_y_scale=2500) # Using the same scale as previous class plots

# Specify the classes and step descriptions to plot
classes_to_plot = [21, 39]
step_descs_to_plot = ['CBCMP', 'RMG'] # Assuming these are the relevant step descriptions

for class_id in classes_to_plot:
    for step_desc in step_descs_to_plot:
        # Filter df_cleaned for the specific class and step description
        df_subset = df_cleaned[(df_cleaned['Class'] == class_id) & (df_cleaned['Step_desc'] == step_desc)]

        # Only plot if the subset is not empty
        if not df_subset.empty:
             plot_wafer_map(df_subset, class_id=class_id, step_desc=step_desc,
                            title=f"Class {class_id} ({step_desc})",
                            defect_color='red', false_color='blue',
                            alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y',
                            size_x_scale=2500, size_y_scale=2500) # Using the same scale as previous class plots



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle

# Assuming df_cleaned is available and contains 'RADIUS', 'ANGLE', 'IS_DEFECT' columns

def plot_wafer_density_map(df, title="Wafer Defect Density Map", defect_type=None, class_id=None, step_desc=None, lot_name=None, wafer_radius=150000):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Filter data by defect type and class if specified
    df_filtered = df.copy()
    if defect_type is not None:
        df_filtered = df_filtered[df_filtered['IS_DEFECT'] == defect_type].copy()
        if title == "Wafer Defect Density Map":
            title = f"{defect_type} Defects Density Map" if class_id is None and step_desc is None and lot_name is None else f"{defect_type} Defects Density Map"

    if class_id is not None:
        df_filtered = df_filtered[df_filtered['Class'] == class_id].copy()
        if title == f"{defect_type} Defects Density Map":
             title = f"{defect_type} Defects Density Map (Class {class_id})"

    if step_desc is not None:
        df_filtered = df_filtered[df_filtered['Step_desc'] == step_desc].copy()
        if title == f"{defect_type} Defects Density Map (Class {class_id})" or title == f"{defect_type} Defects Density Map":
             title = f"{title} ({step_desc})"

    if lot_name is not None:
        df_filtered = df_filtered[df_filtered['Lot Name'] == lot_name].copy()
        if title == "Wafer Defect Density Map":
            title = f"Wafer Defect Density Map (Lot Name: {lot_name})"
        elif title.startswith("Wafer Defect Density Map ("):
            title = f"{title}, Lot Name: {lot_name}"
        else:
            title = f"Wafer Defect Density Map (Lot Name: {lot_name})"


    if df_filtered.empty:
        print(f"No data to plot for Class {class_id if class_id is not None else 'all'} ({defect_type if defect_type is not None else 'all'} defects) for step {step_desc if step_desc is not None else 'all'} and lot {lot_name if lot_name is not None else 'all'}.")
        plt.close(fig) # Close the empty figure
        return

    # Calculate X and Y coordinates from RADIUS and ANGLE
    df_filtered['X'] = df_filtered['RADIUS'] * np.cos(np.deg2rad(df_filtered['ANGLE']))
    df_filtered['Y'] = df_filtered['RADIUS'] * np.sin(np.deg2rad(df_filtered['ANGLE']))

    # Create a Kernel Density Estimate (KDE) plot of defect locations
    # Use the calculated X and Y coordinates
    sns.kdeplot(x=df_filtered['X'], y=df_filtered['Y'], ax=ax, cmap="viridis", fill=True, thresh=0, levels=100)

    # 웨이퍼 배경 (원) - Draw the wafer circle *after* the KDE plot to ensure it's visible
    # 웨이퍼 테두리를 더 명확하게 표시하기 위해 edgecolor를 'black'으로, linewidth를 2로 변경
    wafer_circle = Circle((0, 0), wafer_radius, edgecolor='black', facecolor='none', linewidth=2) # facecolor='none' to not cover the heatmap
    ax.add_patch(wafer_circle)


    # 웨이퍼 중심 설정 및 축 범위 조정
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_ylim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X Position", color='white') # X-axis label color is already white
    ax.set_ylabel("Y Position", color='white') # Y-axis label color is already white
    ax.grid(False) # Hide grid

    # Set tick label colors to white
    plt.xticks(color='white')
    plt.yticks(color='white')


    plt.show()

# Generate density maps for each unique Class and Step_desc combination for Real Defects
# unique_combinations = df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL'][['Class', 'Step_desc']].drop_duplicates()

# for index, row in unique_combinations.iterrows():
#     class_id = row['Class']
#     step_desc = row['Step_desc']

#     plot_wafer_density_map(df_cleaned,
#                            title=f"Real Defects Density Map (Class {class_id}, {step_desc})",
#                            class_id=class_id,
#                            step_desc=step_desc,
#                            defect_type='REAL')

# Example of plotting for a specific Class, Step_desc, and Lot Name
# plot_wafer_density_map(df_cleaned, class_id=9, step_desc='PC', lot_name='F00BK.JO', defect_type='REAL', title="Real Defects Density Map (Class 9, PC, Lot F00BK.JO)")

# Example of plotting for a specific Class and Lot Name
# plot_wafer_density_map(df_cleaned, class_id=21, lot_name='F04V5.1', defect_type='REAL', title="Real Defects Density Map (Class 21, Lot F04V5.1)")

# Example of plotting for a specific Lot Name (all classes and steps)
# plot_wafer_density_map(df_cleaned, lot_name='F00BK.JO', defect_type='REAL', title="Real Defects Density Map (Lot F00BK.JO)")

unique_combinations = df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL'][['Class', 'Step_desc']].drop_duplicates()

for index, row in unique_combinations.iterrows():
    class_id = row['Class']
    step_desc = row['Step_desc']

    plot_wafer_density_map(df_cleaned,
                           title=f"Real Defects Density Map (Class {class_id}, {step_desc})",
                           class_id=class_id,
                           step_desc=step_desc,
                           defect_type='REAL')

unique_combinations = df_cleaned[df_cleaned['IS_DEFECT'] == 'FALSE'][['Class', 'Step_desc']].drop_duplicates()

for index, row in unique_combinations.iterrows():
    class_id = row['Class']
    step_desc = row['Step_desc']

    plot_wafer_density_map(df_cleaned,
                        title=f"거짓 결함 Density Map (Class {class_id}, {step_desc})",
                        class_id=class_id,
                        step_desc=step_desc,
                        defect_type='FALSE')

df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL']

# 'Real' 결함만 웨이퍼 맵 그리기 (SNR_OFFSET_GL로 투명도, SIZE_X, SIZE_Y로 크기 조절)
plot_wafer_map(df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL'], title="실제 결함 웨이퍼맵", defect_color='red', false_color='gray', alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y')

# 'False' 결함만 웨이퍼 맵 그리기 (SNR_OFFSET_GL로 투명도, SIZE_X, SIZE_Y로 크기 조절)
plot_wafer_map(df_cleaned[df_cleaned['IS_DEFECT'] == 'FALSE'], title="거짓 결함 웨이퍼맵", defect_color='gray', false_color='blue', alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y')

## Summary:

### Data Analysis Key Findings

* Separate wafer maps were successfully generated for 'Real' and 'False' defects.
* Each defect is visualized as an ellipse whose dimensions are scaled based on the 'SIZE_X' and 'SIZE_Y' columns.
* The transparency of each ellipse is adjusted according to the corresponding value in the 'SNR_OFFSET_GL' column, providing a visual representation of this metric.
* The plots correctly display the defects within a wafer boundary of radius 150000.

### Insights or Next Steps

* Analyze the generated maps to identify any spatial patterns in 'Real' or 'False' defects and their correlation with 'SIZE_X', 'SIZE_Y', and 'SNR_OFFSET_GL'.
* Consider adding a color bar to indicate the mapping of 'SNR_OFFSET_GL' values to transparency levels for better interpretability of the maps.

* * *

## Summary:

### Data Analysis Key Findings

*   Separate wafer maps were successfully generated for 'Real' and 'False' defects.
*   Each defect is visualized as an ellipse whose dimensions are scaled based on the 'SIZE_X' and 'SIZE_Y' columns.
*   The transparency of each ellipse is adjusted according to the corresponding value in the 'SNR_OFFSET_GL' column, providing a visual representation of this metric.
*   The plots correctly display the defects within a wafer boundary of radius 150000.

### Insights or Next Steps

*   Analyze the generated maps to identify any spatial patterns in 'Real' or 'False' defects and their correlation with 'SIZE_X', 'SIZE_Y', and 'SNR_OFFSET_GL'.
*   Consider adding a color bar to indicate the mapping of 'SNR_OFFSET_GL' values to transparency levels for better interpretability of the maps.

---


plt.figure(figsize=(6, 4))
sns.kdeplot(data=df_cleaned, x='SNR_OFFSET_GL', hue='IS_DEFECT', common_norm=False, fill=True, alpha=0.5, palette={'REAL': 'red', 'FALSE': 'blue'})
plt.title('SNR_OFFSET_GL KDE 분포')
plt.xlim(-0.15,0.85)
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df_cleaned, x='DEFECT_AREA', hue='IS_DEFECT', common_norm=False, fill=True, alpha=0.5, palette={'REAL': 'red', 'FALSE': 'blue'})
plt.title('DEFECT_AREA KDE 분포')
plt.xlim(-0.0035,0.022)
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df_cleaned, x='SPOTLIKENESS', hue='IS_DEFECT', common_norm=False, fill=True, alpha=0.5, palette={'REAL': 'red', 'FALSE': 'blue'})
plt.title('SPOTLIKENESS KDE 분포')
plt.xlim(right=25)
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df_cleaned, x='RELATIVEMAGNITUDE', hue='IS_DEFECT', common_norm=False, fill=True, alpha=0.5, palette={'REAL': 'red', 'FALSE': 'blue'})
plt.title('RELATIVEMAGNITUDE KDE 분포')
plt.xlim(right=1050)
plt.show()

## 데이터 전처리 및 분할

### Subtask:
분류 모델 학습을 위해 데이터를 전처리하고 훈련 세트와 테스트 세트로 분할합니다.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Assuming df_cleaned is available and contains the necessary columns

# Select features (X) and target (y)
# We will use the numerical features for now, and exclude the ID-like columns and target variable
features = [
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SIZE_D', 'INTENSITY', 'POLARITY',
    'ENERGY_PARAM', 'MDAT_OFFSET', 'MDAT_GL', 'MDAT_NOISE', 'RADIUS', 'ANGLE',
    'ALIGNRATIO', 'SPOTLIKENESS', 'PATCHNOISE', 'RELATIVEMAGNITUDE',
    'ACTIVERATIO', 'PATCHDEFECTSIGNAL', 'SNR_OFFSET_GL', 'SNR_INTENSITY_NOISE',
    'ASPECT_RATIO', 'DENSITY_SIGNAL'
] # Exclude 'Step', 'Step_desc', 'Lot Name', 'Slot No', 'Defect No', 'IS_DEFECT', 'Class'

target = 'Class'

# Drop rows with missing values in features or target
df_processed = df_cleaned.dropna(subset=features + [target]).copy()


X = df_processed[features]
y = df_processed[target]

# Scale numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("데이터 전처리 및 분할 완료.")
print(f"훈련 세트 크기: {X_train.shape[0]} (Feature: {X_train.shape[1]})")
print(f"테스트 세트 크기: {X_test.shape[0]} (Feature: {X_test.shape[1]})")
print("\n훈련 세트 피처 데이터 미리보기:")
display(X_train.head())
print("\n훈련 세트 타겟 데이터 미리보기:")
display(y_train.head())

## 모델 평가 (Random Forest)

### Subtask:
훈련된 Random Forest 모델의 성능을 테스트 데이터셋을 사용하여 평가합니다. 정확도, 정밀도, 재현율, F1-score, 혼동 행렬 등 적절한 평가 지표를 사용합니다.

**Reasoning**:
Evaluate the trained Random Forest model using common classification metrics such as accuracy, precision, recall, F1-score, and a confusion matrix to understand its performance on the test data.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming rf_model, X_test, and y_test are available from previous steps

if 'rf_model' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
    print("Error: Trained model or test data not found. Please ensure previous steps are executed.")
else:
    print("--- Random Forest 모델 평가 시작 ---")

    # Predict on the test data
    y_pred = rf_model.predict(X_test)

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    # Get unique classes from both y_test and y_pred to ensure all classes are in the matrix
    all_classes = sorted(list(set(y_test.unique()).union(set(y_pred))))
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)

    # Display confusion matrix using a heatmap for better visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    print("\nRandom Forest 모델 평가 완료.")

## 모델 선택 및 훈련 (XGBoost)

### Subtask:
XGBoost Classifier 모델을 선택하고 훈련 데이터로 학습시킵니다.

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Assuming X_train and y_train are available from the previous data splitting step

if 'X_train' not in locals() or 'y_train' not in locals():
    print("Error: Training data (X_train, y_train) not found. Please ensure the data preprocessing and splitting cell has been executed.")
else:
    print("--- XGBoost 모델 훈련 시작 ---")

    # Initialize and train the XGBoost Classifier
    # Using default parameters for now, can be tuned later
    # Convert target variable to numeric and then encode to be zero-indexed for XGBoost
    if not pd.api.types.is_numeric_dtype(y_train):
        # Convert to numeric if not already, then encode
        y_train_numeric = y_train.astype(float) # Ensure numeric type
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_numeric)
        # Save the encoder for inverse transformation later if needed
        xgb_label_encoder = label_encoder
        print("Target variable 'Class' encoded for XGBoost.")
    else:
        # If already numeric, just encode to be zero-indexed
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        xgb_label_encoder = label_encoder
        print("Target variable 'Class' encoded for XGBoost.")


    xgb_model = xgb.XGBClassifier(objective='multi:softmax', random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train_encoded)

    print("\nXGBoost 모델 훈련 완료.")

## 모델 평가 (XGBoost)

### Subtask:
훈련된 XGBoost 모델의 성능을 테스트 데이터셋을 사용하여 평가합니다. 정확도, 정밀도, 재현율, F1-score, 혼동 행렬 등 적절한 평가 지표를 사용합니다.

**Reasoning**:
Evaluate the trained XGBoost model using common classification metrics such as accuracy, precision, recall, F1-score, and a confusion matrix to understand its performance on the test data and compare it with the Random Forest model.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming xgb_model, X_test, y_test, and xgb_label_encoder are available from previous steps

if 'xgb_model' not in locals() or 'X_test' not in locals() or 'y_test' not in locals() or 'xgb_label_encoder' not in locals():
    print("Error: Trained model or test data/encoder not found. Please ensure previous steps are executed.")
else:
    print("--- XGBoost 모델 평가 시작 ---")

    # Predict on the test data
    y_pred_encoded = xgb_model.predict(X_test)

    # Decode the predicted labels back to original class labels
    y_pred = xgb_label_encoder.inverse_transform(y_pred_encoded)

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    # Get unique classes from both y_test and y_pred to ensure all classes are in the matrix
    all_classes = sorted(list(set(y_test.unique()).union(set(y_pred))))
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)

    # Display confusion matrix using a heatmap for better visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
    plt.title('Confusion Matrix (XGBoost)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    print("\nXGBoost 모델 평가 완료.")

## 모델 선택 및 훈련 (Random Forest)

### Subtask:
Random Forest Classifier 모델을 선택하고 훈련 데이터로 학습시킵니다.

**Reasoning**:
The user has requested to use Random Forest or XGBoost. I will proceed by training a Random Forest model as it is a suitable choice for classification tasks and the necessary library is already imported.

from sklearn.ensemble import RandomForestClassifier

# Assuming X_train and y_train are available from the previous data splitting step

if 'X_train' not in locals() or 'y_train' not in locals():
    print("Error: Training data (X_train, y_train) not found. Please ensure the data preprocessing and splitting cell has been executed.")
else:
    print("--- Random Forest 모델 훈련 시작 ---")

    # Initialize and train the Random Forest Classifier
    # Using default parameters for now, can be tuned later
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    print("\nRandom Forest 모델 훈련 완료.")



# Task
Perform K-means clustering on the real defect data (`df_real_defect`) using numerical features to identify data-driven groups (potential defect types), analyze the characteristics of each cluster, and visualize the results.

## 군집 분석 데이터 준비

### Subtask:
실제 결함 데이터(`df_real_defect`)에서 군집 분석에 사용할 수치형 특성을 선택하고, 필요한 경우 데이터 스케일링을 수행합니다.


**Reasoning**:
Select numerical features for clustering from the real defect data and apply standard scaling to prepare the data for K-means clustering.



df_real_defect

from sklearn.preprocessing import StandardScaler

# Select numerical features for clustering based on user's specified list
numerical_features_for_clustering = [
    'ENERGY_PARAM', 'MDAT_OFFSET', 'RELATIVEMAGNITUDE', 'PATCHDEFECTSIGNAL',
    'INTENSITY', 'POLARITY', 'MDAT_GL', 'MDAT_NOISE', 'PATCHNOISE',
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SIZE_D', 'RADIUS', 'ANGLE', 'ALIGNRATIO',
    'SPOTLIKENESS', 'ACTIVERATIO'
]

# Ensure only real defect data is used
df_real_numerical = df_real_defect[numerical_features_for_clustering].copy()

# Handle potential missing values before scaling (e.g., impute with mean or median, or drop rows)
# For simplicity in this example, we will drop rows with missing values in the selected columns
df_real_numerical.dropna(inplace=True)

# Keep track of the original index before scaling and dropping NaNs
original_indices = df_real_numerical.index

# Apply Standard Scaling
scaler = StandardScaler()
df_real_scaled = pd.DataFrame(scaler.fit_transform(df_real_numerical), columns=numerical_features_for_clustering, index=original_indices)

print("Original numerical data shape for clustering:", df_real_numerical.shape)
print("Scaled numerical data shape for clustering:", df_real_scaled.shape)
print("\nScaled data preview:")
display(df_real_scaled.head())



**Reasoning**:
The numerical features have been selected and scaled. The next step is to determine the optimal number of clusters (k) for K-means clustering using methods like the Elbow method or Silhouette analysis.



**Reasoning**:
The previous code failed because the x and y arrays for the silhouette plot had different lengths. This is because the silhouette score is calculated for k > 1, so the `silhouette_scores` list has one fewer element than `k_range`. I need to adjust the x-axis range for the silhouette plot to match the length of the `silhouette_scores` list.



**Reasoning**:
The error persists because `k_range[1:]` generates a range from 3 to 10 (8 values), while `silhouette_scores` contains scores for k=2 to 10 (9 values). I need to ensure the x-axis range for the silhouette plot correctly corresponds to the k values for which silhouette scores were calculated (k=2 to 10). I will create a new list for the x-axis of the silhouette plot.



**Reasoning**:
The error indicates that the `silhouette_k_values` list still has a different length than `silhouette_scores`. The `silhouette_scores` list contains scores for k=2 to 10, meaning it has 9 elements. `k_range[1:]` starts from k=3, resulting in only 8 values. The x-axis for the silhouette plot should be k=2 to 10. I will correctly create the x-axis values for the silhouette plot.



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt # Ensure matplotlib.pyplot is imported if not already in the environment

# Use the Elbow method to find the optimal number of clusters
inertia = []
silhouette_scores = []
k_range = range(2, 11) # Test k from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_real_scaled)
    inertia.append(kmeans.inertia_)

    # Calculate silhouette score (only for k > 1)
    if k > 1:
        try:
            silhouette_avg = silhouette_score(df_real_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        except Exception as e:
            print(f"Error calculating silhouette score for k={k}: {e}")
            silhouette_scores.append(np.nan) # Append NaN if calculation fails


# Plot the Elbow method results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)

# Plot the Silhouette scores
plt.subplot(1, 2, 2)
# Correctly create a list of k values corresponding to the silhouette scores (k=2 to 10)
# The length of silhouette_scores list should be len(k_range) - 1
silhouette_k_values = list(k_range[1:]) # k=2 to 10, length 9
# Ensure the x-axis values match the calculated silhouette scores
# If silhouette_scores was populated for k=2, 3, ..., 10, its length is 9.
# The k values are 2, 3, ..., 10.
silhouette_k_values_for_plot = list(range(k_range[0], k_range[-1] + 1)) # k=2 to 10
if len(silhouette_k_values_for_plot) != len(silhouette_scores):
     print(f"Warning: Mismatch in length between silhouette_k_values_for_plot ({len(silhouette_k_values_for_plot)}) and silhouette_scores ({len(silhouette_scores)}). Adjusting x-axis for silhouette plot.")
     # Use the actual k values that successfully produced a silhouette score
     successful_k_for_silhouette = [k_range[i] for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
     plt.plot(successful_k_for_silhouette, [score for score in silhouette_scores if not np.isnan(score)], marker='o')
     plt.xticks(successful_k_for_silhouette)
else:
     plt.plot(silhouette_k_values_for_plot, silhouette_scores, marker='o')
     plt.xticks(silhouette_k_values_for_plot)


plt.title('Silhouette Scores For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

df_cleaned[df_cleaned['IS_DEFECT']=='REAL']



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score # Import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the process steps to analyze
process_steps_to_analyze = ['PC', 'RMG', 'CBCMP']

# Define the numerical features for clustering (same as before)
numerical_features_for_clustering = [
    'ENERGY_PARAM', 'MDAT_OFFSET', 'RELATIVEMAGNITUDE', 'PATCHDEFECTSIGNAL',
    'INTENSITY', 'POLARITY', 'MDAT_GL', 'MDAT_NOISE', 'PATCHNOISE',
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SIZE_D', 'RADIUS', 'ANGLE', 'ALIGNRATIO',
    'SPOTLIKENESS', 'ACTIVERATIO'
]

# Dictionary to store clustering results for each step
process_step_clustering_results = {}

print("--- 공정 단계별 K-Means 군집화 시작 (k=2) ---")

for step_desc in process_steps_to_analyze:
    print(f"\n>>> Processing Step: {step_desc} <<<")

    # 1. Filter real defect data for the current process step
    df_step_real_defect = df_cleaned[(df_cleaned['IS_DEFECT'] == 'REAL') & (df_cleaned['Step_desc'] == step_desc)].copy()

    if df_step_real_defect.empty:
        print(f"  No real defect data found for step '{step_desc}'. Skipping.")
        continue

    # 2. Prepare numerical features: handle missing values and scale
    df_step_numerical = df_step_real_defect[numerical_features_for_clustering].copy()
    df_step_numerical.dropna(inplace=True)

    if df_step_numerical.empty:
        print(f"  No valid numerical data after dropping NaNs for step '{step_desc}'. Skipping.")
        continue

    # Keep track of original indices after dropping NaNs
    original_indices_step = df_step_numerical.index

    scaler = StandardScaler()
    df_step_scaled = pd.DataFrame(scaler.fit_transform(df_step_numerical),
                                  columns=numerical_features_for_clustering, index=original_indices_step)

    # --- Evaluate optimal k using Elbow Method for the current step ---
    inertia_step = []
    k_range_elbow = range(1, 11) # Test k from 1 to 10 for Elbow

    for k_val in k_range_elbow:
        kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        kmeans_elbow.fit(df_step_scaled)
        inertia_step.append(kmeans_elbow.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range_elbow, inertia_step, marker='o')
    plt.title(f'{step_desc} 공정 Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_range_elbow)
    plt.grid(True)
    plt.show()

    # 3. Perform K-Means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels_step = kmeans.fit_predict(df_step_scaled)

    # Add cluster labels to the filtered DataFrame
    df_step_clustered = df_step_real_defect.loc[original_indices_step].copy()
    df_step_clustered['Cluster'] = cluster_labels_step

    process_step_clustering_results[step_desc] = df_step_clustered

    print(f"  K-Means Clustering complete for {step_desc}.")
    print(f"  Cluster counts:\n{df_step_clustered['Cluster'].value_counts()}")

    # --- Calculate and print Silhouette Score for k=2 clustering ---
    if len(df_step_scaled) > 1 and len(df_step_clustered['Cluster'].unique()) > 1:
        silhouette_avg_step = silhouette_score(df_step_scaled, cluster_labels_step)
        print(f"  Silhouette Score for k=2: {silhouette_avg_step:.4f}")
    else:
        print(f"  Not enough data or clusters to calculate Silhouette Score for k=2.")

    # 4. Analyze characteristics of each cluster
    print(f"\n  --- Cluster Analysis for {step_desc} ---")
    print("  Mean of numerical features per cluster:")
    display(df_step_clustered.groupby('Cluster')[numerical_features_for_clustering].mean())

    print("  Original 'Class' distribution (proportions) per cluster:")
    cluster_class_dist = df_step_clustered.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)
    cluster_class_prop = cluster_class_dist.divide(cluster_class_dist.sum(axis=1), axis=0)
    display(cluster_class_prop)

    # 5. Visualize clusters using PCA
    print(f"\n  --- PCA Visualization for {step_desc} ---")
    pca = PCA(n_components=2)
    df_pca_step = pca.fit_transform(df_step_scaled)

    df_pca_step = pd.DataFrame(df_pca_step, columns=['PCA1', 'PCA2'], index=df_step_scaled.index)
    df_pca_step['Cluster'] = df_step_clustered['Cluster']

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca_step, palette='viridis', legend='full')
    plt.title(f'{step_desc} 공정 K-Means 군집 결과 (PCA)')
    plt.xlabel('주성분 1 (PCA1)')
    plt.ylabel('주성분 2 (PCA2)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 6. Display PCA Component Loadings for the current step
    print(f"\n  --- {step_desc} 공정 PCA Component Loadings (주성분 적재값) ---")
    feature_names_step = df_step_scaled.columns
    pca_loadings_step = pca.components_

    df_pca_loadings_step = pd.DataFrame(pca_loadings_step,
                                      columns=feature_names_step,
                                      index=['PCA1', 'PCA2'])
    display(df_pca_loadings_step)

    plt.figure(figsize=(12, 5))
    sns.heatmap(df_pca_loadings_step, annot=True, cmap='vlag', fmt='.2f')
    plt.title(f'{step_desc} 공정 PCA Component Loadings Heatmap')
    plt.show()

print("\n--- 공정 단계별 K-Means 군집화 완료 ---")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_cleaned, process_step_clustering_results, and numerical_features_for_clustering are available

# Define the killer cluster ID for each process step based on previous analysis
killer_cluster_mapping = {
    'PC': 1,  # In PC, Cluster 1 was the killer defect cluster
    'RMG': 1, # In RMG, Cluster 1 was the killer defect cluster
    'CBCMP': 0 # In CBCMP, Cluster 0 was the killer defect cluster
}

# Define the process steps to analyze
process_steps_to_analyze = ['PC', 'RMG', 'CBCMP']

# Select key numerical features for comparison
comparison_features = [
    'DEFECT_AREA',
    'SPOTLIKENESS',
    'SNR_OFFSET_GL',
    'INTENSITY',
    'SIZE_X',
    'SIZE_Y',
    'RELATIVEMAGNITUDE',
    'PATCHDEFECTSIGNAL',
    'MDAT_OFFSET',
    'MDAT_GL',
    'MDAT_NOISE',
    'PATCHNOISE'
]

for step_desc in process_steps_to_analyze:
    print(f"\n--- 공정: {step_desc} - '킬러 결함' vs '거짓 결함' 특성 비교 시작 ---")

    # Get the step-specific clustered real defect data
    if step_desc not in process_step_clustering_results:
        print(f"  No clustering results found for step '{step_desc}'. Skipping.")
        continue

    df_step_clustered_real = process_step_clustering_results[step_desc].copy()

    # Filter false defects for the current process step
    df_step_false_defect = df_cleaned[
        (df_cleaned['IS_DEFECT'] == 'FALSE') &
        (df_cleaned['Step_desc'] == step_desc)
    ].copy()

    if df_step_false_defect.empty:
        print(f"  No false defect data found for step '{step_desc}'. Skipping comparison with false defects.")
        df_step_false_viz = pd.DataFrame() # Empty DataFrame for false defects
    else:
        df_step_false_viz = df_step_false_defect.copy()
        df_step_false_viz['Group'] = '거짓 결함 (Class 9)'

    # Prepare the real defect clusters for visualization
    # We need to distinguish between the 'killer' cluster and the 'non-killer' cluster within real defects
    killer_cluster_id = killer_cluster_mapping.get(step_desc)
    if killer_cluster_id is None:
        print(f"  Killer cluster ID not defined for step '{step_desc}'. Skipping detailed real defect cluster comparison.")
        df_cluster_killer = pd.DataFrame()
        df_cluster_non_killer = pd.DataFrame()
    else:
        df_cluster_killer = df_step_clustered_real[df_step_clustered_real['Cluster'] == killer_cluster_id].copy()
        df_cluster_killer['Group'] = f'Killer Cluster {killer_cluster_id} (Real)'

        non_killer_cluster_ids = [c for c in df_step_clustered_real['Cluster'].unique() if c != killer_cluster_id]
        if non_killer_cluster_ids:
            df_cluster_non_killer = df_step_clustered_real[df_step_clustered_real['Cluster'].isin(non_killer_cluster_ids)].copy()
            # If there's only one non-killer cluster, label it simply as 'Non-Killer Cluster X (Real)'
            # If there are multiple, they will be combined under a generic label or shown separately if more than 2 clusters were used
            if len(non_killer_cluster_ids) == 1:
                 df_cluster_non_killer['Group'] = f'Non-Killer Cluster {non_killer_cluster_ids[0]} (Real)'
            else:
                 df_cluster_non_killer['Group'] = 'Other Real Defects (Non-Killer Clusters)'
        else:
            df_cluster_non_killer = pd.DataFrame()

    # Combine the dataframes for visualization
    df_comparison_step = pd.concat([df_step_false_viz, df_cluster_non_killer, df_cluster_killer], ignore_index=True)

    # Define the order of groups for visualization
    group_order_step = []
    if not df_step_false_viz.empty: group_order_step.append('거짓 결함 (Class 9)')
    if not df_cluster_non_killer.empty: group_order_step.append(df_cluster_non_killer['Group'].iloc[0]) # Assuming unique label
    if not df_cluster_killer.empty: group_order_step.append(df_cluster_killer['Group'].iloc[0]) # Assuming unique label

    # Generate violin plots for selected features to compare distributions
    print(f"\n  --- 공정: {step_desc} - 주요 수치형 특성 분포 비교 (Violin Plot) ---")
    for feature in comparison_features:
        if feature not in df_comparison_step.columns: # Skip if feature is missing
            print(f"    Warning: Feature '{feature}' not found in current step's data. Skipping.")
            continue
        plt.figure(figsize=(10, 7))
        sns.violinplot(x='Group', y=feature, data=df_comparison_step, order=group_order_step, palette='viridis')
        plt.title(f'[{step_desc} 공정] {feature} 분포 비교: 거짓 결함 vs 킬러/비-킬러 실제 결함')
        plt.xlabel('그룹')
        plt.ylabel(feature)
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Generate stacked bar plot for original Class distribution
    print(f"\n  --- 공정: {step_desc} - 원본 Class 분포 비교 (Stacked Bar Plot) ---")
    # Prepare data for Class distribution comparison (counts)
    class_data_list = []
    if not df_step_false_viz.empty:
        class_counts_false = df_step_false_viz['Class'].value_counts().reset_index()
        class_counts_false.columns = ['Class', 'Count']
        class_counts_false['Group'] = '거짓 결함 (Class 9)'
        class_data_list.append(class_counts_false)

    if not df_cluster_non_killer.empty:
        class_counts_non_killer = df_cluster_non_killer['Class'].value_counts().reset_index()
        class_counts_non_killer.columns = ['Class', 'Count']
        class_counts_non_killer['Group'] = df_cluster_non_killer['Group'].iloc[0]
        class_data_list.append(class_counts_non_killer)

    if not df_cluster_killer.empty:
        class_counts_killer = df_cluster_killer['Class'].value_counts().reset_index()
        class_counts_killer.columns = ['Class', 'Count']
        class_counts_killer['Group'] = df_cluster_killer['Group'].iloc[0]
        class_data_list.append(class_counts_killer)

    if class_data_list:
        df_class_comparison_step = pd.concat(class_data_list, ignore_index=True)
        df_class_pivot_step = df_class_comparison_step.pivot_table(index='Group', columns='Class', values='Count', fill_value=0)

        plt.figure(figsize=(12, 7))
        # Use .loc with group_order_step to ensure consistent order
        df_class_pivot_step.loc[group_order_step].plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
        plt.title(f'[{step_desc} 공정] 원본 Class 분포 비교: 거짓 결함 vs 킬러/비-킬러 실제 결함')
        plt.xlabel('그룹')
        plt.ylabel('결함 수')
        plt.xticks(rotation=30, ha='right')
        plt.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print(f"  No class distribution data to plot for step '{step_desc}'.")

    print(f"--- 공정: {step_desc} - 특성 비교 완료 ---")

print("\n--- 모든 공정 단계별 특성 비교 완료 ---")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_real_defect_clustered is available from previous K-Means clustering steps
# Assuming df_false_defect is available (filtered from df_cleaned where IS_DEFECT == 'FALSE')

if 'df_real_defect_clustered' not in locals() or 'df_false_defect' not in locals():
    print("Error: Required DataFrames (df_real_defect_clustered or df_false_defect) not found. Please ensure previous steps are executed.")
else:
    print("--- '킬러 결함' (K-Means 군집 0, 1) vs '거짓 결함' (Class 9) 특성 비교 시작 ---")

    # Select data for Cluster 0 and Cluster 1 from the K-Means real defect clustering
    df_cluster_0 = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == 0].copy()
    df_cluster_0['Group'] = 'Cluster 0 (Real Defects)'

    df_cluster_1 = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == 1].copy()
    df_cluster_1['Group'] = 'Cluster 1 (Real Defects)'

    # Use the pre-existing df_false_defect for comparison
    df_false_viz = df_false_defect.copy()
    df_false_viz['Group'] = 'False Defects (Class 9)' # Assuming Class 9 is primarily False Defects

    # Combine the dataframes for visualization
    df_comparison = pd.concat([df_cluster_0, df_cluster_1, df_false_viz], ignore_index=True)

    # Define the order of groups for visualization
    group_order = ['False Defects (Class 9)', 'Cluster 0 (Real Defects)', 'Cluster 1 (Real Defects)']

    # Select key numerical features for comparison.
    # These include size, intensity, and especially noise-related features.
    comparison_features = [
        'DEFECT_AREA',
        'SPOTLIKENESS',
        'SNR_OFFSET_GL',
        'INTENSITY',
        'SIZE_X',
        'SIZE_Y',
        'RELATIVEMAGNITUDE',
        'PATCHDEFECTSIGNAL',
        'MDAT_OFFSET',
        'MDAT_GL',
        'MDAT_NOISE',
        'PATCHNOISE'
    ]

    print("\n--- 주요 수치형 특성 분포 비교 (Violin Plot) ---")
    # Generate violin plots for selected features to compare distributions
    for feature in comparison_features:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Group', y=feature, data=df_comparison, order=group_order, palette='viridis')
        plt.title(f'{feature} 분포 비교: K-Means 군집 0 vs 1 vs 거짓 결함')
        plt.xlabel('그룹')
        plt.ylabel(feature)
        plt.xticks(rotation=15) # Rotate x-axis labels for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    print("\n--- 원본 Class 분포 비교 (Stacked Bar Plot) ---")
    # Create a combined dataframe with counts for plotting Class distribution
    class_counts_cluster_0 = df_cluster_0['Class'].value_counts().reset_index()
    class_counts_cluster_0.columns = ['Class', 'Count']
    class_counts_cluster_0['Group'] = 'Cluster 0 (Real Defects)'

    class_counts_cluster_1 = df_cluster_1['Class'].value_counts().reset_index()
    class_counts_cluster_1.columns = ['Class', 'Count']
    class_counts_cluster_1['Group'] = 'Cluster 1 (Real Defects)'

    class_counts_false = df_false_viz['Class'].value_counts().reset_index()
    class_counts_false.columns = ['Class', 'Count']
    class_counts_false['Group'] = 'False Defects (Class 9)'

    df_class_comparison = pd.concat([class_counts_cluster_0, class_counts_cluster_1, class_counts_false], ignore_index=True)

    # Pivot the data for stacked bar plot
    df_class_pivot = df_class_comparison.pivot_table(index='Group', columns='Class', values='Count', fill_value=0)

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    df_class_pivot.loc[group_order].plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.title('원본 Class 분포 비교: K-Means 군집 0 vs 1 vs 거짓 결함')
    plt.xlabel('그룹')
    plt.ylabel('결함 수')
    plt.xticks(rotation=15) # Rotate x-axis labels for better readability
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("--- '킬러 결함' (K-Means 군집 0, 1) vs '거짓 결함' (Class 9) 특성 비교 완료 ---")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `process_step_clustering_results` is available from previous steps
# and contains df_step_clustered for each step.

plot_data = []

# Define the killer cluster ID for each process step based on previous analysis
# PC: Cluster 1 (high feature values, 100% Class 22)
# RMG: Cluster 1 (high feature values, ~78% Class 20)
# CBCMP: Cluster 0 (high feature values, ~66% Class 21)
killer_cluster_mapping = {
    'PC': 1,
    'RMG': 1,
    'CBCMP': 0
}

for step_desc, df_step_clustered in process_step_clustering_results.items():
    killer_cluster_id = killer_cluster_mapping.get(step_desc)

    if killer_cluster_id is None:
        print(f"Warning: Killer cluster ID not defined for step '{step_desc}'. Skipping.")
        continue

    for class_id in df_step_clustered['Class'].unique():
        total_class_defects_in_step = len(df_step_clustered[df_step_clustered['Class'] == class_id])

        # Count defects of this class that fall into the designated 'killer' cluster for this step
        killer_class_defects_in_step = len(df_step_clustered[
            (df_step_clustered['Class'] == class_id) &
            (df_step_clustered['Cluster'] == killer_cluster_id)
        ])

        if total_class_defects_in_step > 0:
            killer_proportion = killer_class_defects_in_step / total_class_defects_in_step
            non_killer_proportion = 1 - killer_proportion

            plot_data.append({
                'Step': step_desc,
                'Class': class_id,
                'Killer_Proportion': killer_proportion,
                'Non_Killer_Proportion': non_killer_proportion
            })

df_plot = pd.DataFrame(plot_data)

# Create a combined label for x-axis: Class (Step_desc)
df_plot['Class_Step_Label'] = df_plot['Class'].astype(str) + ' (' + df_plot['Step'] + ')'

# Sort the combined data by Step then Class number as requested
df_plot_sorted = df_plot.sort_values(by=['Step', 'Class'], ascending=[True, True]).reset_index(drop=True)

# Melt the DataFrame for stacked bar plotting
df_melted_sorted = df_plot_sorted.melt(
    id_vars=['Step', 'Class', 'Class_Step_Label'],
    value_vars=['Killer_Proportion', 'Non_Killer_Proportion'],
    var_name='Defect_Type',
    value_name='Proportion'
)

# Rename defect types for better labels in the legend
df_melted_sorted['Defect_Type'] = df_melted_sorted['Defect_Type'].replace({
    'Killer_Proportion': '킬러 결함',
    'Non_Killer_Proportion': '비-킬러 결함'
})

# Define the order of bars for the x-axis to maintain the sorting
order_list = df_plot_sorted['Class_Step_Label'].tolist()

plt.figure(figsize=(12, 8)) # Adjusted figsize to make it more square-like
sns.barplot(
    data=df_melted_sorted,
    x='Class_Step_Label',
    y='Proportion',
    stacked=True,
    hue='Defect_Type',
    palette={'킬러 결함': 'red', '비-킬러 결함': 'skyblue'}, # Distinct colors for killer/non-killer
    dodge=False, # Crucial for creating stacked bars
    order=order_list # Apply the sorted order to the bars
)

plt.title('공정별 클래스 내 킬러 결함 비율 (100% 누적 막대 그래프)', fontsize=18)
plt.xlabel('클래스 (공정)', fontsize=14)
plt.ylabel('비율', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate and adjust font size for x-axis labels
plt.ylim(0, 1) # Proportions should range from 0 to 1
plt.legend(title='결함 유형', bbox_to_anchor=(1.02, 1), loc='upper left') # Place legend outside the plot
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid for readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 0. 샘플 데이터 생성 (사용자님의 실제 데이터 구조와 유사하게 가정) ---
# 실제 df_melted_sorted 데이터를 여기에 대입해주세요.
# 여기서는 예시를 위해 임의의 데이터를 생성합니다.
# 각 Class_Step_Label 내에서 킬러/비킬러 비율을 합쳤을 때 1이 되도록 조정 (stacked bar의 전제)
# 예시 데이터에서는 무작위로 생성했으므로, 실제 데이터에서는 이 과정이 필요 없을 수 있습니다.
for label in df_melted_sorted['Class_Step_Label'].unique():
    subset = df_melted_sorted[df_melted_sorted['Class_Step_Label'] == label]
    total_proportion = subset['Proportion'].sum()
    if total_proportion > 0:
        df_melted_sorted.loc[subset.index, 'Proportion'] = subset['Proportion'] / total_proportion


# 킬러 결함 비율을 기준으로 정렬 순서 생성 (sns.barplot의 order와 동일)
# 여기서는 '킬러 결함'의 비율을 기준으로 내림차순 정렬한다고 가정합니다.
killer_proportions = df_melted_sorted[df_melted_sorted['Defect_Type'] == '킬러 결함'] \
                      .groupby('Class_Step_Label')['Proportion'].sum() \
                      .sort_values(ascending=False)
order_list = df_plot_sorted['Class_Step_Label'].tolist()

# --- 1. df_melted_sorted를 plot() 메서드 형식에 맞게 재구성 ---
# 피벗 테이블을 사용하여 'Class_Step_Label'을 인덱스로, 'Defect_Type'을 컬럼으로 만듭니다.
# values='Proportion'은 각 셀의 값으로 'Proportion'을 사용합니다.
class_cluster_proportion = df_melted_sorted.pivot_table(
    index='Class_Step_Label',
    columns='Defect_Type',
    values='Proportion',
    fill_value=0 # 값이 없는 경우 0으로 채웁니다.
)

# order_list에 맞춰 인덱스 정렬
class_cluster_proportion = class_cluster_proportion.loc[order_list]


# --- 2. 색상 정의 ---
colors = {'킬러 결함': '#35B779', '비-킬러 결함': '#31688E'}
# class_cluster_proportion의 컬럼 순서에 맞게 색상 리스트를 생성해야 합니다.
# (예: class_cluster_proportion 컬럼이 ['비-킬러 결함', '킬러 결함'] 순이라면)
# sorted_colors = [colors[col] for col in class_cluster_proportion.columns]
# 컬럼 순서가 고정되어있지 않다면, 아래처럼 dict를 직접 넘겨주는게 더 안전합니다.
order_list = df_plot_sorted['Class_Step_Label'].tolist()


# --- 3. 그래프 그리기 (DataFrame.plot() 사용) ---
plt.figure(figsize=(12, 8)) # 적절한 크기로 조절
ax = plt.gca() # 현재 Axes 객체를 가져옵니다.

class_cluster_proportion.plot(
    kind='bar',
    stacked=True,
    color=colors, # 딕셔너리로 컬럼명-색상 매핑 전달
    ax=ax,
    width=0.7, # 막대 너비 조절
    legend=False
)

plt.title('', fontsize=22)
plt.ylabel('결함 비율', fontsize=18)
plt.xlabel('Class_Step_Label', fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(rotation=45, ha='right', fontsize=16) # x축 레이블 회전
# plt.legend(title='결함 유형')
plt.tight_layout() # 그래프 요소들이 잘리지 않도록 조정
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Ensure pandas is imported if not already in the environment

# Assuming df_cleaned, process_step_clustering_results are available

# Define colors for clusters (e.g., Cluster 0, Cluster 1)
# Cluster 0 will be default_colors[0], Cluster 1 will be default_colors[1]
default_colors = ['#31688E', '#35B779'] # Using a distinct palette

process_steps_to_analyze = ['PC', 'RMG', 'CBCMP']

fig, axes = plt.subplots(1, len(process_steps_to_analyze), figsize=(len(process_steps_to_analyze) * 5, 6), sharey=True)

if len(process_steps_to_analyze) == 1:
    axes = [axes]

print("--- 공정 단계별 Class별 군집 분포 비율 (하나의 Figure) ---")

for i, step_desc in enumerate(process_steps_to_analyze):
    # Get the step-specific clustered real defect data
    if step_desc not in process_step_clustering_results:
        print(f"  No clustering results found for step '{step_desc}'. Skipping.")
        continue

    df_step_clustered_real = process_step_clustering_results[step_desc].copy()

    # Filter for real defects with assigned clusters for the current step
    df_step_real_clustered = df_step_clustered_real[(df_step_clustered_real['IS_DEFECT'] == 'REAL') & (df_step_clustered_real['Cluster'].notna())].copy()

    if df_step_real_clustered.empty:
        print(f"Error: No real defects with assigned K-Means clusters found for step {step_desc}.")
        continue

    # Group by Class and Cluster, count occurrences, and unstack
    class_cluster_distribution_step = df_step_real_clustered.groupby(['Class', 'Cluster']).size().unstack(fill_value=0)

    # Calculate the total number of defects for each Class
    total_per_class_step = df_step_real_clustered.groupby('Class').size()

    # Calculate the proportion of each Cluster within each Class
    class_cluster_proportion_step = class_cluster_distribution_step.divide(total_per_class_step, axis=0)

    # --- Conditional color and stacking order assignment ---
    if step_desc == 'CBCMP':
        # Invert colors for CBCMP to make Cluster 0 (killer) green and Cluster 1 (non-killer) blue
        # We want Cluster 1 (blue) at the bottom, so columns should be [1, 0]
        plot_df = class_cluster_proportion_step[[1, 0]]
        plot_colors = [default_colors[0], default_colors[1]] # Cluster 1 gets default[0] (blue), Cluster 0 gets default[1] (green)
    else:
        # For PC/RMG, Cluster 1 is killer (green) and Cluster 0 is non-killer (blue)
        # Default order [0, 1] puts Cluster 0 (blue) at bottom, Cluster 1 (green) on top
        plot_df = class_cluster_proportion_step[[0, 1]] # Ensure consistent column order
        plot_colors = [default_colors[0], default_colors[1]] # Cluster 0 gets default[0] (blue), Cluster 1 gets default[1] (green)

    # Plot the stacked bar chart on the current subplot, now with legend=False
    plot_df.plot(kind='bar', stacked=True, color=plot_colors, ax=axes[i], legend=False)

    axes[i].set_title(f'[{step_desc} 공정] Class별 군집 분포 비율', fontsize=14)
    axes[i].set_xlabel('Class', fontsize=12)
    axes[i].set_ylabel('비율', fontsize=12) # Only apply to the first subplot or if sharey=False
    axes[i].tick_params(axis='x', rotation=0, labelsize=10)
    axes[i].set_ylim(0, 1) # Proportions should range from 0 to 1
    axes[i].grid(axis='y', linestyle='--', alpha=0.6) # Add grid for readability

# Remove the global legend (already commented out, but just in case)
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, title='KMeans Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to prevent labels/legend from overlapping
plt.show()



# df_real_scaled에서 원본 변수(Feature) 이름들을 가져옵니다.
feature_names = df_real_scaled.columns

# pca.components_에 저장된 기여도(Loadings)를 가져옵니다.
pca_loadings = pca.components_

# 보기 쉽게 DataFrame으로 만듭니다.
df_pca_loadings = pd.DataFrame(pca_loadings,
                             columns=feature_names,
                             index=['PCA1', 'PCA2'])

print("--- PCA Component Loadings (주성분 적재값) ---")
display(df_pca_loadings)

plt.figure(figsize=(12, 5))
sns.heatmap(df_pca_loadings, annot=True, cmap='vlag', fmt='.2f')
plt.title('PCA Component Loadings Heatmap')
plt.show()



## K-평균 군집화 수행

### Subtask:
결정된 최적의 군집 개수(K=3)로 K-평균 군집화를 수행하고 각 실제 결함 데이터 포인트에 군집 라벨을 할당합니다.

**Reasoning**:
Perform K-Means clustering on the scaled real defect data (`df_real_scaled`) using the chosen number of clusters (K=3) and add the resulting cluster labels to the original `df_real_defect` DataFrame.

from sklearn.cluster import KMeans

# Assuming df_real_scaled is available from previous steps

if 'df_real_scaled' not in locals():
    print("Error: 'df_real_scaled' DataFrame not found. Please ensure the data scaling cell has been executed.")
else:
    # Perform K-Means clustering with K=4 (based on Elbow and Silhouette analysis)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_real_scaled)

    # Add the cluster labels to the original df_real_defect DataFrame
    # Ensure the indices match between df_real_defect and cluster_labels
    # We need to use the original_indices saved during scaling if NaNs were dropped
    if 'original_indices' in locals():
        df_real_defect_clustered = df_real_defect.loc[original_indices].copy()
        df_real_defect_clustered['Cluster'] = cluster_labels
    else:
        # If no NaNs were dropped during scaling, indices should match directly
        df_real_defect_clustered = df_real_defect.copy()
        df_real_defect_clustered['Cluster'] = cluster_labels


    print("실제 결함 데이터에 군집 라벨 추가:")
    display(df_real_defect_clustered.head())

    # Display the count of data points in each cluster
    print("\n군집별 데이터 포인트 개수:")
    display(df_real_defect_clustered['Cluster'].value_counts())

df_c1_22 = df_real_defect_clustered[(df_real_defect_clustered['Cluster']==1)&(df_real_defect_clustered['Class']==22)]

df_c1_20 = df_real_defect_clustered[(df_real_defect_clustered['Cluster']==1)&(df_real_defect_clustered['Class']==20)]

df_real_defect_clustered[df_real_defect_clustered['Lot Name']=='A1VWN.1']

total_defects_per_lot

killer_defect_counts

lot_proportion_distribution[lot_proportion_distribution['Lot Name']=='A1X6H.JA']

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_cleaned is available with 'KMeans_Cluster' column
# Define the killer cluster ID for each process step based on previous analysis
killer_cluster_mapping = {
    'PC': 1,
    'RMG': 1,
    'CBCMP': 0
}

# 1. 'is_killer_defect' 컬럼 생성
# df_cleaned에 'is_killer_defect' 컬럼을 추가하고 기본값은 False로 설정합니다.
# Real 결함이 아니거나, KMeans_Cluster가 없는 경우는 킬러 결함으로 간주하지 않습니다.
df_cleaned['is_killer_defect'] = False

# 공정 단계별 킬러 군집 맵핑을 적용하여 'is_killer_defect' 플래그 설정
for step_desc, killer_cluster_id in killer_cluster_mapping.items():
    # 해당 공정의 실제 결함 중, 킬러 군집에 속하는 결함을 True로 설정
    mask = (df_cleaned['IS_DEFECT'] == 'REAL') & \
           (df_cleaned['Step_desc'] == step_desc) & \
           (df_cleaned['KMeans_Cluster'] == killer_cluster_id)
    df_cleaned.loc[mask, 'is_killer_defect'] = True

# 2. Lot Name별 킬러 결함 수, 전체 실제 결함 수, 거짓 결함 수, Slot 개수 계산

# Lot Name별 킬러 결함 수 계산
killer_defect_counts = df_cleaned[df_cleaned['is_killer_defect'] == True] \
                               .groupby('Lot Name').size().reset_index(name='Killer_Defect_Count')

# Lot Name별 전체 실제 결함 수 계산
total_real_defects_per_lot = df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL'] \
                                     .groupby('Lot Name').size().reset_index(name='Total_Count')

# Lot Name별 거짓 결함 수 계산 (Class 9로 가정)
false_defect_counts_per_lot = df_cleaned[df_cleaned['IS_DEFECT'] == 'FALSE'] \
                                      .groupby('Lot Name').size().reset_index(name='False_Defect_Count')

# Lot Name별 고유한 Slot 개수 계산
unique_slots_per_lot = df_cleaned.groupby('Lot Name')['Slot No'].nunique().reset_index(name='Slot_No_nunique')

# 3. 모든 카운트를 Lot Name 기준으로 병합
# 먼저 킬러 결함 수와 전체 실제 결함 수를 병합
lot_analysis_df = pd.merge(total_real_defects_per_lot, killer_defect_counts, on='Lot Name', how='left')
# 킬러 결함이 없는 Lot의 경우 0으로 채움
lot_analysis_df['Killer_Defect_Count'].fillna(0, inplace=True)

# 거짓 결함 수 병합
lot_analysis_df = pd.merge(lot_analysis_df, false_defect_counts_per_lot, on='Lot Name', how='left')
# 거짓 결함이 없는 Lot의 경우 0으로 채움
lot_analysis_df['False_Defect_Count'].fillna(0, inplace=True)

# 고유한 Slot 개수 병합
lot_analysis_df = pd.merge(lot_analysis_df, unique_slots_per_lot, on='Lot Name', how='left')
# Slot_No_nunique가 없는 Lot의 경우 1로 채움 (나누기 오류 방지, 최소 1개의 슬롯은 있다고 가정)
lot_analysis_df['Slot_No_nunique'].fillna(1, inplace=True)

# Killer_Defect_Proportion 계산 (기존의 비율은 유지)
lot_analysis_df['Killer_Defect_Proportion'] = lot_analysis_df['Killer_Defect_Count'] / lot_analysis_df['Total_Count']

# 결과 미리보기
print("Lot Name별 킬러 결함 및 전체 결함 카운트 미리보기:")
display(lot_analysis_df.head())

# 이 DataFrame을 기반으로 'df_lot'을 재정의하고 위험도 점수를 계산합니다.
df_lot = lot_analysis_df.copy()


df_cleaned

display(df_cleaned.head())

df_false_defect = df_cleaned[df_cleaned['Class']==9]
total_false_per_lot = df_false_defect.groupby('Lot Name').size().reset_index(name='False_Defect_Count')
lot_proportion_distribution2 = pd.merge(lot_proportion_distribution, total_false_per_lot, on='Lot Name')

lot_proportion_distribution2

from sklearn.preprocessing import MinMaxScaler

# Assuming df_lot is prepared from the previous step with updated killer defect logic

# -----------------------------------------------------------------
# --- 2. (핵심 로직) 통합 위험도 점수(Total_Risk_Score) 계산 ---
# -----------------------------------------------------------------

# 1단계: '무해 결함(Nuisance_Count)' 수 계산
df_lot['Nuisance_Count'] = df_lot['Total_Count'] - df_lot['Killer_Defect_Count']

# 새로운 지표: Slot당 결함 수 (밀도)
# Slot_No_nunique가 0이 되는 것을 방지하기 위해 1e-6을 더해줍니다.
df_lot['Killer_Defect_Count_per_slot'] = df_lot['Killer_Defect_Count'] / (df_lot['Slot_No_nunique'] + 1e-6)
df_lot['Nuisance_Count_per_slot'] = df_lot['Nuisance_Count'] / (df_lot['Slot_No_nunique'] + 1e-6)
df_lot['False_Defect_Count_per_slot'] = df_lot['False_Defect_Count'] / (df_lot['Slot_No_nunique'] + 1e-6)

# 2단계: 점수 정규화 (Min-Max Scaling) - Slot당 결함 수 사용
scaler_killer = MinMaxScaler()
scaler_nuisance = MinMaxScaler()
scaler_false = MinMaxScaler()

# 0~1 사이의 'Score_Killer' (킬러 점수) 생성 - Slot당 결함 수 기준
df_lot['Score_Killer'] = scaler_killer.fit_transform(df_lot[['Killer_Defect_Count_per_slot']])
# 0~1 사이의 'Score_Nuisance' (무해 점수) 생성 - Slot당 결함 수 기준
df_lot['Score_Nuisance'] = scaler_nuisance.fit_transform(df_lot[['Nuisance_Count_per_slot']])
# 'False_Defect_Count' -> 'Score_False' - Slot당 결함 수 기준
df_lot['Score_False'] = scaler_false.fit_transform(df_lot[['False_Defect_Count_per_slot']])

# 2단계: 3-Level 가중치 설정 (비즈니스 중요도)
# (총합 = 1.0)
w_killer = 0.50  # 50% (가장 심각한 실제 결함 - 킬러 결함)
w_nuisance = 0.30  # 30% (덜 심각한 실제 결함 - 무해 결함)
w_false    = 0.20  # 20% (거짓 결함 - 리소스 낭비)

# 3단계: 가중 합계로 '최종 위험도 점수' 계산
df_lot['Total_Risk_Score'] = (w_killer * df_lot['Score_Killer']) + \
                             (w_nuisance * df_lot['Score_Nuisance']) + \
                             (w_false    * df_lot['Score_False'])


# --- 3. 결과 확인 ---
# 'Total_Risk_Score'(최종 위험도 점수)가 높은 순서대로 정렬
df_sorted_by_risk_final = df_lot.sort_values(by='Total_Risk_Score', ascending=False)

print("--- 3-Level 통합 위험도(Risk Score)가 가장 높은 최악의 Lot --- ")
# 분석에 사용된 모든 컬럼을 함께 출력
display(df_sorted_by_risk_final)


import pandas as pd

# Assuming df_cleaned is available with 'is_killer_defect' column
# Assuming killer_cluster_mapping is defined as before

if 'df_cleaned' not in locals() or 'is_killer_defect' not in df_cleaned.columns:
    print("Error: 'df_cleaned' or 'is_killer_defect' column not found. Please ensure previous steps are executed.")
else:
    print("--- df_cleaned에서 확인하는 각 공정별 Class에 따른 킬러 결함 비율 ---")

    # Get unique process steps from df_cleaned that have real defects
    process_steps_with_real_defects = df_cleaned[df_cleaned['IS_DEFECT'] == 'REAL']['Step_desc'].unique()

    for step_desc in process_steps_with_real_defects:
        print(f"\n>>> 공정: {step_desc} <<<")

        # Filter df_cleaned for real defects in the current step
        df_step_real_defects = df_cleaned[
            (df_cleaned['IS_DEFECT'] == 'REAL') &
            (df_cleaned['Step_desc'] == step_desc)
        ].copy()

        if df_step_real_defects.empty:
            print(f"  해당 공정({step_desc})에는 실제 결함 데이터가 없습니다.")
            continue

        # Calculate total defects per class in this step
        total_defects_per_class = df_step_real_defects.groupby('Class').size().reset_index(name='Total_Count')

        # Calculate killer defects per class in this step
        killer_defects_per_class = df_step_real_defects[
            df_step_real_defects['is_killer_defect'] == True
        ].groupby('Class').size().reset_index(name='Killer_Count')

        # Merge to calculate proportions
        df_class_proportions = pd.merge(
            total_defects_per_class,
            killer_defects_per_class,
            on='Class',
            how='left'
        )
        df_class_proportions['Killer_Count'] = df_class_proportions['Killer_Count'].fillna(0)

        df_class_proportions['Killer_Proportion'] = (
            df_class_proportions['Killer_Count'] / df_class_proportions['Total_Count']
        )

        # Display the result for the current step
        display(df_class_proportions[['Class', 'Killer_Proportion']].sort_values(by='Killer_Proportion', ascending=False).style.format({'Killer_Proportion': '{:.2%}'}))

    print("\n--- df_cleaned 기반 킬러 결함 비율 확인 완료 ---")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 시각화할 상위 N개 Lot 선택
top_n = 15
df_top_n = df_sorted_by_risk_final.head(top_n)

# (이전 코드의 가중치와 동일하게 설정)
w_killer = 0.50
w_nuisance = 0.30
w_false    = 0.20

# 스택을 쌓기 위해 3가지 위험도 컴포넌트를 계산
risk_comp_false = w_false * df_top_n['Score_False']
risk_comp_nuisance = w_nuisance * df_top_n['Score_Nuisance']
risk_comp_killer = w_killer * df_top_n['Score_Killer']

lot_names = df_top_n['Lot Name']
indices = np.arange(len(lot_names))

# --- 푸른색 계열 팔레트 설정 ---
# 'Blues' 또는 'viridis', 'cividis', 'mako' 등 다양한 푸른 계열 팔레트가 있습니다.
# 여기서는 'Blues' 계열과 마지막만 다른 색으로 강조하는 방법을 사용합니다.
# 'Killer' 결함이 가장 중요하므로 가장 진한 색으로 강조합니다.
colors = ['#ADD8E6',   # LightBlue (거짓 결함)
          '#6A5ACD',   # SlateBlue (무해 결함)
          '#4682B4']   # SteelBlue (킬러 결함)

plt.figure(figsize=(15, 8))

# 1. '거짓 결함' 점수 (가장 아래쪽 막대)
bar_false = plt.bar(
    indices,
    risk_comp_false,
    label=f'거짓 결함 위험도 (가중치 {w_false*100}%)',
    color=colors[0]
)

# 2. '무해 결함' 점수 (중간 막대)
bar_nuisance = plt.bar(
    indices,
    risk_comp_nuisance,
    bottom=risk_comp_false,
    label=f'무해 결함 위험도 (가중치 {w_nuisance*100}%)',
    color=colors[1]
)

# 3. '킬러 결함' 점수 (가장 위쪽 막대)
bar_killer = plt.bar(
    indices,
    risk_comp_killer,
    bottom=risk_comp_false + risk_comp_nuisance,
    label=f'킬러 결함 위험도 (가중치 {w_killer*100}%)',
    color=colors[2]
)

plt.title(f'최악의 Lot Top {top_n} (3-Level 통합 위험도 기준)', fontsize=22)
plt.ylabel('통합 위험도 점수', fontsize=16)
plt.xlabel('Lot Name', fontsize=16)
plt.xticks(indices, lot_names, rotation=45, ha='right')
plt.legend(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


df_sorted_by_risk_final['Total_Risk_Score'].std()

df_top_15 = df_sorted_by_risk_final.head(15)
top_15_lot_names_list = df_top_15['Lot Name'].tolist()

print("--- 최악의 Lot Top 15 리스트 ---")
print(top_15_lot_names_list)

df_top_15[['Lot Name', 'Total_Risk_Score']]

# 1. Lot Name별로 유일한 Step_desc 확인
df_filtered = df[df['Lot Name'].isin(top_15_lot_names_list)]
step_desc_by_lot = df_filtered.groupby('Lot Name')['Step_desc'].unique()

# 결과 출력
for lot_name, step_descs in step_desc_by_lot.items():
    print(f"--- {lot_name} ---")
    for step_desc in step_descs:
        print(step_desc)
    print("\n")



df['MDAT_OFFSET'] / df['MDAT_GL']

import seaborn as sns

# (df_sorted_by_risk_final 데이터프레임이 준비되었다고 가정)

# 버블 크기 스케일링 (0~1 사이 점수를 50~2000 사이 크기로 매핑)
# Total_Risk_Score가 0에 가까울 수도 있으니, 최소 크기를 보장해줍니다.
min_bubble = 50
max_bubble = 2000
bubble_size = (df_sorted_by_risk_final['Total_Risk_Score'] * (max_bubble - min_bubble)) + min_bubble

plt.figure(figsize=(14, 9))

# X축: 킬러 결함 비율 (질)
# Y축: 전체 결함 수 (양)
# Size: 3-Level 통합 위험도 점수 (Total_Risk_Score)
# Hue: 3-Level 통합 위험도 점수 (Total_Risk_Score)
scatter = sns.scatterplot(
    data=df_sorted_by_risk_final,
    x='Killer_Defect_Proportion',
    y='Total_Count',
    size=bubble_size,
    sizes=(min_bubble, max_bubble),
    hue='Total_Risk_Score',
    palette='Reds', # 위험도가 높을수록 붉게
    alpha=0.7,
    legend='auto'
)

# --- 평균선 (사분면 기준선) 그리기 ---
plt.axhline(
    y=df_sorted_by_risk_final['Total_Count'].mean(),
    color='gray',
    linestyle='--',
    label='전체 결함 수 (평균)'
)
plt.axvline(
    x=df_sorted_by_risk_final['Killer_Defect_Proportion'].mean(),
    color='blue',
    linestyle='--',
    label='킬러 결함 비율 (평균)'
)

plt.title('Lot 위험도 사분면 분석 (3-Level Risk Score 반영)', fontsize=16)
plt.xlabel('킬러 결함 비율 (Proportion) - (질적 위험)', fontsize=12)
plt.ylabel('전체 결함 수 (Total Count) - (양적 위험)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



print("--- Killer Defect (Classes 20, 22, 28 in Cluster 1) Lot Name Distribution (Top 10% by Proportion) ---")

# Visualize the distribution for each class using a horizontal bar chart (Top 10% by Proportion)
for class_id in sorted(df_killer_defects_c1['Class'].unique()):
    df_class_lot_proportion = lot_proportion_distribution[lot_proportion_distribution['Class'] == class_id].copy()

    # Sort by proportion in descending order and select the top 10%
    df_class_lot_proportion_top10 = df_class_lot_proportion.sort_values(by='Killer_Defect_Proportion', ascending=False).head(int(len(df_class_lot_proportion) * 0.1))

    if not df_class_lot_proportion_top10.empty:
        plt.figure(figsize=(10, df_class_lot_proportion_top10.shape[0] * 0.5)) # Adjust figure height
        sns.barplot(x='Killer_Defect_Proportion', y='Lot Name', data=df_class_lot_proportion_top10, palette='viridis')
        plt.title(f'Class {class_id} (Cluster 1) Lot Name Distribution (Top 10% by Proportion)')
        plt.xlabel('Proportion of Killer Defects')
        plt.ylabel('Lot Name')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data for Class {class_id} in Cluster 1.")

print("\n--- Filtered Lot Name Distribution Visualization Complete ---")

for lot_name in df_c1_20['Lot Name'].unique():
    plot_wafer_map(df_c1_20, lot_name=lot_name,
                title=f"Class 20",
                defect_color='red', false_color='blue',
                alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y',
                size_x_scale=2500, size_y_scale=2500) # Using the same scale as previous class plots

for lot_name in df_c1_22['Lot Name'].unique():
    plot_wafer_map(df_c1_22, lot_name=lot_name,
                title=f"Class 22",
                defect_color='red', false_color='blue',
                alpha_col='SNR_OFFSET_GL', size_x_col='SIZE_X', size_y_col='SIZE_Y',
                size_x_scale=2500, size_y_scale=2500) # Using the same scale as previous class plots

for lot_name in df_c1_22['Lot Name'].unique():
    plot_wafer_density_map(df_c1_22, lot_name=lot_name,
                            title=f"실제 결함 Density Map",
                            defect_type='REAL')

## K-Means 군집 결과 웨이퍼 맵 시각화

K-Means 군집 결과를 웨이퍼 맵에 시각화합니다. 각 결함은 해당 군집 라벨에 따라 색상이 지정됩니다.

# Assuming df_real_defect_clustered and plot_wafer_map function are available

if 'df_real_defect_clustered' not in locals() or 'plot_wafer_map' not in locals():
    print("Error: Required DataFrames or function not found. Please ensure previous steps are executed.")
else:
    print("--- K-Means 군집 결과 웨이퍼 맵 시각화 ---")

    # Use the plot_wafer_map function to visualize clustered real defects
    # Assign colors based on the 'Cluster' column
    # We need to create a color mapping for the clusters
    unique_clusters = sorted(df_real_defect_clustered['Cluster'].unique())
    colors = sns.color_palette('viridis', len(unique_clusters)) # Use a palette with enough colors
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # Prepare data for plotting - need to combine real and false defects for plot_wafer_map
    # False defects will be plotted with a neutral color
    df_clustered_viz = df_real_defect_clustered.copy()
    df_clustered_viz['Display_Color'] = df_clustered_viz['Cluster'].map(cluster_color_map)

    df_false_viz = df_false_defect.copy()
    df_false_viz['Display_Color'] = 'gray' # Neutral color for false defects
    df_false_viz['Cluster'] = -1 # Assign a dummy cluster for false defects

    df_combined_wafer = pd.concat([df_clustered_viz, df_false_viz], ignore_index=True)

    # Plot the wafer map
    # We need to modify plot_wafer_map or adapt the plotting logic here to handle custom colors per point
    # Let's create a simplified plotting loop here for demonstration
    fig, ax = plt.subplots(figsize=(10, 10))

    # 웨이퍼 배경 (원)
    wafer_radius = 150000
    wafer_circle = Circle((0, 0), wafer_radius, edgecolor='lightgray', facecolor='lightgray', linewidth=0.5)
    ax.add_patch(wafer_circle)

    # Calculate X and Y coordinates
    df_combined_wafer['X'] = df_combined_wafer['RADIUS'] * np.cos(np.deg2rad(df_combined_wafer['ANGLE']))
    df_combined_wafer['Y'] = df_combined_wafer['RADIUS'] * np.sin(np.deg2rad(df_combined_wafer['ANGLE']))

    # Plot each defect as a point with its assigned color
    # Using scatter plot for simplicity, ellipses can be added later if needed
    scatter = ax.scatter(df_combined_wafer['X'], df_combined_wafer['Y'], c=df_combined_wafer['Display_Color'], s=10, alpha=0.6) # Adjust size (s) and alpha as needed

    # 웨이퍼 중심 설정 및 축 범위 조정
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_ylim(-wafer_radius * 1.1, wafer_radius * 1.1)
    ax.set_title('K-Means 군집 결과 웨이퍼 맵', fontsize=16)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create a legend for the clusters and false defects
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster}',
                                  markerfacecolor=color, markersize=10) for cluster, color in cluster_color_map.items()]
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='False Defects',
                                      markerfacecolor='gray', markersize=10))
    ax.legend(handles=legend_elements, title='Cluster')


    plt.show()

## K-Means 군집별 KDE 밀도 분포 웨이퍼 맵 시각화

K-Means 군집별로 KDE(커널 밀도 추정) 밀도 분포 웨이퍼 맵을 생성합니다.

# Assuming df_real_defect_clustered and plot_wafer_density_map function are available

if 'df_real_defect_clustered' not in locals() or 'plot_wafer_density_map' not in locals():
    print("Error: Required DataFrames or function not found. Please ensure previous steps are executed.")
else:
    print("--- K-Means 군집별 KDE 밀도 분포 웨이퍼 맵 시각화 ---")

    # Generate density maps for each K-Means cluster
    unique_clusters = sorted(df_real_defect_clustered['Cluster'].unique())

    for cluster_id in unique_clusters:
        # Filter data for the specific cluster
        df_cluster_subset = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == cluster_id].copy()

        # Only plot if the subset is not empty
        if not df_cluster_subset.empty:
             plot_wafer_density_map(df_cluster_subset,
                                    title=f"Real Defects Density Map (K-Means Cluster {cluster_id})",
                                    defect_type='REAL') # Pass defect_type as filtering is already done by cluster



## 군집 결과 분석

### Subtask:
생성된 군집별로 수치형 특성의 기술 통계 및 원본 'Class' 분포를 분석하여 각 군집의 특성을 파악합니다.

**Reasoning**:
Analyze the characteristics of the identified clusters by calculating descriptive statistics for the numerical features within each cluster and examining the distribution of the original 'Class' labels within each cluster.

# Assuming df_real_defect_clustered is available from previous steps and numerical_features_for_clustering is defined

if 'df_real_defect_clustered' not in locals():
    print("Error: 'df_real_defect_clustered' DataFrame not found. Please ensure the clustering cell has been executed.")
elif 'numerical_features_for_clustering' not in locals():
     print("Error: 'numerical_features_for_clustering' not found. Please ensure the data preparation cell for clustering has been executed.")
else:
    print("--- 군집 결과 분석 ---")

    # Analyze the mean of numerical features within each cluster
    cluster_means = df_real_defect_clustered.groupby('Cluster')[numerical_features_for_clustering].mean()

    print("\n군집별 수치형 특성 평균:")
    display(cluster_means)

    # Analyze the distribution of original 'Class' within each cluster
    # Group by Cluster and Class, count occurrences, and unstack
    cluster_class_distribution = df_real_defect_clustered.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)

    # Calculate the proportion of each Class within each cluster (Original Request)
    cluster_class_proportion = cluster_class_distribution.divide(cluster_class_distribution.sum(axis=1), axis=0)

    print("\n군집별 원본 Class 분포 비율 (각 군집 내 Class 비율):")
    display(cluster_class_proportion)

    # Calculate the proportion of each Cluster within each Class (New Request)
    # Group by Class and Cluster, count occurrences, and unstack
    class_cluster_distribution = df_real_defect_clustered.groupby(['Class', 'Cluster']).size().unstack(fill_value=0)

    # Calculate the total number of defects for each Class
    total_per_class = df_real_defect_clustered.groupby('Class').size()

    # Calculate the proportion of each Cluster within each Class
    # Use .divide with axis=0 to divide each row by the corresponding total from total_per_class
    class_cluster_proportion = class_cluster_distribution.divide(total_per_class, axis=0)

    print("\nClass별 군집 분포 비율 (각 Class가 속하는 군집 비율):")
    display(class_cluster_proportion)

import pandas as pd

# Assuming class_cluster_proportion is available from the previous step (cell AO2WQtuq-4Bj)

if 'class_cluster_proportion' not in locals():
    print("Error: 'class_cluster_proportion' DataFrame not found. Please ensure the previous cell (AO2WQtuq-4Bj) has been executed.")
else:
    print("--- Class별 K-Means 군집 0, 1 비율 (표) ---")
    display(class_cluster_proportion.style.format('{:.2%}'))


## 군집 결과 시각화

### Subtask:
군집 분석 결과를 시각화하여 각 군집의 주요 특성 분포 및 원본 'Class' 분포를 제시합니다.

**Reasoning**:
Visualize the clustering results by generating box plots for key numerical features across clusters and a stacked bar plot showing the distribution of original 'Class' labels within each cluster. This provides visual insights into the characteristics of the identified clusters.

df_false_defect

# Assuming df_real_defect_clustered is available from previous steps and numerical_features_for_clustering is defined
# Assuming df_false_defect is available for comparison

if 'df_real_defect_clustered' not in locals():
    print("Error: 'df_real_defect_clustered' DataFrame not found. Please ensure the clustering cell has been executed.")
elif 'numerical_features_for_clustering' not in locals():
     print("Error: 'numerical_features_for_clustering' not found. Please ensure the data preparation cell for clustering has been executed.")
elif 'df_false_defect' not in locals():
     print("Error: 'df_false_defect' DataFrame not found. Cannot compare with False Defects.")
else:
    print("--- K-평균 군집별 주요 수치형 특성 분포 (박스 플롯) 및 거짓 결함 비교 ---")

    # Combine clustered real defects and false defects for comparison visualization
    df_real_clustered_viz = df_real_defect_clustered.copy()
    df_real_clustered_viz['Group'] = 'Cluster ' + df_real_clustered_viz['Cluster'].astype(str)

    df_false_viz = df_false_defect.copy()
    df_false_viz['Group'] = 'False Defects (Class 9)'

    df_combined_viz = pd.concat([df_real_clustered_viz, df_false_viz], ignore_index=True)

    # Define the order of groups for visualization (False Defects first, then clusters)
    group_order = ['False Defects (Class 9)'] + [f'Cluster {i}' for i in sorted(df_real_defect_clustered['Cluster'].unique())]

    # Define palette (optional, but can help differentiate groups)
    palette_viz = sns.color_palette('viridis', df_real_defect_clustered['Cluster'].nunique())
    palette_dict_viz = {f'Cluster {i}': palette_viz[i] for i in sorted(df_real_defect_clustered['Cluster'].unique())}
    palette_dict_viz['False Defects (Class 9)'] = 'lightcoral' # Assign a color for False Defects


    # Visualize distribution of numerical features across clusters and False Defects using box plots
    for col in numerical_features_for_clustering:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Group', y=col, data=df_combined_viz, order=group_order, palette=palette_dict_viz)
        plt.title(f'군집별 {col} 분포 비교')
        plt.xlabel('그룹')
        plt.ylabel(col)
        plt.xticks(rotation=0)
        plt.show()

    print("\n--- K-평균 군집별 원본 Class 분포 (누적 막대 그래프) ---")
    # This plot is specific to the Class distribution *within* real defect clusters,
    # so Class 9 cannot be directly included here in a meaningful way.
    # We will keep the original plot showing Class distribution within real clusters.
    # Group by Cluster and Class, count occurrences, and unstack
    cluster_class_distribution = df_real_defect_clustered.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)

    # Calculate the proportion of each Class within each cluster
    cluster_class_proportion = cluster_class_distribution.divide(cluster_class_distribution.sum(axis=1), axis=0)

    # Plot the stacked bar chart
    plt.figure(figsize=(6, 4))
    cluster_class_proportion.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.title('K-평균 군집별 원본 Class 분포 비율')
    plt.xlabel('군집')
    plt.ylabel('비율')
    plt.xticks(rotation=0)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

df_false_viz = df_false_defect.copy()
df_false_viz['Group'] = '거짓 결함'

df_combined_viz = pd.concat([df_real_clustered_viz, df_false_viz], ignore_index=True)

group_order = ['거짓 결함'] + [f'Cluster {i}' for i in sorted(df_real_defect_clustered['Cluster'].unique())]

# Define palette (optional, but can help differentiate groups)
palette_viz = sns.color_palette('viridis', df_real_defect_clustered['Cluster'].nunique())
palette_dict_viz = {f'Cluster {i}': palette_viz[i] for i in sorted(df_real_defect_clustered['Cluster'].unique())}
palette_dict_viz['거짓 결함'] = 'lightcoral' # Assign a color for False Defects

plt.figure(figsize=(6, 4))
sns.boxplot(x='Group', y='RELATIVEMAGNITUDE', data=df_combined_viz, order=group_order, palette=palette_dict_viz)
plt.title(f'RELATIVEMAGNITUDE 분포 비교')
plt.xlabel('그룹')
plt.ylabel('RELATIVEMAGNITUDE')
plt.xticks(rotation=0)
plt.ylim(-50, 1200)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Group', y='DEFECT_AREA', data=df_combined_viz, order=group_order, palette=palette_dict_viz)
plt.title(f'DEFECT_AREA 분포 비교')
plt.xlabel('그룹')
plt.ylabel('DEFECT_AREA')
plt.xticks(rotation=0)
plt.ylim(-0.0005,0.029)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Group', y='SPOTLIKENESS', data=df_combined_viz, order=group_order, palette=palette_dict_viz)
plt.title(f'SPOTLIKENESS 분포 비교')
plt.xlabel('그룹')
plt.ylabel('SPOTLIKENESS')
plt.xticks(rotation=0)
plt.ylim(-0.5,14.5)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Group', y='SNR_OFFSET_GL', data=df_combined_viz, order=group_order, palette=palette_dict_viz)
plt.title(f'SNR_OFFSET_GL 분포 비교')
plt.xlabel('그룹')
plt.ylabel('SNR_OFFSET_GL')
plt.xticks(rotation=0)
plt.ylim(-0.02,0.85)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Ensure pandas is imported if not already in the environment

# Assuming class_cluster_proportion is available from cell 060a54fc

if 'class_cluster_proportion' not in locals():
    print("Error: 'class_cluster_proportion' DataFrame not found. Please ensure cell 060a54fc has been executed.")
else:
    print("--- Class별 군집 분포 비율 (누적 막대 그래프) ---")
    # 원하는 색상 지정 (Cluster 0, Cluster 1 순서라고 가정)
    colors = ['#31688E', '#35B779']

    plt.figure(figsize=(6, 4))
    # colormap 대신 color 매개변수 사용
    class_cluster_proportion.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
    plt.title('Class별 군집 분포 비율')
    plt.xlabel('Class')
    plt.ylabel('비율')
    plt.xticks(rotation=0)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Assuming df_real_defect_clustered and df_real_scaled are available

if 'df_real_defect_clustered' not in locals() or 'df_real_scaled' not in locals():
    print("Error: Required DataFrames for evaluation not found.")
else:
    print("--- 군집 성능 평가 ---")

    # Get the cluster labels
    cluster_labels = df_real_defect_clustered['Cluster']

    # Calculate Silhouette Score
    # Note: Silhouette score requires more than one cluster (k > 1) and more than one sample.
    if len(cluster_labels.unique()) > 1 and len(df_real_scaled) > 1:
        try:
            silhouette_avg = silhouette_score(df_real_scaled, cluster_labels)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        except Exception as e:
            print(f"Error calculating Silhouette Score: {e}")
    else:
        print("Cannot calculate Silhouette Score with less than 2 clusters or less than 2 samples.")


    # Calculate Davies-Bouldin Index
    # Note: Davies-Bouldin Index requires more than one cluster (k > 1).
    if len(cluster_labels.unique()) > 1:
         try:
            db_index = davies_bouldin_score(df_real_scaled, cluster_labels)
            print(f"Davies-Bouldin Index: {db_index:.4f}")
         except Exception as e:
            print(f"Error calculating Davies-Bouldin Index: {e}")
    else:
        print("Cannot calculate Davies-Bouldin Index with less than 2 clusters.")


    # Calculate Calinski-Harabasz Index
    # Note: Calinski-Harabasz Index requires more than one cluster (k > 1).
    if len(cluster_labels.unique()) > 1:
        try:
            ch_index = calinski_harabasz_score(df_real_scaled, cluster_labels)
            print(f"Calinski-Harabasz Index: {ch_index:.4f}")
        except Exception as e:
            print(f"Error calculating Calinski-Harabasz Index: {e}")
    else:
        print("Cannot calculate Calinski-Harabasz Index with less than 2 clusters.")


print("\n--- 군집 결과 시각화 (2D 산점도 - PCA) ---")
from sklearn.decomposition import PCA

if 'df_real_scaled' not in locals() or 'df_real_defect_clustered' not in locals():
     print("Error: Required DataFrames for visualization not found.")
else:
    # Perform PCA for dimensionality reduction to 2 components for visualization
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_real_scaled)

    # Create a new DataFrame for the PCA results and include cluster labels
    df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'], index=df_real_scaled.index)
    df_pca['Cluster'] = df_real_defect_clustered['Cluster']

    # Visualize the clusters in a 2D scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='viridis', legend='full')
    plt.title('K-Means 군집 결과')
    plt.xlabel('주성분 1 (PCA1)')
    plt.ylabel('주성분 2 (PCA2)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# df_real_scaled에서 원본 변수(Feature) 이름들을 가져옵니다.
feature_names = df_real_scaled.columns

# pca.components_에 저장된 기여도(Loadings)를 가져옵니다.
pca_loadings = pca.components_

# 보기 쉽게 DataFrame으로 만듭니다.
df_pca_loadings = pd.DataFrame(pca_loadings,
                             columns=feature_names,
                             index=['PCA1', 'PCA2'])

print("--- PCA Component Loadings (주성분 적재값) ---")
display(df_pca_loadings)

plt.figure(figsize=(12, 5))
sns.heatmap(df_pca_loadings, annot=True, cmap='vlag', fmt='.2f')
plt.title('PCA Component Loadings Heatmap')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_real_defect_clustered and df_false_defect are available

if 'df_real_defect_clustered' not in locals() or 'df_false_defect' not in locals():
    print("Error: Required DataFrames for comparison not found.")
else:
    print("--- K-Means 군집 0, 1 및 거짓 결함 특성 비교 ---")

    # Select data for Cluster 0, Cluster 1 (from real defects), and False Defects
    df_cluster_0 = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == 0].copy()
    df_cluster_0['Group'] = 'Cluster 0 (Real Defects)'

    df_cluster_1 = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == 1].copy()
    df_cluster_1['Group'] = 'Cluster 1 (Real Defects)'

    df_false_viz = df_false_defect.copy()
    df_false_viz['Group'] = 'False Defects (Class 9)' # Assuming Class 9 is primarily False Defects based on earlier analysis

    # Combine the dataframes for visualization
    df_comparison = pd.concat([df_cluster_0, df_cluster_1, df_false_viz], ignore_index=True)

    # Define the order of groups for visualization
    group_order = ['False Defects (Class 9)', 'Cluster 0 (Real Defects)', 'Cluster 1 (Real Defects)']

    # Select key numerical features for comparison.
    # Based on previous KDE plots and clustering analysis, features like
    # DEFECT_AREA, SPOTLIKENESS, SNR_OFFSET_GL, INTENSITY, and size-related features might show differences.
    comparison_features = [
        'DEFECT_AREA',
        'SPOTLIKENESS',
        'SNR_OFFSET_GL',
        'INTENSITY',
        'SIZE_X',
        'SIZE_Y',
        'RELATIVEMAGNITUDE',
        'PATCHDEFECTSIGNAL',
        'MDAT_OFFSET',
        'MDAT_GL',
        'PATCHNOISE'

    ]

    # Generate violin plots for selected features
    for feature in comparison_features:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Group', y=feature, data=df_comparison, order=group_order, palette='viridis')
        plt.title(f'{feature} 분포 비교: K-Means 군집 0 vs 1 vs 거짓 결함')
        plt.xlabel('그룹')
        plt.ylabel(feature)
        plt.xticks(rotation=0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    print("\n--- 원본 Class 분포 비교 ---")
    # Compare the distribution of original 'Class' within each group (Clusters 0, 1 and False Defects)
    # Note: False Defects are assumed to be primarily Class 9, but we can verify this.
    # For Clusters 0 and 1, we look at the Class distribution within those clusters.
    # For False Defects, we look at the Class distribution within the df_false_defect DataFrame.

    print("\nK-Means 군집 0의 원본 Class 분포:")
    display(df_cluster_0['Class'].value_counts(normalize=True))

    print("\nK-Means 군집 1의 원본 Class 분포:")
    display(df_cluster_1['Class'].value_counts(normalize=True))

    print("\n거짓 결함의 원본 Class 분포:")
    display(df_false_defect['Class'].value_counts(normalize=True)) # Show distribution of Class within False Defects

    # Visualize original Class distribution as stacked bar plots for Real Clusters vs False Defects
    # Create a combined dataframe with counts for plotting
    class_counts_cluster_0 = df_cluster_0['Class'].value_counts().reset_index()
    class_counts_cluster_0.columns = ['Class', 'Count']
    class_counts_cluster_0['Group'] = 'Cluster 0 (Real Defects)'

    class_counts_cluster_1 = df_cluster_1['Class'].value_counts().reset_index()
    class_counts_cluster_1.columns = ['Class', 'Count']
    class_counts_cluster_1['Group'] = 'Cluster 1 (Real Defects)'

    class_counts_false = df_false_viz['Class'].value_counts().reset_index()
    class_counts_false.columns = ['Class', 'Count']
    class_counts_false['Group'] = 'False Defects (Class 9)'

    df_class_comparison = pd.concat([class_counts_cluster_0, class_counts_cluster_1, class_counts_false], ignore_index=True)

    # Pivot the data for stacked bar plot
    df_class_pivot = df_class_comparison.pivot_table(index='Group', columns='Class', values='Count', fill_value=0)

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    df_class_pivot.loc[group_order].plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.title('원본 Class 분포 비교: K-Means 군집 0 vs 1 vs 거짓 결함')
    plt.xlabel('그룹')
    plt.ylabel('결함 수')
    plt.xticks(rotation=0)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

## 군집별 주요 특성 분포 시각화 (KDE 플롯)

K-Means 군집 0, 1 및 거짓 결함 그룹 간의 주요 수치형 특성 분포를 KDE 플롯으로 시각화합니다.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_real_defect_clustered and df_false_defect are available

if 'df_real_defect_clustered' not in locals() or 'df_false_defect' not in locals():
    print("Error: Required DataFrames for comparison not found.")
else:
    print("--- K-Means 군집 0, 1 및 거짓 결함 특성 분포 비교 (KDE 플롯) ---")

    # Select data for Cluster 0, Cluster 1 (from real defects), and False Defects
    df_cluster_0 = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == 0].copy()
    df_cluster_0['Group'] = 'Cluster 0 (Real Defects)'

    df_cluster_1 = df_real_defect_clustered[df_real_defect_clustered['Cluster'] == 1].copy()
    df_cluster_1['Group'] = 'Cluster 1 (Real Defects)'

    df_false_viz = df_false_defect.copy()
    df_false_viz['Group'] = 'False Defects (Class 9)' # Assuming Class 9 is primarily False Defects

    # Combine the dataframes for visualization
    df_combined_viz = pd.concat([df_cluster_0, df_cluster_1, df_false_viz], ignore_index=True)


    # Select key numerical features for comparison.
    # Using the same features as the violin plots for consistency.
    comparison_features = [
        'DEFECT_AREA',
        'SPOTLIKENESS',
        'SNR_OFFSET_GL',
        'INTENSITY',
        'SIZE_X',
        'SIZE_Y',
        'RELATIVEMAGNITUDE',
        'PATCHDEFECTSIGNAL',
        'MDAT_OFFSET',
        'MDAT_GL',
        'PATCHNOISE'
    ]

    # Generate KDE plots for selected features, separated by group
    for feature in comparison_features:
        plt.figure(figsize=(8, 6))
        # Use hue to plot different groups on the same axes for easier comparison
        sns.kdeplot(data=df_combined_viz, x=feature, hue='Group', common_norm=False, fill=True, alpha=0.5, palette='viridis')
        plt.title(f'{feature} 분포 비교 (KDE): K-Means 군집 0 vs 1 vs 거짓 결함')
        plt.xlabel(feature)
        plt.ylabel('밀도')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='그룹') # Add legend title
        plt.show()

plt.figure(figsize=(8, 6))
# Use hue to plot different groups on the same axes for easier comparison
sns.kdeplot(data=df_combined_viz, x=DEFECT_AREA, hue='Group', common_norm=False, fill=True, alpha=0.5, palette='viridis')
plt.title(f'{feature} 분포 비교 (KDE): K-Means 군집 0 vs 1 vs 거짓 결함')
plt.xlabel(feature)
plt.ylabel('밀도')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='그룹') # Add legend title
plt.show()

# Task
군집 분석을 다른 모델로 수행하고, 결과를 시각화 및 평가해줘.

## 다른 군집화 모델 선택

### Subtask:
K-평균 외에 DBSCAN, 계층적 군집화, GMM(가우시안 혼합 모델) 등 다른 군집화 알고리즘 중 하나를 선택합니다.


**Reasoning**:
Select an appropriate clustering algorithm based on the analysis goal and data characteristics, and add a markdown cell to explain the choice. Considering the various distributions observed in the KDE plots and the PCA visualization, Gaussian Mixture Models (GMM) might be a suitable choice as they can model clusters with different shapes and sizes, which K-Means struggles with.



# Choose Gaussian Mixture Models (GMM) as another clustering algorithm.
# GMM is chosen because it can model clusters with different shapes and sizes,
# which seems appropriate given the diverse distributions observed in the data,
# and it provides probabilistic assignments to clusters.

## GMM을 이용한 군집 분석

### Subtask:
GMM 모델을 적용하기 위해 최적의 컴포넌트(군집) 개수를 결정합니다. AIC(Akaike Information Criterion) 또는 BIC(Bayesian Information Criterion)와 같은 정보 기준을 사용하여 최적의 개수를 탐색합니다.

**Reasoning**:
Determine the optimal number of components (clusters) for the GMM model using AIC and BIC to prepare for applying the GMM algorithm to the scaled real defect data.

from sklearn.mixture import GaussianMixture

# Assuming df_real_scaled is available from previous steps

if 'df_real_scaled' not in locals():
    print("Error: 'df_real_scaled' DataFrame not found. Please ensure the data scaling cell has been executed.")
else:
    # Determine the optimal number of components for GMM using AIC and BIC
    n_components_range = range(2, 11) # Test components from 2 to 10
    aic = []
    bic = []

    print("--- GMM 최적 컴포넌트 개수 탐색 (AIC/BIC) ---")

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        gmm.fit(df_real_scaled)
        aic.append(gmm.aic(df_real_scaled))
        bic.append(gmm.bic(df_real_scaled))

    # Plot AIC and BIC results
    plt.figure(figsize=(10, 5))

    plt.plot(n_components_range, aic, marker='o', label='AIC')
    plt.plot(n_components_range, bic, marker='o', label='BIC')
    plt.title('GMM 최적 컴포넌트 개수 (AIC/BIC)')
    plt.xlabel('컴포넌트 개수')
    plt.ylabel('정보 기준 값')
    plt.xticks(n_components_range)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Find the optimal number of components based on the minimum AIC and BIC
    optimal_n_aic = n_components_range[np.argmin(aic)]
    optimal_n_bic = n_components_range[np.argmin(bic)]

    print(f"\nAIC 기준 최적 컴포넌트 개수: {optimal_n_aic}")
    print(f"BIC 기준 최적 컴포넌트 개수: {optimal_n_bic}")

    # For the next step, we'll use one of these (e.g., the one suggested by BIC as it penalizes complexity more)
    # Let's choose the optimal_n_bic for now, but this can be adjusted based on the plots.
    optimal_n_gmm = optimal_n_bic
    print(f"\n선택된 GMM 컴포넌트 개수 (BIC 기준): {optimal_n_gmm}")

## GMM 모델 적용 및 군집 라벨 할당

### Subtask:
결정된 최적의 컴포넌트 개수로 GMM 모델을 실제 결함 데이터에 적용하고, 각 데이터 포인트에 군집 라벨을 할당합니다.

**Reasoning**:
Apply the GMM model with the optimal number of components determined in the previous step to the scaled real defect data and assign the resulting cluster labels to the original real defect DataFrame.

from sklearn.mixture import GaussianMixture

# Assuming df_real_scaled is available and optimal_n_gmm is determined

if 'df_real_scaled' not in locals():
    print("Error: 'df_real_scaled' DataFrame not found. Please ensure the data scaling cell has been executed.")
elif 'optimal_n_gmm' not in locals():
    print("Error: 'optimal_n_gmm' not found. Please ensure the GMM optimal component determination cell has been executed.")
else:
    print(f"--- GMM 모델 적용 (컴포넌트 개수: {optimal_n_gmm}) ---")

    # Apply GMM with the optimal number of components
    gmm = GaussianMixture(n_components=optimal_n_gmm, random_state=42, n_init=10)
    gmm.fit(df_real_scaled)

    # Predict the cluster labels for each data point
    gmm_cluster_labels = gmm.predict(df_real_scaled)

    # Add the cluster labels to the original df_real_defect DataFrame
    # Ensure the indices match between df_real_defect and gmm_cluster_labels
    # We need to use the original_indices saved during scaling if NaNs were dropped
    if 'original_indices' in locals():
        df_real_defect_gmm_clustered = df_real_defect.loc[original_indices].copy()
        df_real_defect_gmm_clustered['GMM_Cluster'] = gmm_cluster_labels
    else:
        # If no NaNs were dropped during scaling, indices should match directly
        df_real_defect_gmm_clustered = df_real_defect.copy()
        df_real_defect_gmm_clustered['GMM_Cluster'] = gmm_cluster_labels


    print("실제 결함 데이터에 GMM 군집 라벨 추가:")
    display(df_real_defect_gmm_clustered.head())

    # Display the count of data points in each cluster
    print("\nGMM 군집별 데이터 포인트 개수:")
    display(df_real_defect_gmm_clustered['GMM_Cluster'].value_counts())

## 군집 결과 분석

### Subtask:
생성된 군집별로 수치형 특성의 기술 통계 및 원본 'Class' 분포를 분석하여 각 군집의 특성을 파악합니다.

**Reasoning**:
Analyze the characteristics of the identified clusters by calculating descriptive statistics for the numerical features within each cluster and examining the distribution of the original 'Class' labels within each cluster.

# Assuming df_real_defect_gmm_clustered is available from previous steps and numerical_features_for_clustering is defined

if 'df_real_defect_gmm_clustered' not in locals():
    print("Error: 'df_real_defect_gmm_clustered' DataFrame not found. Please ensure the GMM clustering cell has been executed.")
elif 'numerical_features_for_clustering' not in locals():
     print("Error: 'numerical_features_for_clustering' not found. Please ensure the data preparation cell for clustering has been executed.")
else:
    print("--- GMM 군집 결과 분석 ---")

    # Analyze the mean of numerical features within each cluster
    gmm_cluster_means = df_real_defect_gmm_clustered.groupby('GMM_Cluster')[numerical_features_for_clustering].mean()

    print("\nGMM 군집별 수치형 특성 평균:")
    display(gmm_cluster_means)

    # Analyze the distribution of original 'Class' within each cluster
    # Group by GMM_Cluster and Class, count occurrences, and unstack
    gmm_cluster_class_distribution = df_real_defect_gmm_clustered.groupby(['GMM_Cluster', 'Class']).size().unstack(fill_value=0)

    # Calculate the proportion of each Class within each cluster
    gmm_cluster_class_proportion = gmm_cluster_class_distribution.divide(gmm_cluster_class_distribution.sum(axis=1), axis=0)

    print("\nGMM 군집별 원본 Class 분포 비율 (각 군집 내 Class 비율):")
    display(gmm_cluster_class_proportion)

    # Calculate the proportion of each Cluster within each Class
    # Group by Class and GMM_Cluster, count occurrences, and unstack
    gmm_class_cluster_distribution = df_real_defect_gmm_clustered.groupby(['Class', 'GMM_Cluster']).size().unstack(fill_value=0)

    # Calculate the total number of defects for each Class
    total_per_class = df_real_defect_gmm_clustered.groupby('Class').size()

    # Calculate the proportion of each Cluster within each Class
    gmm_class_cluster_proportion = gmm_class_cluster_distribution.divide(total_per_class, axis=0)

    print("\nClass별 GMM 군집 분포 비율 (각 Class가 속하는 군집 비율):")
    display(gmm_class_cluster_proportion)

## 군집 결과 시각화

### Subtask:
군집 분석 결과를 시각화하여 각 군집의 주요 특성 분포 및 원본 'Class' 분포를 제시합니다.

**Reasoning**:
Visualize the clustering results by generating box plots for key numerical features across clusters and a stacked bar plot showing the distribution of original 'Class' labels within each cluster. This provides visual insights into the characteristics of the identified clusters.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA


# Assuming df_real_defect_gmm_clustered, df_false_defect, and df_real_scaled are available

if 'df_real_defect_gmm_clustered' not in locals() or 'df_false_defect' not in locals() or 'df_real_scaled' not in locals():
    print("Error: Required DataFrames for visualization not found.")
else:
    print("--- GMM 군집별 주요 수치형 특성 분포 (박스 플롯) 및 거짓 결함 비교 ---")

    # Combine clustered real defects (GMM) and false defects for comparison visualization
    df_real_clustered_gmm_viz = df_real_defect_gmm_clustered.copy()
    df_real_clustered_gmm_viz['Group'] = 'GMM Cluster ' + df_real_clustered_gmm_viz['GMM_Cluster'].astype(str)

    df_false_viz = df_false_defect.copy()
    df_false_viz['Group'] = '거짓 결함 (Class 9)'

    df_combined_gmm_viz = pd.concat([df_real_clustered_gmm_viz, df_false_viz], ignore_index=True)

    # Define the order of groups for visualization (False Defects first, then GMM clusters)
    group_order_gmm = ['거짓 결함 (Class 9)'] + [f'GMM Cluster {i}' for i in sorted(df_real_defect_gmm_clustered['GMM_Cluster'].unique())]

    # Define palette (optional, but can help differentiate groups)
    # Using a colorblind-friendly palette with enough distinct colors for 9 clusters + 1 for False
    palette_viz_gmm = sns.color_palette('tab10', df_real_defect_gmm_clustered['GMM_Cluster'].nunique()) # Use tab10 for more colors
    palette_dict_viz_gmm = {f'GMM Cluster {i}': palette_viz_gmm[i] for i in sorted(df_real_defect_gmm_clustered['GMM_Cluster'].unique())}
    palette_dict_viz_gmm['거짓 결함 (Class 9)'] = 'lightcoral' # Assign a distinct color for False Defects


    # Select a few key numerical features for box plots based on previous analysis or domain knowledge
    # Using the same features as the previous K-Means visualization for comparison
    key_numerical_features_viz = ['DEFECT_AREA', 'SPOTLIKENESS', 'SNR_OFFSET_GL']

    # Visualize distribution of selected numerical features across GMM clusters and False Defects using box plots
    for col in key_numerical_features_viz:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Group', y=col, data=df_combined_gmm_viz, order=group_order_gmm, palette=palette_dict_viz_gmm)
        plt.title(f'GMM 군집별 {col} 분포 비교')
        plt.xlabel('그룹')
        plt.ylabel(col)
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()

    print("\n--- GMM 군집별 원본 Class 분포 (누적 막대 그래프) ---")
    # Group by GMM_Cluster and Class, count occurrences, and unstack
    gmm_cluster_class_distribution = df_real_defect_gmm_clustered.groupby(['GMM_Cluster', 'Class']).size().unstack(fill_value=0)

    # Calculate the proportion of each Class within each cluster
    gmm_cluster_class_proportion = gmm_cluster_class_distribution.divide(gmm_cluster_class_distribution.sum(axis=1), axis=0)

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    gmm_cluster_class_proportion.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.title('GMM 군집별 원본 Class 분포 비율')
    plt.xlabel('GMM 군집')
    plt.ylabel('비율')
    plt.xticks(rotation=0)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("\n--- GMM 군집 결과 시각화 (2D 산점도 - PCA) ---")
    # Perform PCA for dimensionality reduction to 2 components for visualization
    pca = PCA(n_components=2)
    df_pca_gmm = pca.fit_transform(df_real_scaled)

    # Create a new DataFrame for the PCA results and include GMM cluster labels
    df_pca_gmm = pd.DataFrame(df_pca_gmm, columns=['PCA1', 'PCA2'], index=df_real_scaled.index)
    df_pca_gmm['GMM_Cluster'] = df_real_defect_gmm_clustered['GMM_Cluster']

    # Visualize the GMM clusters in a 2D scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='GMM_Cluster', data=df_pca_gmm, palette='tab10', legend='full')
    plt.title('GMM 군집 결과 (PCA를 이용한 2D 시각화)')
    plt.xlabel('주성분 1 (PCA1)')
    plt.ylabel('주성분 2 (PCA2)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

## GMM 군집 성능 평가

### Subtask:
적절한 군집 평가 지표(예: 실루엣 점수, Davies-Bouldin 지수, Calinski-Harabasz 지수)를 사용하여 GMM 군집 결과의 성능을 정량적으로 평가합니다.

**Reasoning**:
Evaluate the performance of the GMM clustering using the same quantitative metrics as for K-Means clustering to allow for a direct comparison of the two methods.

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Assuming df_real_defect_gmm_clustered and df_real_scaled are available

if 'df_real_defect_gmm_clustered' not in locals() or 'df_real_scaled' not in locals():
    print("Error: Required DataFrames for evaluation not found.")
else:
    print("--- GMM 군집 성능 평가 ---")

    # Get the GMM cluster labels
    gmm_cluster_labels = df_real_defect_gmm_clustered['GMM_Cluster']

    # Calculate Silhouette Score
    if len(gmm_cluster_labels.unique()) > 1 and len(df_real_scaled) > 1:
        try:
            silhouette_avg_gmm = silhouette_score(df_real_scaled, gmm_cluster_labels)
            print(f"GMM Silhouette Score: {silhouette_avg_gmm:.4f}")
        except Exception as e:
            print(f"Error calculating GMM Silhouette Score: {e}")
    else:
        print("Cannot calculate GMM Silhouette Score with less than 2 clusters or less than 2 samples.")

    # Calculate Davies-Bouldin Index
    if len(gmm_cluster_labels.unique()) > 1:
        try:
            db_index_gmm = davies_bouldin_score(df_real_scaled, gmm_cluster_labels)
            print(f"GMM Davies-Bouldin Index: {db_index_gmm:.4f}")
        except Exception as e:
            print(f"Error calculating GMM Davies-Bouldin Index: {e}")
    else:
        print("Cannot calculate GMM Davies-Bouldin Index with less than 2 clusters.")

    # Calculate Calinski-Harabasz Index
    if len(gmm_cluster_labels.unique()) > 1:
        try:
            ch_index_gmm = calinski_harabasz_score(df_real_scaled, gmm_cluster_labels)
            print(f"GMM Calinski-Harabasz Index: {ch_index_gmm:.4f}")
        except Exception as e:
            print(f"Error calculating GMM Calinski-Harabasz Index: {e}")
    else:
        print("Cannot calculate GMM Calinski-Harabasz Index with less than 2 clusters.")

# You can also print the evaluation metrics from the K-Means clustering (if available) for comparison
if 'silhouette_avg' in locals():
    print(f"\nK-Means Silhouette Score: {silhouette_avg:.4f}")
if 'db_index' in locals():
    print(f"K-Means Davies-Bouldin Index: {db_index:.4f}")
if 'ch_index' in locals():
    print(f"K-Means Calinski-Harabasz Index: {ch_index:.4f}")

# Task
Generate process-step-specific K-Means clustering for 'PC', 'RMG', and 'CBCMP' steps in the `df_cleaned` DataFrame. For each step, apply K-Means clustering with k=2 to the numerical features of real defects, assign cluster labels, and analyze the characteristics of each cluster by calculating the mean of features and the distribution of original 'Class' labels. Finally, visualize the clusters using PCA for each process step.

## Process Step Clustering

### Subtask:
Iterate through 'PC', 'RMG', 'CBCMP' process steps. For each step, filter the real defect data, prepare numerical features by handling missing values and scaling.


## Summary:

### Data Analysis Key Findings
- The "Process Step Clustering" subtask has been initiated to analyze defect patterns within specific manufacturing process steps.
- The current process focuses on iterating through 'PC', 'RMG', and 'CBCMP' steps individually.
- For each identified process step, the real defect data is to be filtered.
- Numerical features will be prepared by handling missing values and applying scaling before further analysis.

### Insights or Next Steps
- The next crucial step is to proceed with the K-Means clustering (k=2) on the prepared numerical features for each process step.
- Analyzing the cluster characteristics, including mean feature values and 'Class' label distribution, along with PCA visualizations, will be essential to identify distinct defect profiles and potential root causes.


# Task
Lot 단위의 예측 모델 데이터셋을 구축합니다. 이를 위해 Lot별 통합 위험도 점수를 바탕으로 '고위험 Lot' 여부를 타겟 변수로 정의하고, `df_cleaned`의 결함 데이터를 Lot 단위로 집계하여 예측 피처를 생성합니다.

## 예측 모델 데이터셋 준비

### Subtask:
Lot 단위의 예측 모델 데이터셋을 구축합니다. 이를 위해 Lot별 통합 위험도 점수를 바탕으로 '고위험 Lot' 여부를 타겟 변수로 정의하고, `df_cleaned`의 결함 데이터를 Lot 단위로 집계하여 예측 피처를 생성합니다.


## Summary:

### Data Analysis Key Findings
*   A prediction dataset was constructed at the Lot level.
*   The target variable, '고위험 Lot' (High-risk Lot), was defined based on an integrated risk score for each Lot.
*   Prediction features were generated by aggregating defect data from the `df_cleaned` dataframe at the Lot level.

### Insights or Next Steps
*   The prepared dataset can now be used to train a predictive model to identify high-risk Lots.
*   Further feature engineering or selection could be performed on the aggregated Lot-level data to optimize model performance.


# Task
## 예측 모델 데이터셋 준비 (회귀)

Lot 단위의 예측 모델 데이터셋을 구축합니다. 이를 위해 Lot별 통합 위험도 점수(`Total_Risk_Score`)를 타겟 변수로 정의하고, `df_cleaned`의 결함 데이터를 Lot 단위로 집계하여 예측 피처를 생성합니다.

```python
import pandas as pd
import numpy as np

# df_cleaned와 df_lot는 이전 단계에서 생성되어 있다고 가정합니다.

# --- 수치형 피처 정의 및 집계 함수 설정 ---
numerical_features = [
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SIZE_D', 'INTENSITY', 'POLARITY',
    'ENERGY_PARAM', 'MDAT_OFFSET', 'MDAT_GL', 'MDAT_NOISE', 'RADIUS', 'ANGLE',
    'ALIGNRATIO', 'SPOTLIKENESS', 'PATCHNOISE', 'RELATIVEMAGNITUDE',
    'ACTIVERATIO', 'PATCHDEFECTSIGNAL', 'SNR_OFFSET_GL', 'SNR_INTENSITY_NOISE',
    'ASPECT_RATIO', 'DENSITY_SIGNAL'
]

# 결측치에 강인하도록 중앙값을 추가하여 집계 함수 정의
aggregation_functions = ['mean', 'std', 'min', 'max', 'median']

# --- Lot 단위 수치형 피처 집계 ---
print("--- Lot 단위 피처 집계 시작 (수치형) ---")
# 'Lot Name'으로 그룹화하여 정의된 수치형 피처에 집계 함수 적용
df_aggregated_numerical = df_cleaned.groupby('Lot Name')[numerical_features].agg(aggregation_functions)

# 다중 인덱스 컬럼 이름을 단일 문자열로 변환 (예: 'SIZE_X_mean')
df_aggregated_numerical.columns = ['_'.join(col).strip() for col in df_aggregated_numerical.columns.values]
print(f"  숫자형 피처 집계 완료. 생성된 피처 수: {len(df_aggregated_numerical.columns)}")

# --- Lot 단위 범주형/고유값 피처 집계 ---
print("--- Lot 단위 피처 집계 시작 (범주형/고유값) ---")
# 'Lot Name'으로 그룹화하여 범주형 피처의 고유값 개수 집계
df_aggregated_categorical_unique = df_cleaned.groupby('Lot Name').agg(
    Step_desc_nunique=('Step_desc', 'nunique'),
    Class_nunique=('Class', 'nunique'),
    Slot_No_nunique=('Slot No', 'nunique')
)
print(f"  범주형/고유값 피처 집계 완료. 생성된 피처 수: {len(df_aggregated_categorical_unique.columns)}")

# --- 모든 집계된 피처를 기존 df_lot (타겟 변수 포함)에 병합 ---
# df_lot에 이미 Total_Count, Killer_Defect_Count, False_Defect_Count 등이 포함되어 있습니다.
df_prediction_dataset = df_lot.copy()

# 수치형 피처 병합
df_prediction_dataset = pd.merge(df_prediction_dataset, df_aggregated_numerical, on='Lot Name', how='left')

# 범주형/고유값 피처 병합
df_prediction_dataset = pd.merge(df_prediction_dataset, df_aggregated_categorical_unique, on='Lot Name', how='left')

# --- 최종 데이터셋 확인 및 결측치 처리 ---
print("\n--- 예측 모델 데이터셋 준비 완료 ---")
print(f"최종 데이터셋 형태: {df_prediction_dataset.shape}")

print("\n데이터셋 미리보기:")
display(df_prediction_dataset.head())

print("\n결측치 확인:")
# 결측치가 있는 컬럼 및 개수 식별
nan_columns = df_prediction_dataset.isnull().sum()
nan_columns = nan_columns[nan_columns > 0]
if not nan_columns.empty:
    display(nan_columns)
    # 수치형 집계 피처의 결측치는 해당 컬럼의 평균으로 대체
    # 고유값 개수의 결측치는 0으로 대체 (해당 Lot에 데이터가 없어 고유값 개수를 계산할 수 없는 경우)
    print("\n결측치 처리 (수치형: 평균, 고유값: 0으로 대체) ---")
    for col in nan_columns.index:
        if col in df_aggregated_numerical.columns:
            df_prediction_dataset[col].fillna(df_prediction_dataset[col].mean(), inplace=True)
        elif col in df_aggregated_categorical_unique.columns:
            df_prediction_dataset[col].fillna(0, inplace=True)
    print("  결측치 처리 완료.")
    print("\n결측치 재확인:")
    # 처리 후 결측치가 남아있는지 재확인
    display(df_prediction_dataset.isnull().sum()[df_prediction_dataset.isnull().sum() > 0])
else:
    print("  결측치가 없습니다.")


# --- 특징 변수(X)와 타겟 변수(y) 정의 ---
target_variable = 'Total_Risk_Score'
# 'Lot Name'과 타겟 변수를 제외한 모든 컬럼을 특징 변수로 사용
features_for_prediction = [col for col in df_prediction_dataset.columns if col not in ['Lot Name', target_variable]]

X_reg = df_prediction_dataset[features_for_prediction]
y_reg = df_prediction_dataset[target_variable]

print(f"\n특징 변수(X_reg) 수: {X_reg.shape[1]}")
print(f"타겟 변수(y_reg) 이름: {target_variable}")
print("\n특징 변수 데이터 미리보기:")
display(X_reg.head())
print("\n타겟 변수 데이터 미리보기:")
display(y_reg.head())
```

## 회귀 모델 훈련 (RandomForestRegressor)

데이터를 훈련 세트와 테스트 세트로 분할하고, RandomForestRegressor 모델을 사용하여 Lot별 `Total_Risk_Score`를 예측하도록 훈련합니다.

import pandas as pd
import numpy as np

# df_cleaned와 df_lot는 이전 단계에서 생성되어 있다고 가정합니다.

# --- 수치형 피처 정의 및 집계 함수 설정 ---
numerical_features = [
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SIZE_D', 'INTENSITY', 'POLARITY',
    'ENERGY_PARAM', 'MDAT_OFFSET', 'MDAT_GL', 'MDAT_NOISE', 'RADIUS', 'ANGLE',
    'ALIGNRATIO', 'SPOTLIKENESS', 'PATCHNOISE', 'RELATIVEMAGNITUDE',
    'ACTIVERATIO', 'PATCHDEFECTSIGNAL', 'SNR_OFFSET_GL', 'SNR_INTENSITY_NOISE',
    'ASPECT_RATIO', 'DENSITY_SIGNAL'
]

# 결측치에 강인하도록 중앙값을 추가하여 집계 함수 정의
aggregation_functions = ['mean', 'std', 'min', 'max', 'median']

# --- Lot 단위 수치형 피처 집계 ---
print("--- Lot 단위 피처 집계 시작 (수치형) ---")
# 'Lot Name'으로 그룹화하여 정의된 수치형 피처에 집계 함수 적용
df_aggregated_numerical = df_cleaned.groupby('Lot Name')[numerical_features].agg(aggregation_functions)

# 다중 인덱스 컬럼 이름을 단일 문자열로 변환 (예: 'SIZE_X_mean')
df_aggregated_numerical.columns = ['_'.join(col).strip() for col in df_aggregated_numerical.columns.values]
print(f"  숫자형 피처 집계 완료. 생성된 피처 수: {len(df_aggregated_numerical.columns)}")

# --- Lot 단위 범주형/고유값 피처 집계 ---
print("--- Lot 단위 피처 집계 시작 (범주형/고유값) ---")
# 'Lot Name'으로 그룹화하여 범주형 피처의 고유값 개수 집계
df_aggregated_categorical_unique = df_cleaned.groupby('Lot Name').agg(
    Step_desc_nunique=('Step_desc', 'nunique'),
    Class_nunique=('Class', 'nunique'),
    Slot_No_nunique=('Slot No', 'nunique')
)
print(f"  범주형/고유값 피처 집계 완료. 생성된 피처 수: {len(df_aggregated_categorical_unique.columns)}")

# --- 모든 집계된 피처를 기존 df_lot (타겟 변수 포함)에 병합 ---
# df_lot에 이미 Total_Count, Killer_Defect_Count, False_Defect_Count 등이 포함되어 있습니다.
df_prediction_dataset = df_lot.copy()

# 수치형 피처 병합
df_prediction_dataset = pd.merge(df_prediction_dataset, df_aggregated_numerical, on='Lot Name', how='left')

# 범주형/고유값 피처 병합
df_prediction_dataset = pd.merge(df_prediction_dataset, df_aggregated_categorical_unique, on='Lot Name', how='left')

# --- 최종 데이터셋 확인 및 결측치 처리 ---
print("\n--- 예측 모델 데이터셋 준비 완료 ---")
print(f"최종 데이터셋 형태: {df_prediction_dataset.shape}")

print("\n데이터셋 미리보기:")
display(df_prediction_dataset.head())

print("\n결측치 확인:")
# 결측치가 있는 컬럼 및 개수 식별
nan_columns = df_prediction_dataset.isnull().sum()
nan_columns = nan_columns[nan_columns > 0]
if not nan_columns.empty:
    display(nan_columns)
    # 수치형 집계 피처의 결측치는 해당 컬럼의 평균으로 대체
    # 고유값 개수의 결측치는 0으로 대체 (해당 Lot에 데이터가 없어 고유값 개수를 계산할 수 없는 경우)
    print("\n결측치 처리 (수치형: 평균, 고유값: 0으로 대체) ---")
    for col in nan_columns.index:
        if col in df_aggregated_numerical.columns:
            df_prediction_dataset[col].fillna(df_prediction_dataset[col].mean(), inplace=True)
        elif col in df_aggregated_categorical_unique.columns:
            df_prediction_dataset[col].fillna(0, inplace=True)
    print("  결측치 처리 완료.")
    print("\n결측치 재확인:")
    # 처리 후 결측치가 남아있는지 재확인
    display(df_prediction_dataset.isnull().sum()[df_prediction_dataset.isnull().sum() > 0])
else:
    print("  결측치ㅁㅁㅁㅁ가 없습니다.")


# --- 특징 변수(X)와 타겟 변수(y) 정의 ---
target_variable = 'Total_Risk_Score'
# 'Lot Name'과 타겟 변수를 제외한 모든 컬럼을 특징 변수로 사용
features_for_prediction = [col for col in df_prediction_dataset.columns if col not in ['Lot Name', target_variable]]

X_reg = df_prediction_dataset[features_for_prediction]
y_reg = df_prediction_dataset[target_variable]

print(f"\n특징 변수(X_reg) 수: {X_reg.shape[1]}")
print(f"타겟 변수(y_reg) 이름: {target_variable}")
print("\n특징 변수 데이터 미리보기:")
display(X_reg.head())
print("\n타겟 변수 데이터 미리보기:")
display(y_reg.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming X_reg and y_reg are available from the previous step

if 'X_reg' not in locals() or 'y_reg' not in locals():
    print("Error: Regression features (X_reg) or target (y_reg) not found. Please ensure the data preparation step has been executed.")
else:
    print("--- RandomForestRegressor 모델 훈련 시작 ---")

    # 1. 데이터 분할 (훈련 세트와 테스트 세트)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"훈련 세트 크기: {X_train_reg.shape[0]}")
    print(f"테스트 세트 크기: {X_test_reg.shape[0]}")

    # 2. RandomForestRegressor 모델 초기화 및 훈련
    rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1로 병렬 처리
    rf_reg_model.fit(X_train_reg, y_train_reg)

    print("\nRandomForestRegressor 모델 훈련 완료.")

    # 3. 모델 예측
    y_pred_reg = rf_reg_model.predict(X_test_reg)

    # 4. 모델 평가
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"\n모델 평가 결과:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R2 Score): {r2:.4f}")

    # 5. 예측값과 실제값 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2) # y=x 선
    plt.title('RandomForestRegressor: 실제값 vs 예측값')
    plt.xlabel('실제 Total_Risk_Score')
    plt.ylabel('예측 Total_Risk_Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 6. 피처 중요도 시각화
    if hasattr(rf_reg_model, 'feature_importances_'):
        feature_importances = pd.Series(rf_reg_model.feature_importances_, index=X_reg.columns)
        top_n_features = 20 # 상위 20개 피처 시각화
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances.nlargest(top_n_features).values,
                    y=feature_importances.nlargest(top_n_features).index,
                    palette='viridis')
        plt.title(f'RandomForestRegressor: 상위 {top_n_features}개 피처 중요도')
        plt.xlabel('중요도')
        plt.ylabel('피처')
        plt.tight_layout()
        plt.show()
    print("\nRandomForestRegressor 모델 훈련 및 평가 완료.")

## 예측 모델 데이터셋 준비 (회귀)

### Subtask:
Lot 단위의 예측 모델 데이터셋을 구축합니다. 이를 위해 Lot별 통합 위험도 점수(`Total_Risk_Score`)를 타겟 변수로 정의하고, `df_cleaned`의 결함 데이터를 Lot 단위로 집계하여 예측 피처를 생성합니다.


## Summary:

### Data Analysis Key Findings

*   **Feature Generation:** A total of 110 numerical features were generated by applying `mean`, `std`, `min`, `max`, and `median` aggregation functions to 22 base numerical features (`SIZE_X`, `SIZE_Y`, `DEFECT_AREA`, etc.) grouped by 'Lot Name'.
*   **Categorical/Unique Feature Generation:** Three additional features were created by counting the number of unique values for 'Step\_desc', 'Class', and 'Slot No' per 'Lot Name'.
*   **Final Dataset Shape:** The final prediction dataset, `df_prediction_dataset`, has a shape of (number of lots, original df\_lot columns + 110 numerical + 3 categorical unique features). The exact row count depends on the number of unique 'Lot Name' entries.
*   **Missing Value Handling:** Missing values were identified in the aggregated features. Numerical aggregated features' missing values were imputed using the mean of their respective columns, while missing values in the unique count features were filled with 0. After this process, no missing values remained in the prepared dataset.
*   **Target and Feature Variables:** The target variable `y_reg` was defined as `Total_Risk_Score`. The feature set `X_reg` consists of all columns in `df_prediction_dataset` except 'Lot Name' and the `Total_Risk_Score`, resulting in 113 features (110 numerical + 3 categorical unique).

### Insights or Next Steps

*   The prepared dataset is now suitable for training a regression model to predict `Total_Risk_Score` based on Lot-level aggregated defect characteristics.
*   Further feature engineering, such as creating interaction terms or polynomial features, could be explored to potentially enhance model performance. Additionally, feature selection techniques can be applied to reduce dimensionality and improve model interpretability.


# Task
새로운 Lot 단위 피처들을 활용하여 회귀 모델을 훈련하고 평가하기 위해, 먼저 `df_cleaned` 데이터프레임에 `KMeans_Cluster` 및 `is_killer_defect` 컬럼을 정확히 할당한 후, 이를 기반으로 Lot별 상세 피처들을 생성합니다.

새로운 Lot 단위 피처 엔지니어링에는 다음이 포함됩니다:
*   각 Lot의 `Step_desc`, `Class`, `KMeans_Cluster` 조합별 결함 수 및 비율 피처 생성.
*   각 Lot의 (킬러/비킬러) 결함 그룹에 대한 주요 수치형 특성(SIZE_X, DEFECT_AREA, SNR_OFFSET_GL 등)의 추가 통계량(평균, 중앙값) 피처 생성.

이후, 기존 Lot 단위 피처와 새로 생성된 피처를 결합하여 확장된 예측 모델 데이터셋(`df_prediction_dataset`)을 재구축하고 결측치를 처리합니다. 마지막으로, 확장된 데이터셋으로 RandomForestRegressor 모델을 재훈련하고 성능을 평가합니다.

## 새로운 Lot 단위 피처 엔지니어링

### Subtask:
`df_cleaned` 데이터프레임에 `KMeans_Cluster` 및 `is_killer_defect` 컬럼을 할당하고, 이를 기반으로 Lot별 상세 피처들을 생성합니다.


**Reasoning**:
First, I need to check if `df_cleaned` and `df_real_defect_clustered` are available in the current environment to ensure the necessary data is loaded for the subsequent steps. Then, I will initialize the `KMeans_Cluster` column in `df_cleaned` and map the cluster labels from `df_real_defect_clustered`.



import pandas as pd

# Ensure df_cleaned and df_real_defect_clustered are available
if 'df_cleaned' not in locals() or 'df_real_defect_clustered' not in locals():
    print("Error: 'df_cleaned' or 'df_real_defect_clustered' not found. Please ensure previous steps are executed.")
else:
    print("--- df_cleaned에 KMeans_Cluster 컬럼 할당 시작 ---")
    # 1. df_cleaned에 새로운 nullable integer 타입의 KMeans_Cluster 컬럼을 생성하고 pd.NA로 초기화합니다.
    df_cleaned['KMeans_Cluster'] = pd.NA

    # df_real_defect_clustered의 'Cluster' 값을 해당하는 인덱스에 매핑하여 업데이트합니다.
    df_cleaned.loc[df_real_defect_clustered.index, 'KMeans_Cluster'] = df_real_defect_clustered['Cluster']
    df_cleaned['KMeans_Cluster'] = df_cleaned['KMeans_Cluster'].astype('Int64') # Ensure nullable integer type

    print("  'KMeans_Cluster' 컬럼 할당 완료.")
    display(df_cleaned.head())

**Reasoning**:
The previous step successfully added the `KMeans_Cluster` column. Now, I will proceed with creating the `is_killer_defect` column, defining numerical features for statistics, and then generating various Lot-level aggregated features including counts, proportions, means, and medians, finally merging them into `df_new_lot_features`.



import pandas as pd
import numpy as np

# Ensure df_cleaned and df_real_defect_clustered are available (already checked in previous cell)

# 2. df_cleaned 데이터프레임에 새로운 boolean 타입의 is_killer_defect 컬럼을 생성하고 False로 초기화합니다.
df_cleaned['is_killer_defect'] = False

# 3. killer_cluster_mapping 딕셔너리를 사용하여 is_killer_defect 값을 설정합니다.
killer_cluster_mapping = {
    'PC': 1,  # PC 공정의 킬러 결함 군집 ID
    'RMG': 1, # RMG 공정의 킬러 결함 군집 ID
    'CBCMP': 0 # CBCMP 공정의 킬러 결함 군집 ID
}

for step_desc, killer_cluster_id in killer_cluster_mapping.items():
    # 해당 공정의 실제 결함 중, 킬러 군집에 속하는 결함을 True로 설정
    mask = (df_cleaned['IS_DEFECT'] == 'REAL') & \
           (df_cleaned['Step_desc'] == step_desc) & \
           (df_cleaned['KMeans_Cluster'] == killer_cluster_id)
    df_cleaned.loc[mask, 'is_killer_defect'] = True

print("  'is_killer_defect' 컬럼 할당 완료.")

# 4. 통계량을 계산할 주요 수치형 피처 리스트를 정의합니다.
key_numerical_features_for_stats = [
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SNR_OFFSET_GL', 'INTENSITY',
    'SPOTLIKENESS', 'RELATIVEMAGNITUDE', 'PATCHDEFECTSIGNAL', 'MDAT_OFFSET', 'MDAT_GL', 'PATCHNOISE'
]

# --- Lot별 상세 피처 생성 시작 ---
print("--- Lot별 상세 피처 생성 시작 ---")

# 5. df_cleaned를 Lot Name, Step_desc, Class, KMeans_Cluster 기준으로 그룹화하여 각 조합별 결함의 개수(_count) 계산
df_grouped_counts = df_cleaned.groupby(['Lot Name', 'Step_desc', 'Class', 'KMeans_Cluster']).size().unstack(fill_value=0)
df_grouped_counts.columns = [f'Count_Step_{s}_Class_{c}_Cluster_{k}' for s, c, k in df_grouped_counts.columns]

print(f"  Step/Class/Cluster별 결함 수 피처 생성 완료. ({df_grouped_counts.shape[1]}개)")

# 6. df_cleaned를 Lot Name, Step_desc, Class, KMeans_Cluster 기준으로 그룹화하여 각 조합별 결함의 비율(_prop) 계산
# 먼저 Lot별 전체 결함 수를 계산
total_defects_per_lot = df_cleaned.groupby('Lot Name').size()

df_grouped_props = df_cleaned.groupby(['Lot Name', 'Step_desc', 'Class', 'KMeans_Cluster']).size().unstack(fill_value=0)
df_grouped_props = df_grouped_props.divide(total_defects_per_lot, axis=0)
df_grouped_props.columns = [f'Prop_Step_{s}_Class_{c}_Cluster_{k}' for s, c, k in df_grouped_props.columns]

print(f"  Step/Class/Cluster별 결함 비율 피처 생성 완료. ({df_grouped_props.shape[1]}개)")

# 7. Lot Name과 is_killer_defect 기준으로 key_numerical_features_for_stats의 평균 계산
df_killer_means = df_cleaned.groupby(['Lot Name', 'is_killer_defect'])[key_numerical_features_for_stats].mean().unstack(fill_value=0)
df_killer_means.columns = [f'{col}_killer_mean' if is_killer else f'{col}_non_killer_mean' for col, is_killer in df_killer_means.columns]

print(f"  킬러/비킬러 결함별 수치형 피처 평균 생성 완료. ({df_killer_means.shape[1]}개)")

# 8. Lot Name과 is_killer_defect 기준으로 key_numerical_features_for_stats의 중앙값 계산
df_killer_medians = df_cleaned.groupby(['Lot Name', 'is_killer_defect'])[key_numerical_features_for_stats].median().unstack(fill_value=0)
df_killer_medians.columns = [f'{col}_killer_median' if is_killer else f'{col}_non_killer_median' for col, is_killer in df_killer_medians.columns]

print(f"  킬러/비킬러 결함별 수치형 피처 중앙값 생성 완료. ({df_killer_medians.shape[1]}개)")

# 9. 생성된 모든 Lot 단위 피처 DataFrames를 병합
df_new_lot_features = df_grouped_counts.join(df_grouped_props, how='left')
df_new_lot_features = df_new_lot_features.join(df_killer_means, how='left')
df_new_lot_features = df_new_lot_features.join(df_killer_medians, how='left')

print("--- Lot별 상세 피처 생성 완료 ---")

# 10. df_new_lot_features의 head()와 shape를 출력
print(f"\n최종 Lot 단위 피처 데이터셋 형태: {df_new_lot_features.shape}")
print("데이터셋 미리보기:")
display(df_new_lot_features.head())

**Reasoning**:
The error occurs because `df_grouped_counts.columns` is a MultiIndex after `unstack()`, and directly iterating `for s, c, k in df_grouped_counts.columns` attempts to unpack each single-level value. I need to iterate through the MultiIndex levels correctly to form the new column names.



import pandas as pd
import numpy as np

# Ensure df_cleaned and df_real_defect_clustered are available (already checked in previous cell)

# 2. df_cleaned 데이터프레임에 새로운 boolean 타입의 is_killer_defect 컬럼을 생성하고 False로 초기화합니다.
df_cleaned['is_killer_defect'] = False

# 3. killer_cluster_mapping 딕셔너리를 사용하여 is_killer_defect 값을 설정합니다.
killer_cluster_mapping = {
    'PC': 1,  # PC 공정의 킬러 결함 군집 ID
    'RMG': 1, # RMG 공정의 킬러 결함 군집 ID
    'CBCMP': 0 # CBCMP 공정의 킬러 결함 군집 ID
}

for step_desc, killer_cluster_id in killer_cluster_mapping.items():
    # 해당 공정의 실제 결함 중, 킬러 군집에 속하는 결함을 True로 설정
    mask = (df_cleaned['IS_DEFECT'] == 'REAL') & \
           (df_cleaned['Step_desc'] == step_desc) & \
           (df_cleaned['KMeans_Cluster'] == killer_cluster_id)
    df_cleaned.loc[mask, 'is_killer_defect'] = True

print("  'is_killer_defect' 컬럼 할당 완료.")

# 4. 통계량을 계산할 주요 수치형 피처 리스트를 정의합니다.
key_numerical_features_for_stats = [
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SNR_OFFSET_GL', 'INTENSITY',
    'SPOTLIKENESS', 'RELATIVEMAGNITUDE', 'PATCHDEFECTSIGNAL', 'MDAT_OFFSET', 'MDAT_GL', 'PATCHNOISE'
]

# --- Lot별 상세 피처 생성 시작 ---
print("--- Lot별 상세 피처 생성 시작 ---")

# 5. df_cleaned를 Lot Name, Step_desc, Class, KMeans_Cluster 기준으로 그룹화하여 각 조합별 결함의 개수(_count) 계산
df_grouped_counts = df_cleaned.groupby(['Lot Name', 'Step_desc', 'Class', 'KMeans_Cluster']).size().unstack(fill_value=0)
# The columns are a MultiIndex, so iterate through them correctly
df_grouped_counts.columns = [f'Count_Step_{col[0]}_Class_{col[1]}_Cluster_{col[2]}' for col in df_grouped_counts.columns]

print(f"  Step/Class/Cluster별 결함 수 피처 생성 완료. ({df_grouped_counts.shape[1]}개)")

# 6. df_cleaned를 Lot Name, Step_desc, Class, KMeans_Cluster 기준으로 그룹화하여 각 조합별 결함의 비율(_prop) 계산
# 먼저 Lot별 전체 결함 수를 계산
total_defects_per_lot = df_cleaned.groupby('Lot Name').size()

df_grouped_props = df_cleaned.groupby(['Lot Name', 'Step_desc', 'Class', 'KMeans_Cluster']).size().unstack(fill_value=0)
df_grouped_props = df_grouped_props.divide(total_defects_per_lot, axis=0)
df_grouped_props.columns = [f'Prop_Step_{col[0]}_Class_{col[1]}_Cluster_{col[2]}' for col in df_grouped_props.columns]

print(f"  Step/Class/Cluster별 결함 비율 피처 생성 완료. ({df_grouped_props.shape[1]}개)")

# 7. Lot Name과 is_killer_defect 기준으로 key_numerical_features_for_stats의 평균 계산
df_killer_means = df_cleaned.groupby(['Lot Name', 'is_killer_defect'])[key_numerical_features_for_stats].mean().unstack(fill_value=0)
df_killer_means.columns = [f'{col[0]}_killer_mean' if col[1] else f'{col[0]}_non_killer_mean' for col in df_killer_means.columns]

print(f"  킬러/비킬러 결함별 수치형 피처 평균 생성 완료. ({df_killer_means.shape[1]}개)")

# 8. Lot Name과 is_killer_defect 기준으로 key_numerical_features_for_stats의 중앙값 계산
df_killer_medians = df_cleaned.groupby(['Lot Name', 'is_killer_defect'])[key_numerical_features_for_stats].median().unstack(fill_value=0)
df_killer_medians.columns = [f'{col[0]}_killer_median' if col[1] else f'{col[0]}_non_killer_median' for col in df_killer_medians.columns]

print(f"  킬러/비킬러 결함별 수치형 피처 중앙값 생성 완료. ({df_killer_medians.shape[1]}개)")

# 9. 생성된 모든 Lot 단위 피처 DataFrames를 병합
df_new_lot_features = df_grouped_counts.join(df_grouped_props, how='left')
df_new_lot_features = df_new_lot_features.join(df_killer_means, how='left')
df_new_lot_features = df_new_lot_features.join(df_killer_medians, how='left')

print("--- Lot별 상세 피처 생성 완료 ---")

# 10. df_new_lot_features의 head()와 shape를 출력
print(f"\n최종 Lot 단위 피처 데이터셋 형태: {df_new_lot_features.shape}")
print("데이터셋 미리보기:")
display(df_new_lot_features.head())


**Reasoning**:
The `IndexError` occurred because `df_grouped_counts.columns` was a simple Index after unstacking, not a MultiIndex as expected by the column renaming loop. To fix this, I need to explicitly unstack the `Step_desc`, `Class`, and `KMeans_Cluster` levels to create a MultiIndex for the columns. This ensures that when iterating over `df_grouped_counts.columns`, each `col` is a tuple representing the `(Step_desc, Class, KMeans_Cluster)` combination, allowing for correct unpacking and column renaming. The same logic will be applied to `df_grouped_props`.



import pandas as pd
import numpy as np

# Ensure df_cleaned and df_real_defect_clustered are available (already checked in previous cell)

# 2. df_cleaned 데이터프레임에 새로운 boolean 타입의 is_killer_defect 컬럼을 생성하고 False로 초기화합니다.
df_cleaned['is_killer_defect'] = False

# 3. killer_cluster_mapping 딕셔너리를 사용하여 is_killer_defect 값을 설정합니다.
killer_cluster_mapping = {
    'PC': 1,  # PC 공정의 킬러 결함 군집 ID
    'RMG': 1, # RMG 공정의 킬러 결함 군집 ID
    'CBCMP': 0 # CBCMP 공정의 킬러 결함 군집 ID
}

for step_desc, killer_cluster_id in killer_cluster_mapping.items():
    # 해당 공정의 실제 결함 중, 킬러 군집에 속하는 결함을 True로 설정
    mask = (df_cleaned['IS_DEFECT'] == 'REAL') & \
           (df_cleaned['Step_desc'] == step_desc) & \
           (df_cleaned['KMeans_Cluster'] == killer_cluster_id)
    df_cleaned.loc[mask, 'is_killer_defect'] = True

print("  'is_killer_defect' 컬럼 할당 완료.")

# 4. 통계량을 계산할 주요 수치형 피처 리스트를 정의합니다.
key_numerical_features_for_stats = [
    'SIZE_X', 'SIZE_Y', 'DEFECT_AREA', 'SNR_OFFSET_GL', 'INTENSITY',
    'SPOTLIKENESS', 'RELATIVEMAGNITUDE', 'PATCHDEFECTSIGNAL', 'MDAT_OFFSET', 'MDAT_GL', 'PATCHNOISE'
]

# --- Lot별 상세 피처 생성 시작 ---
print("--- Lot별 상세 피처 생성 시작 ---")

# 5. df_cleaned를 Lot Name, Step_desc, Class, KMeans_Cluster 기준으로 그룹화하여 각 조합별 결함의 개수(_count) 계산
df_grouped_counts = df_cleaned.groupby(['Lot Name', 'Step_desc', 'Class', 'KMeans_Cluster']).size().unstack(level=['Step_desc', 'Class', 'KMeans_Cluster'], fill_value=0)
df_grouped_counts.columns = [f'Count_Step_{col[0]}_Class_{col[1]}_Cluster_{col[2]}' for col in df_grouped_counts.columns]

print(f"  Step/Class/Cluster별 결함 수 피처 생성 완료. ({df_grouped_counts.shape[1]}개)")

# 6. df_cleaned를 Lot Name, Step_desc, Class, KMeans_Cluster 기준으로 그룹화하여 각 조합별 결함의 비율(_prop) 계산
# 먼저 Lot별 전체 결함 수를 계산
total_defects_per_lot = df_cleaned.groupby('Lot Name').size()

df_grouped_props = df_cleaned.groupby(['Lot Name', 'Step_desc', 'Class', 'KMeans_Cluster']).size().unstack(level=['Step_desc', 'Class', 'KMeans_Cluster'], fill_value=0)
df_grouped_props = df_grouped_props.divide(total_defects_per_lot, axis=0)
df_grouped_props.columns = [f'Prop_Step_{col[0]}_Class_{col[1]}_Cluster_{col[2]}' for col in df_grouped_props.columns]

print(f"  Step/Class/Cluster별 결함 비율 피처 생성 완료. ({df_grouped_props.shape[1]}개)")

# 7. Lot Name과 is_killer_defect 기준으로 key_numerical_features_for_stats의 평균 계산
df_killer_means = df_cleaned.groupby(['Lot Name', 'is_killer_defect'])[key_numerical_features_for_stats].mean().unstack(fill_value=0)
df_killer_means.columns = [f'{col[0]}_killer_mean' if col[1] else f'{col[0]}_non_killer_mean' for col in df_killer_means.columns]

print(f"  킬러/비킬러 결함별 수치형 피처 평균 생성 완료. ({df_killer_means.shape[1]}개)")

# 8. Lot Name과 is_killer_defect 기준으로 key_numerical_features_for_stats의 중앙값 계산
df_killer_medians = df_cleaned.groupby(['Lot Name', 'is_killer_defect'])[key_numerical_features_for_stats].median().unstack(fill_value=0)
df_killer_medians.columns = [f'{col[0]}_killer_median' if col[1] else f'{col[0]}_non_killer_median' for col in df_killer_medians.columns]

print(f"  킬러/비킬러 결함별 수치형 피처 중앙값 생성 완료. ({df_killer_medians.shape[1]}개)")

# 9. 생성된 모든 Lot 단위 피처 DataFrames를 병합
df_new_lot_features = df_grouped_counts.join(df_grouped_props, how='left')
df_new_lot_features = df_new_lot_features.join(df_killer_means, how='left')
df_new_lot_features = df_new_lot_features.join(df_killer_medians, how='left')

print("--- Lot별 상세 피처 생성 완료 ---")

# 10. df_new_lot_features의 head()와 shape를 출력
print(f"\n최종 Lot 단위 피처 데이터셋 형태: {df_new_lot_features.shape}")
print("데이터셋 미리보기:")
display(df_new_lot_features.head())


**Reasoning**:
Now that the detailed Lot-level features have been generated in `df_new_lot_features`, the next step is to merge these with the existing `df_lot` DataFrame, which contains the target variable (`Total_Risk_Score`). After merging, any new missing values introduced during the aggregation will be handled, and then the final `X_reg` and `y_reg` for the regression model will be defined.



import pandas as pd

# Assuming df_lot and df_new_lot_features are available from previous steps

if 'df_lot' not in locals() or 'df_new_lot_features' not in locals():
    print("Error: 'df_lot' or 'df_new_lot_features' not found. Please ensure previous steps are executed.")
elif 'df_prediction_dataset' not in locals(): # Check if df_prediction_dataset exists to avoid overwriting early
    # 1. 기존 df_lot와 새로 생성된 df_new_lot_features 병합
    print("--- 기존 df_lot와 새 Lot 단위 피처 병합 시작 ---")
    df_prediction_dataset = pd.merge(df_lot, df_new_lot_features, on='Lot Name', how='left')
    print(f"  병합 완료. 최종 데이터셋 형태: {df_prediction_dataset.shape}")

    # 2. 결측치 처리 (새로 생성된 피처에서 발생할 수 있음)
    print("--- 병합 후 결측치 확인 및 처리 시작 ---")
    nan_columns_merged = df_prediction_dataset.isnull().sum()
    nan_columns_merged = nan_columns_merged[nan_columns_merged > 0]

    if not nan_columns_merged.empty:
        print("  병합 후 결측치가 있는 컬럼:")
        display(nan_columns_merged)
        # 새로 생성된 피처들의 결측치는 0으로 대체 (해당 Lot에 특정 조합의 결함이 없거나, 킬러/비킬러 결함이 없었을 경우)
        # 만약 평균이나 중앙값으로 대체하는 것이 더 적절하다고 판단되면 변경 가능
        df_prediction_dataset.fillna(0, inplace=True)
        print("  결측치 0으로 대체 완료.")
    else:
        print("  병합 후 결측치가 없습니다.")

    # 3. 특징 변수(X_reg)와 타겟 변수(y_reg) 정의
    print("--- 예측 모델을 위한 X_reg, y_reg 정의 시작 ---")
    target_variable = 'Total_Risk_Score'
    features_for_prediction = [col for col in df_prediction_dataset.columns if col not in ['Lot Name', target_variable, 'Nuisance_Count', 'Score_Killer', 'Score_Nuisance', 'Score_False', 'Killer_Defect_Proportion']]

    X_reg = df_prediction_dataset[features_for_prediction]
    y_reg = df_prediction_dataset[target_variable]

    print(f"  특징 변수(X_reg) 수: {X_reg.shape[1]}")
    print(f"  타겟 변수(y_reg) 이름: {target_variable}")
    print("  예측 모델 데이터셋 준비 완료.")

    print("\n특징 변수 데이터 미리보기:")
    display(X_reg.head())
    print("\n타겟 변수 데이터 미리보기:")
    display(y_reg.head())
else:
    print("df_prediction_dataset already exists. Skipping recreation.")

**Reasoning**:
The previous step prepared the expanded dataset `X_reg` and `y_reg` for the regression model. Now, I need to train a RandomForestRegressor model using this dataset, evaluate its performance with relevant metrics and visualizations, and display the feature importances.



from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming X_reg and y_reg are available from the previous step

if 'X_reg' not in locals() or 'y_reg' not in locals():
    print("Error: Regression features (X_reg) or target (y_reg) not found. Please ensure the data preparation step has been executed.")
else:
    print("--- RandomForestRegressor 모델 훈련 및 과적합 검증 시작 ---")

    # 1. 데이터 분할 (훈련 세트와 테스트 세트)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"훈련 세트 크기: {X_train_reg.shape[0]}")
    print(f"테스트 세트 크기: {X_test_reg.shape[0]}")

    # 2. RandomForestRegressor 모델 초기화 및 훈련
    rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1로 병렬 처리
    rf_reg_model.fit(X_train_reg, y_train_reg)

    print("\nRandomForestRegressor 모델 훈련 완료.")

    # 3. 모델 예측
    y_pred_train_reg = rf_reg_model.predict(X_train_reg)
    y_pred_test_reg = rf_reg_model.predict(X_test_reg)

    # 4. 모델 평가 (훈련 세트와 테스트 세트 모두)
    r2_train = r2_score(y_train_reg, y_pred_train_reg)
    r2_test = r2_score(y_test_reg, y_pred_test_reg)
    mse_test = mean_squared_error(y_test_reg, y_pred_test_reg)
    rmse_test = np.sqrt(mse_test)

    print(f"\n모델 평가 결과 (단일 분할):")
    print(f"  훈련 세트 R-squared: {r2_train:.4f}")
    print(f"  테스트 세트 R-squared: {r2_test:.4f}")
    print(f"  테스트 세트 MSE: {mse_test:.4f}")
    print(f"  테스트 세트 RMSE: {rmse_test:.4f}")

    # 과적합 여부 1차 판단
    if r2_train > r2_test and (r2_train - r2_test) > 0.1: # 예시 임계값, 상황에 따라 조절
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 커서 과적합 가능성이 있습니다.")
    elif r2_train > r2_test:
        print("  => 훈련 세트 R-squared가 테스트 세트보다 높지만, 큰 차이는 아니어서 일반적인 현상일 수 있습니다.")
    else:
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 크지 않아 과적합 위험이 낮을 수 있습니다.")


    print("\n--- K-Fold 교차 검증 시작 (5-Fold) ---")
    # 5. K-Fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_mse_scores = []

    for train_index, val_index in kf.split(X_reg):
        X_train_cv, X_val_cv = X_reg.iloc[train_index], X_reg.iloc[val_index]
        y_train_cv, y_val_cv = y_reg.iloc[train_index], y_reg.iloc[val_index]

        model_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_val_cv)

        cv_r2_scores.append(r2_score(y_val_cv, y_pred_cv))
        cv_mse_scores.append(mean_squared_error(y_val_cv, y_pred_cv))

    print(f"  교차 검증 R-squared 점수: {cv_r2_scores}")
    print(f"  교차 검증 평균 R-squared: {np.mean(cv_r2_scores):.4f} (표준편차: {np.std(cv_r2_scores):.4f})")
    print(f"  교차 검증 평균 MSE: {np.mean(cv_mse_scores):.4f} (표준편차: {np.std(cv_mse_scores):.4f})")

    if np.mean(cv_r2_scores) < (r2_test - 0.1): # 교차 검증 결과가 단일 테스트보다 훨씬 낮으면 과적합 의심
        print("  => 교차 검증 결과가 단일 테스트 결과보다 크게 낮아 과적합 위험이 있을 수 있습니다.")
    else:
        print("  => 교차 검증 결과가 단일 테스트 결과와 유사하여 모델이 비교적 안정적입니다.")


    # 6. 예측값과 실제값 시각화 (테스트 세트)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_reg, y=y_pred_test_reg, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2) # y=x 선
    plt.title('RandomForestRegressor: 실제값 vs 예측값 (테스트 세트)')
    plt.xlabel('실제 Total_Risk_Score')
    plt.ylabel('예측 Total_Risk_Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 7. 피처 중요도 시각화
    if hasattr(rf_reg_model, 'feature_importances_'):
        feature_importances = pd.Series(rf_reg_model.feature_importances_, index=X_reg.columns)
        top_n_features = 20 # 상위 20개 피처 시각화
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances.nlargest(top_n_features).values,
                    y=feature_importances.nlargest(top_n_features).index,
                    palette='viridis')
        plt.title(f'RandomForestRegressor: 상위 {top_n_features}개 피처 중요도')
        plt.xlabel('중요도')
        plt.ylabel('피처')
        plt.tight_layout()
        plt.show()
    print("\nRandomForestRegressor 모델 훈련 및 과적합 검증 완료.")

## Final Task

### Subtask:
새로운 Lot 단위 피처를 활용한 회귀 모델의 성능과 이전 모델 대비 개선 사항을 요약하고, 가장 중요한 피처들을 설명합니다.


## Summary:

### Data Analysis Key Findings
*   The `df_cleaned` DataFrame was successfully augmented with a `KMeans_Cluster` column and an `is_killer_defect` boolean column, the latter based on specific killer cluster mappings for 'REAL' defects per `Step_desc`.
*   A comprehensive set of 92 new Lot-level features was engineered for 474 Lots, including:
    *   Defect counts and proportions for combinations of `Step_desc`, `Class`, and `KMeans_Cluster`.
    *   Mean and median statistics for 11 key numerical features (e.g., `SIZE_X`, `DEFECT_AREA`, `SNR_OFFSET_GL`) categorized by whether the defect was a 'killer' or 'non-killer'.
*   A `RandomForestRegressor` model, trained on the expanded dataset incorporating these new features, achieved strong performance in predicting `Total_Risk_Score` on the test set:
    *   Mean Squared Error (MSE): 0.0018
    *   Root Mean Squared Error (RMSE): 0.0422
    *   R-squared (R2 Score): 0.8332, indicating a good fit and significant explanatory power.
*   Visualizations of actual versus predicted values and the top 20 feature importances were successfully generated, aiding in model interpretability.

### Insights or Next Steps
*   The significant improvement in the R-squared score (0.8332) suggests that the new Lot-level features, especially those differentiating between killer and non-killer defects and incorporating cluster information, are highly valuable for predicting `Total_Risk_Score`.
*   Further analysis of the top feature importances from the RandomForestRegressor model is recommended to identify which specific newly engineered features contribute most to the prediction and to gain deeper insights into the drivers of `Total_Risk_Score`.


# Task
새로운 회귀 모델(XGBoost Regressor)을 선택하고, 이전 단계에서 준비된 Lot 단위 예측 데이터셋(`X_reg`, `y_reg`)으로 훈련 세트와 테스트 세트를 분할한 후 모델을 학습시킵니다. 이전에 사용된 RandomForestRegressor와 비교할 수 있도록 동일한 데이터 분할 방식을 사용합니다.

## 새로운 회귀 모델 선택 및 훈련 (XGBoost Regressor)

### Subtask:
XGBoost Regressor 모델을 선택하고, 이전 단계에서 준비된 Lot 단위 예측 데이터셋(`X_reg`, `y_reg`)으로 훈련 세트와 테스트 세트를 분할한 후 모델을 학습시킵니다. 이전에 사용된 RandomForestRegressor와 비교할 수 있도록 동일한 데이터 분할 방식을 사용합니다.


**Reasoning**:
I will prepare the data by checking for `X_reg` and `y_reg`, then split it into training and testing sets. After that, I will import and initialize an `XGBRegressor` model and train it using the prepared training data.



import xgboost as xgb
from sklearn.model_selection import train_test_split

# Assuming X_reg and y_reg are available from the previous step

if 'X_reg' not in locals() or 'y_reg' not in locals():
    print("Error: Regression features (X_reg) or target (y_reg) not found. Please ensure the data preparation step has been executed.")
else:
    print("--- XGBoost Regressor 모델 훈련 시작 ---")

    # 1. 데이터 분할 (훈련 세트와 테스트 세트) - RandomForestRegressor와 동일한 방식 사용
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"훈련 세트 크기: {X_train_reg.shape[0]}")
    print(f"테스트 세트 크기: {X_test_reg.shape[0]}")

    # 2. XGBRegressor 모델 초기화 및 훈련
    xgb_reg_model = xgb.XGBRegressor(random_state=42, n_jobs=-1) # n_jobs=-1로 병렬 처리
    xgb_reg_model.fit(X_train_reg, y_train_reg)

    print("\nXGBoost Regressor 모델 훈련 완료.")

**Reasoning**:
The XGBoost Regressor model has been trained. Now, I need to evaluate its performance using metrics such as MSE, RMSE, and R-squared, and visualize the actual vs. predicted values and feature importances. This will follow the pattern used for the RandomForestRegressor evaluation.



from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold # Import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming xgb_reg_model, X_train_reg, X_test_reg, y_train_reg, y_test_reg are available from the previous step

if 'xgb_reg_model' not in locals() or 'X_test_reg' not in locals() or 'y_test_reg' not in locals():
    print("Error: Trained model or test data not found. Please ensure previous steps are executed.")
else:
    print("--- XGBoost Regressor 모델 훈련 및 과적합 검증 시작 ---")

    # Predict on the training and test data
    y_pred_train_xgb = xgb_reg_model.predict(X_train_reg)
    y_pred_test_xgb = xgb_reg_model.predict(X_test_reg)

    # Calculate and print evaluation metrics for test set
    mse_xgb = mean_squared_error(y_test_reg, y_pred_test_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_test_xgb = r2_score(y_test_reg, y_pred_test_xgb)
    r2_train_xgb = r2_score(y_train_reg, y_pred_train_xgb)

    print(f"\n모델 평가 결과 (XGBoost Regressor - 단일 분할):")
    print(f"  훈련 세트 R-squared: {r2_train_xgb:.4f}")
    print(f"  테스트 세트 R-squared: {r2_test_xgb:.4f}")
    print(f"  테스트 세트 MSE: {mse_xgb:.4f}")
    print(f"  테스트 세트 RMSE: {rmse_xgb:.4f}")

    # Check for overfitting (1차 판단)
    if r2_train_xgb > r2_test_xgb and (r2_train_xgb - r2_test_xgb) > 0.1: # 예시 임계값, 상황에 따라 조절
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 커서 과적합 가능성이 있습니다.")
    elif r2_train_xgb > r2_test_xgb:
        print("  => 훈련 세트 R-squared가 테스트 세트보다 높지만, 큰 차이는 아니어서 일반적인 현상일 수 있습니다.")
    else:
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 크지 않아 과적합 위험이 낮을 수 있습니다.")

    print("\n--- K-Fold 교차 검증 시작 (5-Fold) ---")
    # K-Fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores_xgb = []
    cv_mse_scores_xgb = []

    for train_index, val_index in kf.split(X_reg):
        X_train_cv, X_val_cv = X_reg.iloc[train_index], X_reg.iloc[val_index]
        y_train_cv, y_val_cv = y_reg.iloc[train_index], y_reg.iloc[val_index]

        model_cv_xgb = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        model_cv_xgb.fit(X_train_cv, y_train_cv)
        y_pred_cv_xgb = model_cv_xgb.predict(X_val_cv)

        cv_r2_scores_xgb.append(r2_score(y_val_cv, y_pred_cv_xgb))
        cv_mse_scores_xgb.append(mean_squared_error(y_val_cv, y_pred_cv_xgb))

    print(f"  교차 검증 R-squared 점수: {cv_r2_scores_xgb}")
    print(f"  교차 검증 평균 R-squared: {np.mean(cv_r2_scores_xgb):.4f} (표준편차: {np.std(cv_r2_scores_xgb):.4f})")
    print(f"  교차 검증 평균 MSE: {np.mean(cv_mse_scores_xgb):.4f} (표준편차: {np.std(cv_mse_scores_xgb):.4f})")

    if np.mean(cv_r2_scores_xgb) < (r2_test_xgb - 0.1): # 교차 검증 결과가 단일 테스트보다 훨씬 낮으면 과적합 의심
        print("  => 교차 검증 결과가 단일 테스트 결과보다 크게 낮아 과적합 위험이 있을 수 있습니다.")
    else:
        print("  => 교차 검증 결과가 단일 테스트 결과와 유사하여 모델이 비교적 안정적입니다.")

    # Visualize actual vs. predicted values for the test set
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_reg, y=y_pred_test_xgb, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2) # y=x line
    plt.title('XGBoost Regressor: 실제값 vs 예측값 (테스트 세트)')
    plt.xlabel('실제 Total_Risk_Score')
    plt.ylabel('예측 Total_Risk_Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Visualize feature importances
    if hasattr(xgb_reg_model, 'feature_importances_'):
        feature_importances_xgb = pd.Series(xgb_reg_model.feature_importances_, index=X_reg.columns)
        top_n_features = 20 # Visualize top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances_xgb.nlargest(top_n_features).values,
                    y=feature_importances_xgb.nlargest(top_n_features).index,
                    palette='viridis')
        plt.title(f'XGBoost Regressor: 상위 {top_n_features}개 피처 중요도')
        plt.xlabel('중요도')
        plt.ylabel('피처')
        plt.tight_layout()
        plt.show()

    print("\nXGBoost Regressor 모델 훈련 및 과적합 검증 완료.")

## Summary: XGBoost Regressor Model

### Data Analysis Key Findings
*   **Model Performance**: The XGBoost Regressor model achieved an excellent R-squared score of **0.9527** on the test set, with a Mean Squared Error (MSE) of **0.0005** and a Root Mean Squared Error (RMSE) of **0.0225**. The training R-squared was 1.0000, indicating a very strong fit to the training data. The difference between training and test R-squared suggests minimal overfitting given the high test performance.
*   **Comparison with RandomForestRegressor**: The XGBoost model shows a significant improvement in performance compared to the previously trained RandomForestRegressor, which had an R-squared of 0.8332 and an RMSE of 0.0422. The XGBoost model's higher R-squared and lower RMSE indicate better predictive accuracy for the `Total_Risk_Score`.
*   **Feature Importances**: Similar to the Random Forest model, the XGBoost model's feature importance plot identifies the most influential features in predicting the `Total_Risk_Score`. Analyzing these features further can provide insights into what drives a Lot's risk.

### Insights or Next Steps
*   The XGBoost Regressor is clearly the superior model for predicting `Total_Risk_Score` based on the current feature set. Its robust performance suggests it can be reliably used for identifying high-risk Lots.
*   Further hyperparameter tuning for the XGBoost model could potentially yield even better performance, although current results are already very strong.
*   Investigate the top feature importances from the XGBoost model to understand the specific Lot-level characteristics that contribute most to the total risk score. This can guide targeted interventions to reduce defect rates.

## 두 모델 성능 비교 및 요약

### Subtask:
XGBoost Regressor 모델의 성능을 RandomForestRegressor 모델과 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.


## 두 모델 성능 비교 및 요약

### Subtask:
XGBoost Regressor 모델의 성능을 RandomForestRegressor 모델과 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.

#### Instructions
1. RandomForestRegressor 모델과 XGBoost Regressor 모델의 테스트 세트 성능 지표(R-squared, MSE, RMSE)를 각각 확인합니다.
2. 두 모델의 R-squared, MSE, RMSE 값을 비교하여 어떤 모델이 예측 성능이 더 우수한지 명확히 밝힙니다.
3. 두 모델의 성능 차이점과 각 모델의 장단점(예: XGBoost의 높은 성능과 잠재적 복잡성, Random Forest의 일반적인 견고함과 해석 용이성 등)을 분석하여 요약합니다.
4. 각 모델의 피처 중요도 시각화를 검토하여 예측에 가장 큰 영향을 미치는 피처들이 모델마다 어떻게 다르게 나타나는지 설명합니다.

### 모델 성능 비교:

**1. RandomForestRegressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.8332
*   **테스트 세트 MSE:** 0.0018
*   **테스트 세트 RMSE:** 0.0422

**2. XGBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9527
*   **테스트 세트 MSE:** 0.0005
*   **테스트 세트 RMSE:** 0.0225

### 분석 및 요약:

**성능 우수성:**
*   **XGBoost Regressor**가 RandomForestRegressor보다 모든 지표에서 현저히 우수한 성능을 보였습니다.
    *   R-squared는 0.8332에서 0.9527로 크게 향상되어, XGBoost 모델이 `Total_Risk_Score` 변동의 약 95%를 설명하는 반면 Random Forest는 83%를 설명합니다.
    *   MSE와 RMSE 값도 XGBoost가 훨씬 낮아 예측 오차가 더 작음을 나타냅니다.

**모델별 장단점 및 예측 결과 차이점:**

1.  **RandomForestRegressor:**
    *   **장점:** 구현이 비교적 간단하고, 과적합에 강하며, 해석이 용이한 편입니다. 기본적으로 강력하고 안정적인 성능을 제공합니다.
    *   **단점:** XGBoost와 같은 부스팅 모델에 비해 최적의 예측 성능을 내지 못할 수 있습니다. 특히 복잡한 데이터 패턴에서 잠재력을 충분히 발휘하지 못할 수 있습니다.
    *   **예측 결과:** 훈련 세트 R-squared(0.9710)와 테스트 세트 R-squared(0.8332) 간에 약간의 차이가 있어, 어느 정도의 과적합 가능성이 있지만, 교차 검증 결과(평균 R-squared 0.8881)를 고려할 때 모델이 비교적 안정적으로 작동한다고 볼 수 있습니다.

2.  **XGBoost Regressor:**
    *   **장점:** 뛰어난 예측 성능을 자랑하며, 다양한 종류의 데이터와 문제에 잘 적용됩니다. 병렬 처리가 가능하고, 결측값 처리, 정규화 등 고급 기능이 내장되어 있습니다.
    *   **단점:** 하이퍼파라미터 튜닝이 복잡하고, 과적합 가능성이 있어 세심한 주의가 필요합니다. 모델이 블랙박스에 가까워 해석이 Random Forest보다 어렵습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9527) 모두 매우 높게 나타났습니다. 훈련 세트에서의 완벽한 성능은 잠재적인 과적합 신호일 수 있지만, 테스트 세트에서의 높은 R-squared 값(0.9527)이 이를 뒷받침하므로 강력한 일반화 성능을 보여줍니다.

**피처 중요도 (상위 20개 비교):**

두 모델 모두 `Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count`와 같은 Lot 단위 집계 피처들을 중요한 피처로 간주하고 있습니다.

*   **RandomForestRegressor**의 경우, `Total_Count`의 중요도가 압도적으로 높고, `Slot_No_nunique`가 그 뒤를 잇습니다. 이는 Lot의 전체 결함 수와 슬롯 고유 개수가 위험도 예측에 매우 중요함을 시사합니다.
*   **XGBoost Regressor** 또한 `Slot_No_nunique`와 `Total_Count`를 가장 중요한 피처로 평가하지만, 상대적으로 `Slot_No_nunique`의 중요도가 `Total_Count`보다 훨씬 높게 나타났습니다. 이는 XGBoost가 `Slot_No_nunique`의 미묘한 패턴을 더 잘 학습하여 예측에 활용했음을 의미할 수 있습니다.
*   두 모델 모두 `DEFECT_AREA_mean`, `SIZE_X_mean` 등 결함의 평균적인 크기 관련 피처와 `PATCHDEFECTSIGNAL_median`, `RELATIVEMAGNITUDE_median` 등 결함 신호 관련 통계량 피처들이 예측에 기여하는 것으로 나타났습니다.

**결론적으로, XGBoost Regressor는 Lot 단위의 통합 위험도 점수 예측에서 RandomForestRegressor보다 훨씬 뛰어난 성능을 제공하며, 이는 복잡한 패턴을 학습하는 데 더 적합한 부스팅 모델의 특성을 반영합니다. 따라서 이 예측 작업에는 XGBoost 모델이 더 권장됩니다.**

## 두 모델 성능 비교 및 요약

### Subtask:
XGBoost Regressor 모델의 성능을 RandomForestRegressor 모델과 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.

#### Instructions
1. RandomForestRegressor 모델과 XGBoost Regressor 모델의 테스트 세트 성능 지표(R-squared, MSE, RMSE)를 각각 확인합니다.
2. 두 모델의 R-squared, MSE, RMSE 값을 비교하여 어떤 모델이 예측 성능이 더 우수한지 명확히 밝힙니다.
3. 두 모델의 성능 차이점과 각 모델의 장단점(예: XGBoost의 높은 성능과 잠재적 복잡성, Random Forest의 일반적인 견고함과 해석 용이성 등)을 분석하여 요약합니다.
4. 각 모델의 피처 중요도 시각화를 검토하여 예측에 가장 큰 영향을 미치는 피처들이 모델마다 어떻게 다르게 나타나는지 설명합니다.

### 모델 성능 비교:

**1. RandomForestRegressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.8332
*   **테스트 세트 MSE:** 0.0018
*   **테스트 세트 RMSE:** 0.0422

**2. XGBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9527
*   **테스트 세트 MSE:** 0.0005
*   **테스트 세트 RMSE:** 0.0225

### 분석 및 요약:

**성능 우수성:**
*   **XGBoost Regressor**가 RandomForestRegressor보다 모든 지표에서 현저히 우수한 성능을 보였습니다.
    *   R-squared는 0.8332에서 0.9527로 크게 향상되어, XGBoost 모델이 `Total_Risk_Score` 변동의 약 95%를 설명하는 반면 Random Forest는 83%를 설명합니다.
    *   MSE와 RMSE 값도 XGBoost가 훨씬 낮아 예측 오차가 더 작음을 나타냅니다.

**모델별 장단점 및 예측 결과 차이점:**

1.  **RandomForestRegressor:**
    *   **장점:** 구현이 비교적 간단하고, 과적합에 강하며, 해석이 용이한 편입니다. 기본적으로 강력하고 안정적인 성능을 제공합니다.
    *   **단점:** XGBoost와 같은 부스팅 모델에 비해 최적의 예측 성능을 내지 못할 수 있습니다. 특히 복잡한 데이터 패턴에서 잠재력을 충분히 발휘하지 못할 수 있습니다.
    *   **예측 결과:** 훈련 세트 R-squared(0.9710)와 테스트 세트 R-squared(0.8332) 사이에 약간의 차이가 있어, 어느 정도의 과적합 가능성이 있지만, 교차 검증 결과(평균 R-squared 0.8881)를 고려할 때 모델이 비교적 안정적으로 작동한다고 볼 수 있습니다.

2.  **XGBoost Regressor:**
    *   **장점:** 뛰어난 예측 성능을 자랑하며, 다양한 종류의 데이터와 문제에 잘 적용됩니다. 병렬 처리가 가능하고, 결측값 처리, 정규화 등 고급 기능이 내장되어 있습니다.
    *   **단점:** 하이퍼파라미터 튜닝이 복잡하고, 과적합 가능성이 있어 세심한 주의가 필요합니다. 모델이 블랙박스에 가까워 해석이 Random Forest보다 어렵습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9527) 모두 매우 높게 나타났습니다. 훈련 세트에서의 완벽한 성능은 잠재적인 과적합 신호일 수 있지만, 테스트 세트에서의 높은 R-squared 값(0.9527)이 이를 뒷받침하므로 강력한 일반화 성능을 보여줍니다.

**피처 중요도 (상위 20개 비교):**

두 모델 모두 `Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count`와 같은 Lot 단위 집계 피처들을 중요한 피처로 간주하고 있습니다.

*   **RandomForestRegressor**의 경우, `Total_Count`의 중요도가 압도적으로 높고, `Slot_No_nunique`가 그 뒤를 잇습니다. 이는 Lot의 전체 결함 수와 슬롯 고유 개수가 위험도 예측에 매우 중요함을 시사합니다.
*   **XGBoost Regressor** 또한 `Slot_No_nunique`와 `Total_Count`를 가장 중요한 피처로 평가하지만, 상대적으로 `Slot_No_nunique`의 중요도가 `Total_Count`보다 훨씬 높게 나타났습니다. 이는 XGBoost가 `Slot_No_nunique`의 미묘한 패턴을 더 잘 학습하여 예측에 활용했음을 의미할 수 있습니다.
*   두 모델 모두 `DEFECT_AREA_mean`, `SIZE_X_mean` 등 결함의 평균적인 크기 관련 피처와 `PATCHDEFECTSIGNAL_median`, `RELATIVEMAGNITUDE_median` 등 결함 신호 관련 통계량 피처들이 예측에 기여하는 것으로 나타났습니다.

**결론적으로, XGBoost Regressor는 Lot 단위의 통합 위험도 점수 예측에서 RandomForestRegressor보다 훨씬 뛰어난 성능을 제공하며, 이는 복잡한 패턴을 학습하는 데 더 적합한 부스팅 모델의 특성을 반영합니다. 따라서 이 예측 작업에는 XGBoost 모델이 더 권장됩니다.**

## Final Task

### Subtask:
새로운 회귀 모델(XGBoost Regressor)의 성능과 이전 모델(RandomForestRegressor) 대비 개선 사항을 요약하고, 가장 중요한 피처들을 설명합니다.


## Summary:

### Q&A
1.  **What is the performance of the new regression model (XGBoost Regressor) compared to the previous model (RandomForestRegressor)?**
    The XGBoost Regressor model demonstrates significantly superior performance compared to the RandomForestRegressor.
    *   XGBoost achieved a test R-squared of 0.9527, a test MSE of 0.0005, and a test RMSE of 0.0225.
    *   RandomForestRegressor had a test R-squared of 0.8332, a test MSE of 0.0018, and a test RMSE of 0.0422.
    XGBoost explains approximately 95% of the `Total_Risk_Score` variance, while RandomForestRegressor explains about 83%, indicating a substantial improvement in predictive accuracy.

2.  **What are the key improvement areas and most important features explained by the new model?**
    The XGBoost model significantly improved the R-squared by 0.1195 (from 0.8332 to 0.9527) and reduced the RMSE by nearly half (from 0.0422 to 0.0225). Both models identified `Total_Count` and `Slot_No_nunique` as critical features. However, XGBoost showed a relatively higher importance for `Slot_No_nunique` compared to `Total_Count`, suggesting it leverages more subtle patterns from this feature for prediction. Other defect-related statistical features like `DEFECT_AREA_mean`, `SIZE_X_mean`, `PATCHDEFECTSIGNAL_median`, and `RELATIVEMAGNITUDE_median` also played a role in both models.

### Data Analysis Key Findings
*   The XGBoost Regressor model achieved a strong performance with a test R-squared of 0.9527, a Mean Squared Error (MSE) of 0.0005, and a Root Mean Squared Error (RMSE) of 0.0225 on the test set.
*   The training R-squared for XGBoost was 1.0000, and the difference with the test R-squared suggests strong generalization capability with minimal signs of significant overfitting, given the high test performance.
*   Compared to the RandomForestRegressor, XGBoost showed a marked improvement, with its test R-squared of 0.9527 substantially outperforming RandomForest's 0.8332, and its RMSE of 0.0225 being significantly lower than RandomForest's 0.0422.
*   Both models highlighted `Total_Count` and `Slot_No_nunique` as the most influential features. While RandomForest emphasized `Total_Count`, XGBoost assigned relatively higher importance to `Slot_No_nunique`, indicating its ability to capture nuanced patterns from this feature.

### Insights or Next Steps
*   The XGBoost Regressor is the recommended model for predicting `Total_Risk_Score` due to its significantly superior performance and robust generalization capabilities.
*   Further hyperparameter tuning of the XGBoost model could potentially extract even marginal improvements, while a deeper analysis of the most important features could provide actionable insights into specific Lot-level characteristics driving risk.




# Task
## 새로운 회귀 모델 선택 및 훈련 (CatBoost Regressor)

### Subtask:
CatBoost Regressor 모델을 선택하고, 이전 단계에서 준비된 Lot 단위 예측 데이터셋(`X_reg`, `y_reg`)으로 훈련 세트와 테스트 세트를 분할한 후 모델을 학습시킵니다. 이전에 사용된 RandomForestRegressor 및 XGBoost Regressor와 비교할 수 있도록 동일한 데이터 분할 방식을 사용합니다.

## 새로운 회귀 모델 선택 및 훈련 (CatBoost Regressor)

### Subtask:
CatBoost Regressor 모델을 선택하고, 이전 단계에서 준비된 Lot 단위 예측 데이터셋(`X_reg`, `y_reg`)으로 훈련 세트와 테스트 세트를 분할한 후 모델을 학습시킵니다. 이전에 사용된 RandomForestRegressor 및 XGBoost Regressor와 비교할 수 있도록 동일한 데이터 분할 방식을 사용합니다.


**Reasoning**:
The subtask requires training a CatBoost Regressor model. I will first import the necessary libraries, split the data into training and testing sets using the specified parameters, then initialize and train the CatBoostRegressor model.



import catboost as cb
from sklearn.model_selection import train_test_split

# Assuming X_reg and y_reg are available from the previous step

if 'X_reg' not in locals() or 'y_reg' not in locals():
    print("Error: Regression features (X_reg) or target (y_reg) not found. Please ensure the data preparation step has been executed.")
else:
    print("--- CatBoost Regressor 모델 훈련 시작 ---")

    # 1. 데이터 분할 (훈련 세트와 테스트 세트) - 이전 모델들과 동일한 방식 사용
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"훈련 세트 크기: {X_train_reg.shape[0]}")
    print(f"테스트 세트 크기: {X_test_reg.shape[0]}")

    # 2. CatBoostRegressor 모델 초기화 및 훈련
    cat_reg_model = cb.CatBoostRegressor(random_state=42, verbose=0) # verbose=0 for silent training
    cat_reg_model.fit(X_train_reg, y_train_reg)

    print("\nCatBoost Regressor 모델 훈련 완료.")

**Reasoning**:
The previous code failed because the `catboost` library was not found. I need to install it first using `pip`.



import sys
!{sys.executable} -m pip install catboost

**Reasoning**:
Now that the `catboost` library is installed, I can proceed with training the CatBoost Regressor model. This involves importing the necessary libraries, splitting the data into training and testing sets, and then initializing and training the `CatBoostRegressor` model.



import catboost as cb
from sklearn.model_selection import train_test_split

# Assuming X_reg and y_reg are available from the previous step

if 'X_reg' not in locals() or 'y_reg' not in locals():
    print("Error: Regression features (X_reg) or target (y_reg) not found. Please ensure the data preparation step has been executed.")
else:
    print("--- CatBoost Regressor 모델 훈련 시작 ---")

    # 1. 데이터 분할 (훈련 세트와 테스트 세트) - 이전 모델들과 동일한 방식 사용
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"훈련 세트 크기: {X_train_reg.shape[0]}")
    print(f"테스트 세트 크기: {X_test_reg.shape[0]}")

    # 2. CatBoostRegressor 모델 초기화 및 훈련
    cat_reg_model = cb.CatBoostRegressor(random_state=42, verbose=0) # verbose=0 for silent training
    cat_reg_model.fit(X_train_reg, y_train_reg)

    print("\nCatBoost Regressor 모델 훈련 완료.")

**Reasoning**:
The CatBoost Regressor model has been trained. Now, I need to evaluate its performance using metrics such as MSE, RMSE, and R-squared, and visualize the actual vs. predicted values and feature importances. This will follow the pattern used for the RandomForestRegressor and XGBoost Regressor evaluation to ensure consistency for comparison.



from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold # Import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming cat_reg_model, X_train_reg, X_test_reg, y_train_reg, y_test_reg are available from the previous step

if 'cat_reg_model' not in locals() or 'X_test_reg' not in locals() or 'y_test_reg' not in locals():
    print("Error: Trained model or test data not found. Please ensure previous steps are executed.")
else:
    print("--- CatBoost Regressor 모델 훈련 및 과적합 검증 시작 ---")

    # Predict on the training and test data
    y_pred_train_cat = cat_reg_model.predict(X_train_reg)
    y_pred_test_cat = cat_reg_model.predict(X_test_reg)

    # Calculate and print evaluation metrics for test set
    mse_cat = mean_squared_error(y_test_reg, y_pred_test_cat)
    rmse_cat = np.sqrt(mse_cat)
    r2_test_cat = r2_score(y_test_reg, y_pred_test_cat)
    r2_train_cat = r2_score(y_train_reg, y_pred_train_cat)

    print(f"\n모델 평가 결과 (CatBoost Regressor - 단일 분할):")
    print(f"  훈련 세트 R-squared: {r2_train_cat:.4f}")
    print(f"  테스트 세트 R-squared: {r2_test_cat:.4f}")
    print(f"  테스트 세트 MSE: {mse_cat:.4f}")
    print(f"  테스트 세트 RMSE: {rmse_cat:.4f}")

    # Check for overfitting (1차 판단)
    if r2_train_cat > r2_test_cat and (r2_train_cat - r2_test_cat) > 0.1: # 예시 임계값, 상황에 따라 조절
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 커서 과적합 가능성이 있습니다.")
    elif r2_train_cat > r2_test_cat:
        print("  => 훈련 세트 R-squared가 테스트 세트보다 높지만, 큰 차이는 아니어서 일반적인 현상일 수 있습니다.")
    else:
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 크지 않아 과적합 위험이 낮을 수 있습니다.")

    print("\n--- K-Fold 교차 검증 시작 (5-Fold) ---")
    # K-Fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores_cat = []
    cv_mse_scores_cat = []

    for train_index, val_index in kf.split(X_reg):
        X_train_cv, X_val_cv = X_reg.iloc[train_index], X_reg.iloc[val_index]
        y_train_cv, y_val_cv = y_reg.iloc[train_index], y_reg.iloc[val_index]

        model_cv_cat = cb.CatBoostRegressor(random_state=42, verbose=0)
        model_cv_cat.fit(X_train_cv, y_train_cv)
        y_pred_cv_cat = model_cv_cat.predict(X_val_cv)

        cv_r2_scores_cat.append(r2_score(y_val_cv, y_pred_cv_cat))
        cv_mse_scores_cat.append(mean_squared_error(y_val_cv, y_pred_cv_cat))

    print(f"  교차 검증 R-squared 점수: {cv_r2_scores_cat}")
    print(f"  교차 검증 평균 R-squared: {np.mean(cv_r2_scores_cat):.4f} (표준편차: {np.std(cv_r2_scores_cat):.4f})")
    print(f"  교차 검증 평균 MSE: {np.mean(cv_mse_scores_cat):.4f} (표준편차: {np.std(cv_mse_scores_cat):.4f})")

    if np.mean(cv_r2_scores_cat) < (r2_test_cat - 0.1): # 교차 검증 결과가 단일 테스트보다 훨씬 낮으면 과적합 의심
        print("  => 교차 검증 결과가 단일 테스트 결과보다 크게 낮아 과적합 위험이 있을 수 있습니다.")
    else:
        print("  => 교차 검증 결과가 단일 테스트 결과와 유사하여 모델이 비교적 안정적입니다.")

    # Visualize actual vs. predicted values for the test set
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_reg, y=y_pred_test_cat, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2) # y=x line
    plt.title('CatBoost Regressor: 실제값 vs 예측값 (테스트 세트)')
    plt.xlabel('실제 Total_Risk_Score')
    plt.ylabel('예측 Total_Risk_Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Visualize feature importances
    if hasattr(cat_reg_model, 'feature_importances_'):
        feature_importances_cat = pd.Series(cat_reg_model.get_feature_importance(), index=X_reg.columns)
        top_n_features = 20 # Visualize top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances_cat.nlargest(top_n_features).values,
                    y=feature_importances_cat.nlargest(top_n_features).index,
                    palette='viridis')
        plt.title(f'CatBoost Regressor: 상위 {top_n_features}개 피처 중요도')
        plt.xlabel('중요도')
        plt.ylabel('피처')
        plt.tight_layout()
        plt.show()

    print("\nCatBoost Regressor 모델 훈련 및 과적합 검증 완료.")

## 세 모델 성능 비교 및 요약

### Subtask:
RandomForestRegressor, XGBoost Regressor, CatBoost Regressor 세 모델의 성능을 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.


## 세 모델 성능 비교 및 요약

### Subtask:
RandomForestRegressor, XGBoost Regressor, CatBoost Regressor 세 모델의 성능을 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.

#### Instructions
1. RandomForestRegressor, XGBoost Regressor, CatBoost Regressor 각 모델의 테스트 세트 성능 지표(R-squared, MSE, RMSE)를 확인합니다.
2. 세 모델의 R-squared, MSE, RMSE 값을 비교하여 어떤 모델이 예측 성능이 가장 우수한지 명확히 밝힙니다.
3. 세 모델의 성능 차이점과 각 모델의 장단점(예: XGBoost의 높은 성능과 잠재적 복잡성, Random Forest의 일반적인 견고함과 해석 용이성, CatBoost의 범주형 피처 처리 능력)을 분석하여 요약합니다.
4. 각 모델의 피처 중요도 시각화를 검토하여 예측에 가장 큰 영향을 미치는 피처들이 모델마다 어떻게 다르게 나타나는지 설명합니다.

### 모델 성능 비교:

**1. RandomForestRegressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.8910
*   **테스트 세트 MSE:** 0.0007
*   **테스트 세트 RMSE:** 0.0256

**2. XGBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9334
*   **테스트 세트 MSE:** 0.0004
*   **테스트 세트 RMSE:** 0.0200

**3. CatBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9686
*   **테스트 세트 MSE:** 0.0002
*   **테스트 세트 RMSE:** 0.0137

### 분석 및 요약:

**성능 우수성:**
*   **CatBoost Regressor**가 세 모델 중 가장 우수한 성능을 보였습니다.
    *   **CatBoost**는 테스트 세트 R-squared 0.9686으로, `Total_Risk_Score` 변동의 약 97%를 설명합니다.
    *   **XGBoost**는 테스트 세트 R-squared 0.9334로, `Total_Risk_Score` 변동의 약 93%를 설명하며 CatBoost 다음으로 우수합니다.
    *   **RandomForest**는 테스트 세트 R-squared 0.8910으로, 세 모델 중 가장 낮은 성능을 보였습니다.
*   MSE와 RMSE 값 또한 CatBoost가 가장 낮고, XGBoost, RandomForest 순으로 오차가 커지는 경향을 보였습니다. 이는 예측 정확도 측면에서 CatBoost가 가장 뛰어나고, XGBoost가 그 뒤를 잇는다는 것을 의미합니다.

**모델별 장단점 및 예측 결과 차이점:**

1.  **RandomForestRegressor:**
    *   **장점:** 구현이 비교적 간단하고, 과적합에 강하며, 해석이 용이한 편입니다. 기본적으로 강력하고 안정적인 성능을 제공합니다.
    *   **단점:** XGBoost나 CatBoost와 같은 부스팅 모델에 비해 최적의 예측 성능을 내지 못할 수 있습니다. 특히 복잡한 데이터 패턴에서 잠재력을 충분히 발휘하지 못할 수 있습니다.
    *   **예측 결과:** 훈련 세트 R-squared(0.9764)와 테스트 세트 R-squared(0.8910) 간에 어느 정도 차이가 있으나, 교차 검증 평균 R-squared(0.8521)를 고려할 때 모델이 비교적 안정적으로 작동한다고 볼 수 있습니다.

2.  **XGBoost Regressor:**
    *   **장점:** 뛰어난 예측 성능을 자랑하며, 다양한 종류의 데이터와 문제에 잘 적용됩니다. 병렬 처리가 가능하고, 결측값 처리, 정규화 등 고급 기능이 내장되어 있습니다.
    *   **단점:** 하이퍼파라미터 튜닝이 복잡하고, 과적합 가능성이 있어 세심한 주의가 필요합니다. 모델이 블랙박스에 가까워 해석이 Random Forest보다 어렵습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9334) 모두 매우 높게 나타났습니다. 훈련 세트에서의 완벽한 성능은 잠재적인 과적합 신호일 수 있지만, 테스트 세트에서의 높은 R-squared 값이 이를 뒷받침하므로 강력한 일반화 성능을 보여줍니다.

3.  **CatBoost Regressor:**
    *   **장점:** 범주형 피처를 전처리 없이 직접 처리할 수 있는 강력한 기능이 있습니다. 기본적으로 과적합 방지 기능이 내장되어 있으며, 높은 예측 성능을 제공합니다. 안정적이고 견고한 학습 과정을 가집니다.
    *   **단점:** XGBoost와 마찬가지로 하이퍼파라미터 튜닝이 복잡할 수 있으며, 학습 시간이 상대적으로 길어질 수 있습니다. 모델 해석의 용이성은 Random Forest보다 낮습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9686) 모두 매우 높게 나타났으며, 특히 테스트 세트 성능이 가장 뛰어납니다. 훈련 세트에서의 완벽한 성능에도 불구하고 테스트 세트에서 강력한 성능을 유지하여 뛰어난 일반화 능력을 보여줍니다.

**피처 중요도 (상위 20개 비교):**

세 모델 모두 Lot 단위의 집계 피처(`Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count` 등)들을 중요한 피처로 간주하고 있습니다. 각 모델의 피처 중요도 결과는 다음과 같습니다:

*   **RandomForestRegressor:** `Total_Count`의 중요도가 압도적으로 높고, `Slot_No_nunique`가 그 뒤를 잇습니다. 이는 Lot의 전체 결함 수와 슬롯 고유 개수가 위험도 예측에 매우 중요함을 시사합니다.
*   **XGBoost Regressor:** `Total_Count`와 `Slot_No_nunique`를 가장 중요한 피처로 평가하며, `Total_Count`가 가장 높은 중요도를 보입니다. 또한, `Killer_Defect_Proportion`, `DENSITY_SIGNAL_max`, `SIZE_X_min` 등 다양한 상세 집계 피처들이 상위 중요도에 포함됩니다.
*   **CatBoost Regressor:** `Total_Count`, `Killer_Defect_Count`, `Slot_No_nunique`를 포함하여 `SIZE_X_mean_non_killer_mean`, `SNR_OFFSET_GL_killer_mean` 등 킬러/비킬러 결함 특성에 따른 상세 통계 피처들을 매우 중요하게 평가했습니다. 특히 CatBoost는 `Killer_Defect_Count`와 `Slot_No_nunique_y` 피처에 높은 중요도를 부여한 것으로 나타납니다.

**결론적으로, CatBoost Regressor는 Lot 단위의 통합 위험도 점수 예측에서 가장 뛰어난 성능을 제공하며, 이는 복잡한 패턴과 다양한 유형의 피처(범주형 포함)를 학습하는 데 더 적합한 부스팅 모델의 특성을 반영합니다. 따라서 이 예측 작업에는 CatBoost 모델이 가장 권장됩니다.**

## 세 모델 성능 비교 및 요약

### Subtask:
RandomForestRegressor, XGBoost Regressor, CatBoost Regressor 세 모델의 성능을 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.

#### Instructions
1. RandomForestRegressor, XGBoost Regressor, CatBoost Regressor 각 모델의 테스트 세트 성능 지표(R-squared, MSE, RMSE)를 확인합니다.
2. 세 모델의 R-squared, MSE, RMSE 값을 비교하여 어떤 모델이 예측 성능이 가장 우수한지 명확히 밝힙니다.
3. 세 모델의 성능 차이점과 각 모델의 장단점(예: XGBoost의 높은 성능과 잠재적 복잡성, Random Forest의 일반적인 견고함과 해석 용이성, CatBoost의 범주형 피처 처리 능력)을 분석하여 요약합니다.
4. 각 모델의 피처 중요도 시각화를 검토하여 예측에 가장 큰 영향을 미치는 피처들이 모델마다 어떻게 다르게 나타나는지 설명합니다.

### 모델 성능 비교:

**1. RandomForestRegressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.8910
*   **테스트 세트 MSE:** 0.0007
*   **테스트 세트 RMSE:** 0.0256

**2. XGBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9334
*   **테스트 세트 MSE:** 0.0004
*   **테스트 세트 RMSE:** 0.0200

**3. CatBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9686
*   **테스트 세트 MSE:** 0.0002
*   **테스트 세트 RMSE:** 0.0137

### 분석 및 요약:

**성능 우수성:**
*   **CatBoost Regressor**가 세 모델 중 가장 우수한 성능을 보였습니다.
    *   **CatBoost**는 테스트 세트 R-squared 0.9686으로, `Total_Risk_Score` 변동의 약 97%를 설명합니다.
    *   **XGBoost**는 테스트 세트 R-squared 0.9334로, `Total_Risk_Score` 변동의 약 93%를 설명하며 CatBoost 다음으로 우수합니다.
    *   **RandomForest**는 테스트 세트 R-squared 0.8910으로, 세 모델 중 가장 낮은 성능을 보였습니다.
*   MSE와 RMSE 값 또한 CatBoost가 가장 낮고, XGBoost, RandomForest 순으로 오차가 커지는 경향을 보였습니다. 이는 예측 정확도 측면에서 CatBoost가 가장 뛰어나고, XGBoost가 그 뒤를 잇는다는 것을 의미합니다.

**모델별 장단점 및 예측 결과 차이점:**

1.  **RandomForestRegressor:**
    *   **장점:** 구현이 비교적 간단하고, 과적합에 강하며, 해석이 용이한 편입니다. 기본적으로 강력하고 안정적인 성능을 제공합니다.
    *   **단점:** XGBoost나 CatBoost와 같은 부스팅 모델에 비해 최적의 예측 성능을 내지 못할 수 있습니다. 특히 복잡한 데이터 패턴에서 잠재력을 충분히 발휘하지 못할 수 있습니다.
    *   **예측 결과:** 훈련 세트 R-squared(0.9764)와 테스트 세트 R-squared(0.8910) 간에 어느 정도 차이가 있으나, 교차 검증 평균 R-squared(0.8521)를 고려할 때 모델이 비교적 안정적으로 작동한다고 볼 수 있습니다.

2.  **XGBoost Regressor:**
    *   **장점:** 뛰어난 예측 성능을 자랑하며, 다양한 종류의 데이터와 문제에 잘 적용됩니다. 병렬 처리가 가능하고, 결측값 처리, 정규화 등 고급 기능이 내장되어 있습니다.
    *   **단점:** 하이퍼파라미터 튜닝이 복잡하고, 과적합 가능성이 있어 세심한 주의가 필요합니다. 모델이 블랙박스에 가까워 해석이 Random Forest보다 어렵습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9334) 모두 매우 높게 나타났습니다. 훈련 세트에서의 완벽한 성능은 잠재적인 과적합 신호일 수 있지만, 테스트 세트에서의 높은 R-squared 값이 이를 뒷받침하므로 강력한 일반화 성능을 보여줍니다.

3.  **CatBoost Regressor:**
    *   **장점:** 범주형 피처를 전처리 없이 직접 처리할 수 있는 강력한 기능이 있습니다. 기본적으로 과적합 방지 기능이 내장되어 있으며, 높은 예측 성능을 제공합니다. 안정적이고 견고한 학습 과정을 가집니다.
    *   **단점:** XGBoost와 마찬가지로 하이퍼파라미터 튜닝이 복잡할 수 있으며, 학습 시간이 상대적으로 길어질 수 있습니다. 모델 해석의 용이성은 Random Forest보다 낮습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9686) 모두 매우 높게 나타났으며, 특히 테스트 세트 성능이 가장 뛰어납니다. 훈련 세트에서의 완벽한 성능에도 불구하고 테스트 세트에서 강력한 성능을 유지하여 뛰어난 일반화 능력을 보여줍니다.

**피처 중요도 (상위 20개 비교):**

세 모델 모두 Lot 단위의 집계 피처(`Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count` 등)들을 중요한 피처로 간주하고 있습니다. 각 모델의 피처 중요도 결과는 다음과 같습니다:

*   **RandomForestRegressor:** `Total_Count`의 중요도가 압도적으로 높고, `Slot_No_nunique`가 그 뒤를 잇습니다. 이는 Lot의 전체 결함 수와 슬롯 고유 개수가 위험도 예측에 매우 중요함을 시사합니다.
*   **XGBoost Regressor:** `Total_Count`와 `Slot_No_nunique`를 가장 중요한 피처로 평가하며, `Total_Count`가 가장 높은 중요도를 보입니다. 또한, `Killer_Defect_Proportion`, `DENSITY_SIGNAL_max`, `SIZE_X_min` 등 다양한 상세 집계 피처들이 상위 중요도에 포함됩니다.
*   **CatBoost Regressor:** `Total_Count`, `Killer_Defect_Count`, `Slot_No_nunique`를 포함하여 `SIZE_X_mean_non_killer_mean`, `SNR_OFFSET_GL_killer_mean` 등 킬러/비킬러 결함 특성에 따른 상세 통계 피처들을 매우 중요하게 평가했습니다. 특히 CatBoost는 `Killer_Defect_Count`와 `Slot_No_nunique_y` 피처에 높은 중요도를 부여한 것으로 나타납니다.

**결론적으로, CatBoost Regressor는 Lot 단위의 통합 위험도 점수 예측에서 가장 뛰어난 성능을 제공하며, 이는 복잡한 패턴과 다양한 유형의 피처(범주형 포함)를 학습하는 데 더 적합한 부스팅 모델의 특성을 반영합니다. 따라서 이 예측 작업에는 CatBoost 모델이 가장 권장됩니다.**


## Final Task

### Subtask:
새로운 회귀 모델(CatBoost Regressor)의 성능과 이전 모델(RandomForestRegressor, XGBoost Regressor) 대비 개선 사항을 요약하고, 가장 중요한 피처들을 설명합니다.


## Summary:

### Q&A

1.  **새로운 회귀 모델(CatBoost Regressor)의 성능과 이전 모델(RandomForestRegressor, XGBoost Regressor) 대비 개선 사항은 무엇인가요?**
    CatBoost Regressor 모델은 테스트 세트에서 R-squared 0.9686, MSE 0.0002, RMSE 0.0137을 기록하여, 세 모델 중 가장 우수한 성능을 보였습니다. 이는 이전 모델인 RandomForestRegressor(R-squared 0.8910, MSE 0.0007, RMSE 0.0256) 및 XGBoost Regressor(R-squared 0.9334, MSE 0.0004, RMSE 0.0200) 대비 가장 높은 설명력과 낮은 예측 오차를 달성하여 상당한 성능 개선을 보여줍니다.
2.  **세 모델(RandomForestRegressor, XGBoost Regressor, CatBoost Regressor) 중 어떤 모델이 예측 성능이 가장 우수합니까?**
    CatBoost Regressor가 테스트 세트 R-squared 0.9686으로, `Total_Risk_Score` 변동의 약 97%를 설명하며 세 모델 중 가장 우수한 예측 성능을 보였습니다.
3.  **각 모델의 장단점 및 예측 결과의 차이점은 무엇입니까?**
    *   **RandomForestRegressor:** 구현이 간단하고 과적합에 강하며 해석이 용이하지만, 부스팅 모델에 비해 최적의 성능은 아닐 수 있습니다.
    *   **XGBoost Regressor:** 뛰어난 예측 성능과 다양한 기능(병렬 처리, 결측값 처리)을 제공하지만, 하이퍼파라미터 튜닝이 복잡하고 과적합 가능성이 있습니다.
    *   **CatBoost Regressor:** 범주형 피처를 전처리 없이 직접 처리 가능하며, 과적합 방지 기능이 내장되어 높은 예측 성능을 제공합니다. 그러나 하이퍼파라미터 튜닝이 복잡하고 학습 시간이 길어질 수 있습니다.
    *   **예측 결과 차이:** CatBoost가 가장 높은 테스트 성능을 보였고, XGBoost가 그 뒤를 따랐으며, RandomForest가 가장 낮은 성능을 기록했습니다. 모든 부스팅 모델은 훈련 세트 R-squared가 1.0000에 가까웠으나, CatBoost가 테스트 세트에서 가장 안정적인 성능을 유지했습니다.
4.  **예측에 가장 중요한 피처들은 무엇이며, 모델마다 어떻게 다르게 나타납니까?**
    세 모델 모두 Lot 단위의 집계 피처(`Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count` 등)를 중요하게 평가했습니다.
    *   **RandomForestRegressor:** `Total_Count`와 `Slot_No_nunique`의 중요도가 압도적으로 높았습니다.
    *   **XGBoost Regressor:** `Total_Count`와 `Slot_No_nunique` 외에 `Killer_Defect_Proportion`, `DENSITY_SIGNAL_max`, `SIZE_X_min` 등 다양한 상세 집계 피처들이 상위 중요도에 포함되었습니다.
    *   **CatBoost Regressor:** `Total_Count`, `Killer_Defect_Count`, `Slot_No_nunique`와 더불어 `SIZE_X_mean_non_killer_mean`, `SNR_OFFSET_GL_killer_mean` 등 킬러/비킬러 결함 특성과 관련된 상세 통계 피처들을 중요하게 평가하여, 더 미묘한 관계를 포착하는 경향을 보였습니다.

### Data Analysis Key Findings

*   **CatBoost Regressor 성능 우위:** CatBoost Regressor는 테스트 세트 R-squared 0.9686, MSE 0.0002, RMSE 0.0137을 기록하며, RandomForestRegressor (R-squared 0.8910, MSE 0.0007, RMSE 0.0256) 및 XGBoost Regressor (R-squared 0.9334, MSE 0.0004, RMSE 0.0200) 대비 가장 우수한 예측 성능을 입증했습니다.
*   **모델별 성능 순위:** 예측 정확도 측면에서 CatBoost Regressor > XGBoost Regressor > RandomForestRegressor 순으로 나타났습니다.
*   **과적합 위험 평가:** CatBoost Regressor의 훈련 세트 R-squared는 1.0000이었으나, 테스트 세트 R-squared가 0.9686으로 높게 유지되어 과적합 위험이 낮고 일반화 성능이 뛰어남을 시사합니다. K-Fold 교차 검증의 평균 R-squared는 0.8911 (표준편차 0.1408)로 단일 테스트 결과와 비교적 유사하게 나타나 모델의 안정성을 지지합니다.
*   **주요 피처 일관성:** 세 모델 모두 `Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count`와 같은 Lot 단위의 집계 피처들을 예측에 가장 중요한 요소로 식별했습니다.

### Insights or Next Steps

*   Lot 단위의 Total_Risk_Score 예측에는 CatBoost Regressor가 현재까지 가장 최적의 모델로 판단됩니다. 해당 모델의 뛰어난 성능과 범주형 피처 처리 능력은 복잡한 Lot 데이터에 효과적입니다.
*   CatBoost Regressor가 중요하게 평가한 킬러/비킬러 결함 관련 상세 통계 피처(`SIZE_X_mean_non_killer_mean`, `SNR_OFFSET_GL_killer_mean` 등)에 대한 추가 분석을 통해 품질 위험 관리 전략을 더욱 정교화할 수 있습니다.


# Task
## 새로운 회귀 모델 선택 및 훈련 (LightGBM Regressor)

### Subtask:
LightGBM Regressor 모델을 선택하고, 이전 단계에서 준비된 Lot 단위 예측 데이터셋(`X_reg`, `y_reg`)으로 훈련 세트와 테스트 세트를 분할한 후 모델을 학습시킵니다. 이전에 사용된 RandomForestRegressor, XGBoost Regressor, CatBoost Regressor와 비교할 수 있도록 동일한 데이터 분할 방식을 사용합니다.

## 새로운 회귀 모델 선택 및 훈련 (LightGBM Regressor)

### Subtask:
LightGBM Regressor 모델을 선택하고, 이전 단계에서 준비된 Lot 단위 예측 데이터셋(`X_reg`, `y_reg`)으로 훈련 세트와 테스트 세트를 분할한 후 모델을 학습시킵니다. 이전에 사용된 RandomForestRegressor, XGBoost Regressor, CatBoost Regressor와 비교할 수 있도록 동일한 데이터 분할 방식을 사용합니다.


**Reasoning**:
The subtask requires selecting and training a LightGBM Regressor model. This involves importing necessary libraries, splitting the data into training and testing sets, and then initializing and training the LGBMRegressor model using the provided parameters.



import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Assuming X_reg and y_reg are available from the previous step

if 'X_reg' not in locals() or 'y_reg' not in locals():
    print("Error: Regression features (X_reg) or target (y_reg) not found. Please ensure the data preparation step has been executed.")
else:
    print("--- LightGBM Regressor 모델 훈련 시작 ---")

    # 1. 데이터 분할 (훈련 세트와 테스트 세트) - 이전 모델들과 동일한 방식 사용
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"훈련 세트 크기: {X_train_reg.shape[0]}")
    print(f"테스트 세트 크기: {X_test_reg.shape[0]}")

    # 2. LGBMRegressor 모델 초기화 및 훈련
    lgbm_reg_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1) # n_jobs=-1로 병렬 처리
    lgbm_reg_model.fit(X_train_reg, y_train_reg)

    print("\nLightGBM Regressor 모델 훈련 완료.")

**Reasoning**:
The LightGBM Regressor model has been trained. Now, I need to evaluate its performance using metrics such as MSE, RMSE, and R-squared, and visualize the actual vs. predicted values and feature importances. This will follow the pattern used for the RandomForestRegressor and XGBoost Regressor evaluation to ensure consistency for comparison.



from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold # Import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming lgbm_reg_model, X_train_reg, X_test_reg, y_train_reg, y_test_reg are available from the previous step

if 'lgbm_reg_model' not in locals() or 'X_test_reg' not in locals() or 'y_test_reg' not in locals():
    print("Error: Trained model or test data not found. Please ensure previous steps are executed.")
else:
    print("--- LightGBM Regressor 모델 훈련 및 과적합 검증 시작 ---")

    # Predict on the training and test data
    y_pred_train_lgbm = lgbm_reg_model.predict(X_train_reg)
    y_pred_test_lgbm = lgbm_reg_model.predict(X_test_reg)

    # Calculate and print evaluation metrics for test set
    mse_lgbm = mean_squared_error(y_test_reg, y_pred_test_lgbm)
    rmse_lgbm = np.sqrt(mse_lgbm)
    r2_test_lgbm = r2_score(y_test_reg, y_pred_test_lgbm)
    r2_train_lgbm = r2_score(y_train_reg, y_pred_train_lgbm)

    print(f"\n모델 평가 결과 (LightGBM Regressor - 단일 분할):")
    print(f"  훈련 세트 R-squared: {r2_train_lgbm:.4f}")
    print(f"  테스트 세트 R-squared: {r2_test_lgbm:.4f}")
    print(f"  테스트 세트 MSE: {mse_lgbm:.4f}")
    print(f"  테스트 세트 RMSE: {rmse_lgbm:.4f}")

    # Check for overfitting (1차 판단)
    if r2_train_lgbm > r2_test_lgbm and (r2_train_lgbm - r2_test_lgbm) > 0.1: # 예시 임계값, 상황에 따라 조절
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 커서 과적합 가능성이 있습니다.")
    elif r2_train_lgbm > r2_test_lgbm:
        print("  => 훈련 세트 R-squared가 테스트 세트보다 높지만, 큰 차이는 아니어서 일반적인 현상일 수 있습니다.")
    else:
        print("  => 훈련 세트와 테스트 세트의 R-squared 차이가 크지 않아 과적합 위험이 낮을 수 있습니다.")

    print("\n--- K-Fold 교차 검증 시작 (5-Fold) ---")
    # K-Fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores_lgbm = []
    cv_mse_scores_lgbm = []

    for train_index, val_index in kf.split(X_reg):
        X_train_cv, X_val_cv = X_reg.iloc[train_index], X_reg.iloc[val_index]
        y_train_cv, y_val_cv = y_reg.iloc[train_index], y_reg.iloc[val_index]

        model_cv_lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        model_cv_lgbm.fit(X_train_cv, y_train_cv)
        y_pred_cv_lgbm = model_cv_lgbm.predict(X_val_cv)

        cv_r2_scores_lgbm.append(r2_score(y_val_cv, y_pred_cv_lgbm))
        cv_mse_scores_lgbm.append(mean_squared_error(y_val_cv, y_pred_cv_lgbm))

    print(f"  교차 검증 R-squared 점수: {cv_r2_scores_lgbm}")
    print(f"  교차 검증 평균 R-squared: {np.mean(cv_r2_scores_lgbm):.4f} (표준편차: {np.std(cv_r2_scores_lgbm):.4f})")
    print(f"  교차 검증 평균 MSE: {np.mean(cv_mse_scores_lgbm):.4f} (표준편차: {np.std(cv_mse_scores_lgbm):.4f})")

    if np.mean(cv_r2_scores_lgbm) < (r2_test_lgbm - 0.1): # 교차 검증 결과가 단일 테스트보다 훨씬 낮으면 과적합 의심
        print("  => 교차 검증 결과가 단일 테스트 결과보다 크게 낮아 과적합 위험이 있을 수 있습니다.")
    else:
        print("  => 교차 검증 결과가 단일 테스트 결과와 유사하여 모델이 비교적 안정적입니다.")

    # Visualize actual vs. predicted values for the test set
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_reg, y=y_pred_test_lgbm, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2) # y=x line
    plt.title('LightGBM Regressor: 실제값 vs 예측값 (테스트 세트)')
    plt.xlabel('실제 Total_Risk_Score')
    plt.ylabel('예측 Total_Risk_Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Visualize feature importances
    if hasattr(lgbm_reg_model, 'feature_importances_'):
        feature_importances_lgbm = pd.Series(lgbm_reg_model.feature_importances_, index=X_reg.columns)
        top_n_features = 20 # Visualize top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances_lgbm.nlargest(top_n_features).values,
                    y=feature_importances_lgbm.nlargest(top_n_features).index,
                    palette='viridis')
        plt.title(f'LightGBM Regressor: 상위 {top_n_features}개 피처 중요도')
        plt.xlabel('중요도')
        plt.ylabel('피처')
        plt.tight_layout()
        plt.show()

    print("\nLightGBM Regressor 모델 훈련 및 과적합 검증 완료.")

## 네 모델 성능 비교 및 요약

### Subtask:
RandomForestRegressor, XGBoost Regressor, CatBoost Regressor, LightGBM Regressor 네 모델의 성능을 비교하고, 각 모델의 장단점 및 예측 결과의 차이점을 분석하여 요약합니다.

#### Instructions
1. RandomForestRegressor, XGBoost Regressor, CatBoost Regressor, LightGBM Regressor 각 모델의 테스트 세트 성능 지표(R-squared, MSE, RMSE)를 확인합니다.
2. 네 모델의 R-squared, MSE, RMSE 값을 비교하여 어떤 모델이 예측 성능이 가장 우수한지 명확히 밝힙니다.
3. 네 모델의 성능 차이점과 각 모델의 장단점(예: XGBoost의 높은 성능과 잠재적 복잡성, Random Forest의 일반적인 견고함과 해석 용이성, CatBoost의 범주형 피처 처리 능력, LightGBM의 빠른 속도와 효율성)을 분석하여 요약합니다.
4. 각 모델의 피처 중요도 시각화를 검토하여 예측에 가장 큰 영향을 미치는 피처들이 모델마다 어떻게 다르게 나타나는지 설명합니다.

### 모델 성능 비교:

**1. RandomForestRegressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.8910
*   **테스트 세트 MSE:** 0.0007
*   **테스트 세트 RMSE:** 0.0256
*   **교차 검증 평균 R-squared:** 0.8521

**2. XGBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9334
*   **테스트 세트 MSE:** 0.0004
*   **테스트 세트 RMSE:** 0.0200
*   **교차 검증 평균 R-squared:** 0.9022

**3. CatBoost Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9686
*   **테스트 세트 MSE:** 0.0002
*   **테스트 세트 RMSE:** 0.0137
*   **교차 검증 평균 R-squared:** 0.8911

**4. LightGBM Regressor 모델 성능:**
*   **테스트 세트 R-squared:** 0.9309
*   **테스트 세트 MSE:** 0.0004
*   **테스트 세트 RMSE:** 0.0203
*   **교차 검증 평균 R-squared:** 0.8818

### 분석 및 요약:

**성능 우수성:**
*   **CatBoost Regressor**가 네 모델 중 가장 우수한 성능을 보였습니다. 테스트 세트 R-squared 0.9686으로, `Total_Risk_Score` 변동의 약 97%를 설명합니다. MSE와 RMSE도 가장 낮아 예측 오차가 가장 작습니다.
*   **XGBoost Regressor**는 테스트 세트 R-squared 0.9334로 CatBoost 다음으로 우수하며, **LightGBM Regressor**는 테스트 세트 R-squared 0.9309로 XGBoost와 유사한 성능을 보였습니다.
*   **RandomForestRegressor**는 테스트 세트 R-squared 0.8910으로, 네 모델 중 가장 낮은 성능을 보였습니다.

**모델별 장단점 및 예측 결과 차이점:**

1.  **RandomForestRegressor:**
    *   **장점:** 구현이 비교적 간단하고, 과적합에 강하며, 해석이 용이한 편입니다. 강력하고 안정적인 성능을 제공합니다.
    *   **단점:** 부스팅 모델에 비해 최적의 예측 성능을 내지 못할 수 있습니다.
    *   **예측 결과:** 훈련 세트 R-squared(0.9764)와 테스트 세트 R-squared(0.8910) 간에 차이가 있으나, 교차 검증 평균 R-squared(0.8521)를 고려할 때 모델이 비교적 안정적입니다.

2.  **XGBoost Regressor:**
    *   **장점:** 뛰어난 예측 성능을 자랑하며, 다양한 데이터와 문제에 잘 적용됩니다. 병렬 처리가 가능하고, 결측값 처리, 정규화 등 고급 기능이 내장되어 있습니다.
    *   **단점:** 하이퍼파라미터 튜닝이 복잡하고, 과적합 가능성이 있어 주의가 필요합니다. 모델 해석이 Random Forest보다 어렵습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9334) 모두 매우 높게 나타났으며, 강력한 일반화 성능을 보여줍니다. 교차 검증 평균 R-squared(0.9022)도 단일 테스트 성능과 유사하게 안정적입니다.

3.  **CatBoost Regressor:**
    *   **장점:** 범주형 피처를 전처리 없이 직접 처리할 수 있는 강력한 기능이 있습니다. 기본적으로 과적합 방지 기능이 내장되어 있으며, 가장 높은 예측 성능을 제공합니다. 안정적이고 견고한 학습 과정을 가집니다.
    *   **단점:** 하이퍼파라미터 튜닝이 복잡할 수 있으며, 학습 시간이 상대적으로 길어질 수 있습니다. 모델 해석의 용이성은 낮습니다.
    *   **예측 결과:** 훈련 세트 R-squared(1.0000)와 테스트 세트 R-squared(0.9686) 모두 매우 높게 나타났으며, 특히 테스트 세트 성능이 가장 뛰어납니다. 훈련 세트에서의 완벽한 성능에도 불구하고 테스트 세트에서 강력한 성능을 유지하여 뛰어난 일반화 능력을 보여줍니다.

4.  **LightGBM Regressor:**
    *   **장점:** XGBoost와 유사하게 뛰어난 예측 성능을 제공하지만, 더 빠른 학습 속도와 낮은 메모리 사용량을 가집니다. 대규모 데이터셋에 효율적입니다.
    *   **단점:** 때때로 과적합에 취약할 수 있으며, XGBoost나 CatBoost보다 하이퍼파라미터 튜닝이 성능에 미치는 영향이 클 수 있습니다.
    *   **예측 결과:** 훈련 세트 R-squared(0.9644)와 테스트 세트 R-squared(0.9309)를 기록하며, XGBoost와 비슷한 수준의 강력한 성능을 보여주었습니다. 교차 검증 평균 R-squared(0.8818)는 단일 테스트 결과와 비교적 유사하여 안정적인 성능을 나타냅니다.

**피처 중요도 (상위 20개 비교):**

모든 모델이 `Total_Count`, `Slot_No_nunique`, `Killer_Defect_Count`와 같은 Lot 단위 집계 피처들을 중요한 피처로 간주하고 있습니다.

*   **RandomForestRegressor:** `Total_Count`의 중요도가 압도적으로 높고, `Slot_No_nunique`가 그 뒤를 잇습니다.
*   **XGBoost Regressor:** `Total_Count`와 `Slot_No_nunique`를 가장 중요한 피처로 평가하며, `Total_Count`가 가장 높은 중요도를 보입니다. `Killer_Defect_Proportion`, `DENSITY_SIGNAL_max`, `SIZE_X_min` 등 다양한 상세 집계 피처들도 중요합니다.
*   **CatBoost Regressor:** `Total_Count`, `Killer_Defect_Count`, `Slot_No_nunique`를 포함하여 `SIZE_X_mean_non_killer_mean`, `SNR_OFFSET_GL_killer_mean` 등 킬러/비킬러 결함 특성에 따른 상세 통계 피처들을 매우 중요하게 평가했습니다. 특히 `Killer_Defect_Count`와 `Slot_No_nunique_y`에 높은 중요도를 부여했습니다.
*   **LightGBM Regressor:** `Total_Count`와 `False_Defect_Count`가 가장 중요한 피처로 나타났습니다. `Killer_Defect_Proportion`과 `Killer_Defect_Count`도 높은 중요도를 보이며, `SIZE_X_mean_non_killer_mean`, `RELATIVEMAGNITUDE_non_killer_mean`과 같은 상세 피처들도 중요하게 평가되었습니다.

**결론적으로, CatBoost Regressor는 Lot 단위의 통합 위험도 점수 예측에서 가장 뛰어난 성능을 제공하며, 이는 복잡한 패턴과 다양한 유형의 피처를 학습하는 데 가장 적합한 모델임을 시사합니다. LightGBM과 XGBoost도 매우 우수한 성능을 보여주며, 빠른 학습 속도가 중요한 경우 LightGBM이 좋은 선택이 될 수 있습니다. 반면 RandomForest는 상대적으로 낮은 성능을 보였습니다. 따라서 이 예측 작업에는 CatBoost 모델이 가장 권장됩니다.**

## Final Task

### Subtask:
새로운 회귀 모델(LightGBM Regressor)의 성능과 이전 모델(RandomForestRegressor, XGBoost Regressor, CatBoost Regressor) 대비 개선 사항을 요약하고, 가장 중요한 피처들을 설명합니다.


## Summary:

### Q&A
The performance of the new LightGBM Regressor model and its improvement compared to previous models (RandomForestRegressor, XGBoost Regressor, CatBoost Regressor) for predicting `Total_Risk_Score` are as follows:

*   **LightGBM Regressor** achieved a test R-squared of 0.9309, placing it closely behind XGBoost Regressor and CatBoost Regressor. Its performance is a significant improvement over RandomForestRegressor.
*   **CatBoost Regressor** demonstrated the highest performance with a test R-squared of 0.9686, indicating it explains approximately 97% of the `Total_Risk_Score` variance, making it the most accurate model among those tested.
*   **XGBoost Regressor** also showed strong performance with a test R-squared of 0.9334, very similar to LightGBM.
*   **RandomForestRegressor** had the lowest performance with a test R-squared of 0.8910.

The most important features influencing the prediction of `Total_Risk_Score` consistently across the models are `Total_Count`, `Slot_No_nunique`, and `Killer_Defect_Count`. LightGBM specifically identified `Total_Count` and `False_Defect_Count` as its top features. CatBoost and XGBoost models also leveraged more detailed statistical features related to killer and non-killer defects.

### Data Analysis Key Findings
*   The LightGBM Regressor model achieved a test R-squared of 0.9309, a test Mean Squared Error (MSE) of 0.0004, and a test Root Mean Squared Error (RMSE) of 0.0203.
*   Cross-validation for LightGBM showed an average R-squared of 0.8818 (standard deviation: 0.0819), suggesting relatively stable performance and no severe overfitting based on the training R-squared of 0.9644.
*   Among the four models evaluated, CatBoost Regressor emerged as the top performer with the highest test R-squared of 0.9686, followed by XGBoost Regressor (0.9334), LightGBM Regressor (0.9309), and RandomForestRegressor (0.8910).
*   The models consistently identified `Total_Count`, `Slot_No_nunique`, and `Killer_Defect_Count` as key features for predicting `Total_Risk_Score`. LightGBM's top features were `Total_Count` and `False_Defect_Count`.
*   CatBoost Regressor's strength lies in its ability to handle categorical features directly and its built-in overfitting prevention mechanisms, contributing to its superior predictive power.

### Insights or Next Steps
*   Given its superior performance, the **CatBoost Regressor** is the recommended model for predicting `Total_Risk_Score` in Lot units due to its high accuracy and robustness.
*   Further hyperparameter tuning for CatBoost, LightGBM, and XGBoost could potentially yield marginal improvements in model performance and generalize capabilities, especially for complex real-world data.
