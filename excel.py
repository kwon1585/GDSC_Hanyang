import pandas as pd

# 엑셀 파일 경로
excel_path = './data_info.xlsx'

# 엑셀 파일 읽기
df = pd.read_excel(excel_path)

# 조건에 따라 값을 설정
for index, row in df.iterrows():
    cell_value = str(row['code'])  

    if '연질캡슐' in cell_value :
        df.at[index, 'formulation'] = 0
    elif '캡슐' in cell_value:
        df.at[index, 'formulation'] = 1
    elif '나정' in cell_value or '필름' in cell_value or '붕해' in cell_value:
        df.at[index, 'formulation'] = 2
    else : 
        df.at[index, 'formulation'] = 3

# 새로운 값을 추가한 DataFrame을 엑셀 파일로 저장
df.to_excel('data_info_final.xlsx', index=False)
