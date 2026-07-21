import csv
from pathlib import Path
import pandas as pd

src_dir = Path(__file__).resolve().parents[1]
project_dir = src_dir.parent
csv_path = project_dir / 'data' / 'raw' / 'employee_birthday.csv'

with open(csv_path, mode="r", encoding="utf-8", newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {",".join(row)}')
            line_count += 1
        else:
            print(f'{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
    print(f'Processed {line_count - 1} lines.')

df = pd.read_csv(project_dir/'data'/'raw'/'hrdata.csv', names=['Namae', 'Hiduke', 'Okane', 'Yasumi'], index_col='Namae', parse_dates=['Hiduke'], date_format='%m/%d/%y', header=0)
print(df)

# print(type(df['Hire Date'].iloc[0]))