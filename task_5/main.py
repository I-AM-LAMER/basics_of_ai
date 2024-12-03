import pandas as pd
import numpy as np
import openpyxl
import os
import ydata_profiling
import sqlite3

# import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px


file_path = '/home/ilya/Downloads/sales_df.xlsx'

if os.path.exists(file_path):
    print('Файл найден')
else:
    print('Файл не найден')

sales_df = pd.read_excel('/home/ilya/Downloads/sales_df.xlsx', header=2)
# sales_df.head()

salez_size = sales_df.shape
print('Количество строк и столбцов: ', salez_size)
len_sales = len(sales_df)
print('Количество строк: ', len_sales)
column_sales = len(sales_df.columns)
print('Количество столбцов: ', column_sales)
head = sales_df.columns
print('Исходные названия колонок: ', head)
sales_df.columns = head.str.lower().str.replace(' ', '_')
print('Новые названия колонок: ', sales_df.columns)
print("Типы данных в каждом столбце:\n", sales_df.dtypes)

# sales_df.profile_report()

# profile = ydata_profiling.ProfileReport(sales_df, title="Profiling Report")
# profile.to_file("task_5/your_report.html")
# profile.to_notebook_iframe()

sales_df.isnull().sum()

#смотрим пустые значения в данных
sales_df[sales_df.isna().any(axis=1)]

# ~ - оператор НЕ. Мы получаем все ероме тех значений, что в []
sales_df = sales_df[~sales_df.isna().any(axis=1)]
sales_size = sales_df.shape
print('Количество строк и столбцов после удаления пустых строк:', sales_size)

# ищем дубликаты 
sales_df[sales_df.duplicated()].sort_values('traransaction_id')

#Удаляем дубликаты
sales_df.drop_duplicates(inplace=True)
print('Количество строк и столбцов: ', sales_df.shape)
sales_df[sales_df.duplicated()].sort_values('traransaction_id')

# Проверка дубликатов
len(sales_df[sales_df['traransaction_id'].duplicated()])

# смотрим в данных. Вдруг придется делать новый ключ
sales_df[sales_df['traransaction_id'].duplicated()]
sales_df[sales_df['traransaction_id'] == 759]

# Количество неуникальных значений (повторяющихся)
sales_df.nunique()
# Типы данных
sales_df.info()

column_to_convert = ['traransaction_id', 'customer_id', 'line_item_id', 'product_id']
sales_df[column_to_convert].astype(int)

# переводим дату в правильный формат данных
sales_df.transaction_date.apply(pd.to_datetime)

# смотрим заголовки
sales_df.head()

# Меняем Y и N на Yes и No
sales_df['instore_yn'].replace({'Y': 'Yes', 'N': 'No'})
# Приводим все значение в нижний регистр
sales_df['sales_outlet_type'].str.lower()

# Фильтруем данные датафрейма и оставляем транзакции с продует_ид из (23, 25, 27, 32, 34, 35, 49)
product_id_trans = [23, 25, 27, 32, 34, 35, 49]
filter_sales_df = sales_df[sales_df['product_id'].isin(product_id_trans)]
print('Количество строк после фильтрации: ', len(filter_sales_df))

#Фильтрация данных по указанным столбцам
columns_to_keep = [
    'traransaction_id', 'customer_id', 'quantity', 'product_id',
    'transaction_date', 'transaction_time', 'unit_price', 'store_address', 'manager'
    ]
filtered_sales_df = sales_df.loc[:, columns_to_keep]
print(filtered_sales_df.head())
print('Размер нового DataFrame: ', filtered_sales_df.shape)



product_sales = sales_df.groupby('product_id').size()



# plt.figure(figsize=(10, 6))
# product_sales.plot(kind='bar')
# plt.title('Количество проданных товаров')
# plt.xlabel('ID товара')
# plt.ylabel('Количество продаж')
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()



fig = go.Figure()

fig.add_trace(go.Bar(
    x=product_sales.index,
    y=product_sales.values,
    text=product_sales.values,
    hoverinfo='x+y+text'
))

fig.update_layout(
    title='Количество проданных товаров по категориям',
    xaxis_title='ID товара',
    yaxis_title='Количество продаж',
    showlegend=False
)

fig.show()



daily_sales = sales_df.groupby('transaction_date')[['quantity', 'unit_price']].apply(lambda x: (x['quantity'] * x['unit_price']).sum())



# plt.figure(figsize=[10, 6])
# daily_sales.plot()
# plt.title('Сумма проодаж по дням')
# plt.xlabel('Дата')
# plt.ylabel('Сумма продаж')
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()



if isinstance(daily_sales, pd.Series):
    daily_sales = daily_sales.to_frame(name='total_amount')

fig = px.line(daily_sales, 
    x=daily_sales.index, 
    y='total_amount', 
    title='Сумма продаж по дням'
)

fig.show()



# sales_df.profile_report()

# output_file_path = 'task_5/transformed_sales_data.xlsx'
# daily_sales.to_excel(output_file_path)