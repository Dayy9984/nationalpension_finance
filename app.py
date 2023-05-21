import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import streamlit as st
from io import BytesIO

font_location = 'NanumBarunGothicLight.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name)

df_2021 = pd.read_csv('2021.csv')
df_2020 = pd.read_csv('2020.csv')

df_national_pension = list(sorted(set(df_2021['종목명']).intersection(set(df_2020['종목명']))))

df_krx = pd.read_csv('code.csv')
df_krx = df_krx[['한글 종목약명', '단축코드']]
df_krx.rename(columns={'한글 종목약명':'name','단축코드':'code'},inplace=True)
df_krx['code'] = df_krx['code'].apply(lambda x : '{0:0>6}'.format(x))
df_krx = pd.DataFrame(df_krx)
df_code = df_krx.query(f"name in {df_national_pension}")
df_code = pd.DataFrame(df_code)
df_code = df_code.reset_index(drop=True)


def get_url(item_name, df_code):
    code = df_code.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
    return url
   
header = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}


#코스피 데이터 크롤링
df_kospi_price = pd.DataFrame()
for page in range(99,141):
    pg_url = 'https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI&page={page}'.format(page=page)
    res = requests.get(pg_url, headers=header)
    df_kospi_price = pd.concat([df_kospi_price,pd.read_html(res.text, header=0,encoding='euc-kr')[0]], ignore_index=True)
    df_kospi_price = df_kospi_price.dropna()
for i in range(1,31):
    df_kospi_price.drop(df_kospi_price[df_kospi_price['날짜'] == "2019.12.{0:0>2}".format(i)].index , inplace=True)
for i in range(5,31):
    df_kospi_price.drop(df_kospi_price[df_kospi_price['날짜'] == '2021.01.{0:0>2}'.format(i)].index , inplace=True)
df_kospi_2021 = df_kospi_price[df_kospi_price['날짜'] == '2021.01.04']
df_kospi_2020 = df_kospi_price[df_kospi_price['날짜'] == '2020.01.02']
df_kospi_price = df_kospi_price.sort_index(ascending=False)
df_kospi_price = df_kospi_price.reset_index(drop=True)


df_2021_price_item = pd.DataFrame()
df_2020_price_item = pd.DataFrame()
name = st.selectbox('종목선택',list(df_code['name']))
url = get_url(name, df_code)
df_price_item = pd.DataFrame() 
for page in range(59,85):
    pg_url = '{url}&page={page}'.format(url=url, page=page)
    res = requests.get(pg_url, headers=header)
    df_price_item = pd.concat([df_price_item,pd.read_html(res.text, header=0,encoding='euc-kr')[0]], ignore_index=True)
df_price_item = df_price_item.dropna()
for i in range(1,31):
    df_price_item.drop(df_price_item[df_price_item['날짜'] == "2019.12.{0:0>2}".format(i)].index , inplace=True)
for i in range(5,31):
    df_price_item.drop(df_price_item[df_price_item['날짜'] == '2021.01.{0:0>2}'.format(i)].index , inplace=True)
df_price_item = df_price_item.sort_index(ascending=False)
df_price_item = df_price_item.reset_index(drop=True)

df_kospi_price['체결가_normalization'] = df_kospi_price['체결가']/abs(df_kospi_price['체결가'].max())
df_price_item['종가_normalization'] = df_price_item['종가']/abs(df_price_item['종가'].max())

plt.figure(figsize=(10,4))
plt.plot(df_kospi_price['날짜'], df_kospi_price['체결가_normalization'],color='dodgerblue')
plt.xlabel('날짜')
plt.ylabel('종가')
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.plot(df_price_item['날짜'], df_price_item['종가_normalization'],color='orange')
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
variable_x = mpatches.Patch(color='dodgerblue',label='KOSPI')
variable_y = mpatches.Patch(color='orange',label=name)
plt.legend(handles=[variable_x, variable_y],loc='lower left')
plt.title(f'KOSPI/{name} 그래프')
st.pyplot(plt)
