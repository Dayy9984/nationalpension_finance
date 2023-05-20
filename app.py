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
   
df_2021_price = pd.DataFrame()
df_2020_price = pd.DataFrame()
p = [59,84]
header = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}

for name in df_national_pension:
    url = get_url(name, df_code)

    
    # 일자 데이터를 담을 df라는 DataFrame 정의
    df_price = pd.DataFrame() 
    
    # 1페이지에서 100페이지의 데이터만 가져오기
    for page in p:
        pg_url = '{url}&page={page}'.format(url=url, page=page)
        res = requests.get(pg_url, headers=header)
        df_price = pd.concat([df_price,pd.read_html(res.text, header=0,encoding='euc-kr')[0]], ignore_index=True)
    
    
    # df.dropna()를 이용해 결측값 있는 행 제거
    df_price = df_price.dropna()
    
    data = df_price[df_price['날짜'] == '2021.01.04']
    data.insert(0,'name',name)
    df_2021_price = pd.concat([df_2021_price,data[['name','종가']]],ignore_index=True)
    
    data = df_price[df_price['날짜'] == '2020.01.02']
    data.insert(0,'name',name)
    df_2020_price = pd.concat([df_2020_price,data[['name','종가']]],ignore_index=True)
df_2021_price = df_2021_price.drop(df_2021_price.index[df_2021_price['name']=='SK바이오팜'],axis=0)
df_2021_price = df_2021_price.reset_index(drop=True)

#국민연금 종목 수익/손실 그래프
df_result = pd.DataFrame()
df_result['result'] = (df_2021_price['종가'] > df_2020_price['종가']).astype(int)
result_counts = df_result['result'].value_counts()
# plt.bar(result_counts.index, result_counts.values,color=['red', 'dodgerblue'])
# plt.ylabel('Count')
# plt.xticks([0, 1], ['손실', '수익'])
# plt.title('국민연금 종목 수익/손실 그래프')
# plt.show()

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

#코스피 그래프
# plt.figure(figsize=(10,4))
# plt.plot(df_kospi_price['날짜'], df_kospi_price['체결가'])
# plt.xlabel('날짜')
# plt.ylabel('종가')
# plt.tick_params(
#     axis='x',
#     which='both',
#     bottom=False,
#     top=False,
#     labelbottom=False)
# plt.savefig("kospi.png")
# plt.show()

#시가총액 크롤링
d = [20210104,20200102]
df_market_cap = pd.DataFrame()
for date in d:
    gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
    gen_otp_data = {
      'mktId': 'STK',
      'trdDd': f'{date}',
      'money': '1',
      'csvxls_isNo': 'false',
      'name': 'fileDown',
      'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
    }
    headers = {'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader'}
    otp = requests.post(gen_otp_url, gen_otp_data, headers=headers).text
    down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
    down_sector_KS  = requests.post(down_url, {'code':otp}, headers=headers)
    data = pd.read_csv(BytesIO(down_sector_KS.content), encoding='EUC-KR')
    data['date'] = date
    df_market_cap = pd.concat([df_market_cap,data],ignore_index=True)
df_market_cap_2020 = df_market_cap.query('종목명 in {} and date == 20200102'.format(list(df_2020_price['name'])))
df_market_cap_2021 = df_market_cap.query('종목명 in {} and date == 20210104'.format(list(df_2021_price['name'])))
df_2020_sum = df_market_cap_2020['시가총액'].sum()
df_2021_sum = df_market_cap_2021['시가총액'].sum()
#국민연금 코스피대비 수익률
national_pension_20_21 = ((df_2021_sum/df_2020_sum)*100)-100
kospi_20_21 = ((df_kospi_2021['체결가'].values-df_kospi_2020['체결가'].values)/df_kospi_2020['체결가'].values*100)[0]

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
