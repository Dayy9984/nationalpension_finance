import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import time
import concurrent.futures
import streamlit as st
from mpl_finance import candlestick2_ohlc

st.title('　　　　:blue[국민연금] ')
st.title('2020/2021년도 투자종목 분석:page_with_curl:')
st.title('')
font_path = 'NanumBarunGothicLight.ttf'
font_prop = fm.FontProperties(fname=font_path, size= 16)

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
df_code = df_code.drop(df_code.index[df_code['name']=='SK바이오팜'],axis=0)
df_code = df_code.reset_index(drop=True)



def get_url(item_name, df_code):
    code = df_code.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
    return url
   

header = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}

# Create a Session instance to keep connections open between requests
session = requests.Session()
session.headers = header

option = st.selectbox(
    '국민연금 투자종목 분석 보고서 선택',
    ('국민연금 투자종목 손실/수익 분석', '국민연금 투자종목 차트분석'))
if option == '국민연금 투자종목 손실/수익 분석':
    st.subheader('국민연금 투자종목 :blue[손실]/:red[수익] 분석:pencil:')
    p = [59,84]

    def fetch_item(page, url):
        pg_url = f'{url}&page={page}'
        res = session.get(pg_url)
        df_page = pd.read_html(res.text, header=0,encoding='euc-kr')[0]
        return df_page

    with st.spinner('국민연금 투자종목 손실/수익 분석중...'):
        def price(name):
            url = get_url(name, df_code)
            # 일자 데이터를 담을 df라는 DataFrame 정의
            df_price = pd.DataFrame() 

            # 1페이지에서 100페이지의 데이터만 가져오기
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # fetch all pages concurrently
                futures = [executor.submit(fetch_item, page,url) for page in p]
                # as the results come in, append them to the DataFrame
                for future in concurrent.futures.as_completed(futures):
                    df_page = future.result()
                    df_price = pd.concat([df_price, df_page[df_page['날짜'] == '2021.01.04']], ignore_index=True)
                    df_price = pd.concat([df_price, df_page[df_page['날짜'] == '2020.01.02']], ignore_index=True)
            df_price.insert(0,'name',name)
            return df_price

        df_price = pd.DataFrame()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # fetch all pages concurrently
            futures = [executor.submit(price, name) for name in df_national_pension]
            # as the results come in, append them to the DataFrame
            for future in concurrent.futures.as_completed(futures):
                df_page = future.result()
                df_price = pd.concat([df_price, df_page], ignore_index=True)

        df_2021_price = df_price[df_price['날짜'] == '2021.01.04']
        df_2020_price = df_price[df_price['날짜'] == '2020.01.02']
        df_2021_price = df_2021_price.reset_index(drop=True)
        df_2020_price = df_2020_price.reset_index(drop=True)
    with st.spinner('국민연금 투자종목 손실/수익 그래프 생성중...'):
        #국민연금 종목 수익/손실 그래프
        df_2021_price['result'] = (df_2021_price['종가'] > df_2020_price['종가']).astype(int)
        result_counts = df_2021_price['result'].value_counts()
        plt.bar(result_counts.index, result_counts.values,color=['red', 'dodgerblue'])
        plt.ylabel('Count')
        plt.xticks([0, 1], ['손실', '수익'],fontproperties=font_prop,size=10)
        plt.title('국민연금 종목 수익/손실 그래프',fontproperties=font_prop)
        st.pyplot(plt)   
    st.title('')
elif option == '국민연금 투자종목 차트분석':
    with st.spinner('국민연금 투자 종목 데이터 분석중...'):
        # This function will be run concurrently
        def fetch_kospi(page):
            pg_url = f'https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI&page={page}'
            res = session.get(pg_url)
            df_page = pd.read_html(res.text, header=0,encoding='euc-kr')[0]
            return df_page

        def fetch_item(page, url):
            pg_url = f'{url}&page={page}'
            res = session.get(pg_url)
            df_page = pd.read_html(res.text, header=0,encoding='euc-kr')[0]
            return df_page

        # Use a ThreadPoolExecutor to run the fetch function concurrently
        df_kospi_price = pd.DataFrame()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # fetch all pages concurrently
            futures = [executor.submit(fetch_kospi, page) for page in range(99,141)]
            # as the results come in, append them to the DataFrame
            for future in concurrent.futures.as_completed(futures):
                df_page = future.result()
                df_kospi_price = pd.concat([df_kospi_price, df_page], ignore_index=True)

        df_kospi_price = df_kospi_price.dropna()

        for i in range(1,31):
            df_kospi_price.drop(df_kospi_price[df_kospi_price['날짜'] == "2019.12.{0:0>2}".format(i)].index , inplace=True)
        for i in range(5,31):
            df_kospi_price.drop(df_kospi_price[df_kospi_price['날짜'] == '2021.01.{0:0>2}'.format(i)].index , inplace=True)

        df_kospi_2021 = df_kospi_price[df_kospi_price['날짜'] == '2021.01.04']
        df_kospi_2020 = df_kospi_price[df_kospi_price['날짜'] == '2020.01.02']
        df_kospi_price = df_kospi_price.sort_index(ascending=False)
        df_kospi_price = df_kospi_price.reset_index(drop=True)



    st.subheader('국민연금 투자종목 :blue[차트]분석:chart_with_upwards_trend:')
     
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox('종목선택',list(df_code['name']))  
    with col2:
        rangestandard = st.radio(
                "종가범위 방식 지정",
                ('선형스케일링', '정규화'))
    candle = st.checkbox('캔들로 전환')
    with st.spinner('국민연금 투자 종목 그래프 생성중...'):
        url = get_url(name, df_code)
        df_price_item = pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # fetch all pages concurrently
            futures = [executor.submit(fetch_item, page, url) for page in range(59,85)]
            # as the results come in, append them to the DataFrame
            for future in concurrent.futures.as_completed(futures):
                df_page = future.result()
                df_price_item = pd.concat([df_price_item, df_page], ignore_index=True)

        df_price_item = df_price_item.dropna()

        for i in range(1,31):
            df_price_item.drop(df_price_item[df_price_item['날짜'] == "2019.12.{0:0>2}".format(i)].index , inplace=True)
        for i in range(5,31):
            df_price_item.drop(df_price_item[df_price_item['날짜'] == '2021.01.{0:0>2}'.format(i)].index , inplace=True)

        df_price_item = df_price_item.sort_index(ascending=False)
        df_price_item = df_price_item.reset_index(drop=True)
        df_kospi_price = df_kospi_price.sort_values('날짜')
        df_price_item = df_price_item.sort_values('날짜')

        if rangestandard == '선형스케일링':
            kospi_range = df_kospi_price['체결가'].max() - df_kospi_price['체결가'].min()
            item_range = df_price_item['종가'].max() - df_price_item['종가'].min()
            df_kospi_price['체결가_normalization'] = (((df_kospi_price['체결가']-df_kospi_price['체결가'].min())*item_range)/kospi_range) + df_price_item['종가'].min()
        else:
            df_kospi_price['price_normalization'] = df_kospi_price['체결가']/abs(df_kospi_price['체결가'].max())
            df_price_item['price_normalization'] = df_price_item['종가']/abs(df_price_item['종가'].max())
            df_price_item['종가_normalization'] = (df_price_item['종가'] - df_price_item['종가'].mean())/df_price_item['종가'].std()
            df_price_item['시가_normalization'] = (df_price_item['시가'] - df_price_item['시가'].mean())/df_price_item['시가'].std()
            df_price_item['고가_normalization'] = (df_price_item['고가'] - df_price_item['고가'].mean())/df_price_item['고가'].std()
            df_price_item['저가_normalization'] = (df_price_item['저가'] - df_price_item['저가'].mean())/df_price_item['저가'].std()
            df_kospi_price['체결가_normalization_else'] = (df_kospi_price['체결가'] - df_kospi_price['체결가'].mean())/df_kospi_price['체결가'].std()

        if not candle:
            plt.figure(figsize=(16,9))
            if rangestandard == '선형스케일링':
                plt.plot(df_kospi_price['날짜'], df_kospi_price['체결가_normalization'], color='dodgerblue')
                plt.plot(df_price_item['날짜'], df_price_item['종가'], color='orange')
                plt.ylabel('종가',fontproperties=font_prop)
            else:
                plt.plot(df_kospi_price['날짜'], df_kospi_price['price_normalization'], color='dodgerblue')
                plt.plot(df_price_item['날짜'], df_price_item['price_normalization'], color='orange')
                plt.ylabel('종가 (Min-Max 정규화)',fontproperties=font_prop)
            plt.xlabel('날짜',fontproperties=font_prop)
            plt. tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            plt. tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            variable_x = mpatches.Patch(color='dodgerblue',label='KOSPI')
            variable_y = mpatches.Patch(color='orange',label=name)
            plt.legend(handles=[variable_x, variable_y],prop=font_prop)
            plt.title(f'KOSPI/{name} 그래프',fontproperties=font_prop,size=28)
            st.pyplot(plt)
        else:
            fig , ax = plt.subplots(figsize=(16,9))
            if rangestandard == '선형스케일링':
                plt.plot(df_kospi_price['날짜'], df_kospi_price['체결가_normalization'], color='dodgerblue',linewidth=0.7)
                plt.ylabel('종가',fontproperties=font_prop)
                candlestick2_ohlc(ax, df_price_item['시가'], df_price_item['고가'], 
                          df_price_item['저가'], df_price_item['종가'],
                          width=0.5, colorup='r', colordown='b')
            else:
                plt.plot(df_kospi_price['날짜'], df_kospi_price['체결가_normalization_else'], color='dodgerblue',linewidth=0.7)
                plt.ylabel('종가 (Standard 정규화)',fontproperties=font_prop)
                candlestick2_ohlc(ax, df_price_item['시가_normalization'], df_price_item['고가_normalization'], 
                          df_price_item['저가_normalization'], df_price_item['종가_normalization'],
                          width=0.5, colorup='r', colordown='b')
            plt.xlabel('날짜',fontproperties=font_prop)
            plt. tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            variable_x = mpatches.Patch(color='dodgerblue',label='KOSPI')
            plt.legend(handles=[variable_x],prop=font_prop)
            plt.title(f'KOSPI/{name} 그래프',fontproperties=font_prop,size=28)
            st.pyplot(plt)

