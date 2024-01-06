
import streamlit as st
import pandas_ta as ta
import numpy as np
import yfinance as yf
import pandas as pd
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dropout, Dense
from tensorflow.keras.layers import LSTM
import datetime


st.title('Amol Stock Recommedations')


if st.button('Run Code'):

    current_date = datetime.date.today()
    
    import warnings
    
    # Ignore the SettingWithCopyWarning
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
    
    
    # This will ignore all FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    
    
    overall_indicators_list=[]
    def generate_signal(ticker_list,start_date,end_date):
        dict={}
        for ticker in ticker_list:
    
            try:
    
                df=yf.download(ticker,start=start_date,end=end_date)
    
    
                #calculate values
     
                #EMA(5)
                df['ema_5']=ta.ema(df['Close'],length=5)
                
                
                #SMA(5)
                
                df['sma_5']=ta.sma(df['Close'],length=5)
                
                #EMA(10)
                df['ema_10']=ta.ema(df['Close'],length=10)
                
                #SMA(10)
                df['sma_10']=ta.sma(df['Close'],length=10)
                
                #EMA(20)
                df['ema_20']=ta.ema(df['Close'],length=20)
                
                #SMA(20)
                df['sma_20']=ta.sma(df['Close'],length=20)
                
                #EMA(30)
                df['ema_30']=ta.ema(df['Close'],length=30)
                
                #SMA(30)
                df['sma_30']=ta.sma(df['Close'],length=30)
                
                
                #EMA(50)
                df['ema_50']=ta.ema(df['Close'],length=50)
                
                #SMA(50)
                df['sma_50']=ta.sma(df['Close'],length=50)
                
                
                #EMA(100)
                df['ema_100']=ta.ema(df['Close'],length=100)
                
                #SMA(100)
                df['sma_100']=ta.sma(df['Close'],length=100)
                
                
                #EMA(200)
                df['ema_200']=ta.ema(df['Close'],length=200)
                
                #SMA(200)
                df['sma_200']=ta.sma(df['Close'],length=200)
                
                
                #WVMA
                
                df['wvma'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
                
                #HMA(9)
                
                df['hma'] = ta.hma(df['Close'], length=9)
                
                
                
                            
                            
                            
                            
                            
                            
                            
                            
                #RSI
                rsi_period = 14  # Number of periods to consider for RSI calculation
                df['rsi'] = ta.rsi(df['Close'], length=rsi_period)
                
                
                #STOCHASTIC %K
                # Define periods
                k_period = 14
                d_period = 3
                
                df['n_high'] = df['High'].rolling(k_period).max()
                df['n_low'] = df['Low'].rolling(k_period).min()
                df['%K'] = (df['Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
                df['%D'] = df['%K'].rolling(d_period).mean()
                
                
                
                
                
                #Commodity Channel Index (20)
                df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
                adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
                df['ADX'] = adx['ADX_14'] 
                df['DMP_14']=adx['DMP_14']
                df['DMN_14']=adx['DMN_14']
                
                
                # Awesome Oscillator
                df['AO'] = ta.ao(df['High'], df['Low'])
                
                
                # momentum(10)
                df['Momentum'] = ta.mom(df['Close'], length=10)
                
                # MACD Level (12, 26)
                macd = ta.macd(df['Close'])
                df['MACD_Line'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                df['MACD_Hist']=macd['MACDh_12_26_9']
                df['MACD_Level'] = df['MACD_Line'] - df['MACD_Signal']
                
                
                # Stochastic RSI Fast (3, 3, 14, 14)
                # a = ta.stochrsi(df['rsi'])
                
                a = ta.stochrsi(df['Close'])
                df['fastk']=a['STOCHRSIk_14_14_3_3']
                df['fastd']=a['STOCHRSId_14_14_3_3']
                
                
                
                
                
                # Williams Percent Range (14)
                df['WPR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
                
                
                # ultimate osc
                df['bp'] = df["Close"] - np.minimum(df["Low"], df["Close"].shift())
                df['tr'] = np.maximum(df["High"], df["Close"].shift()) - np.minimum(df["Low"], df["Close"].shift())
                df['average7'] = ta.sma(df['bp'], length=7) / ta.sma(df['tr'], length=7)
                df['average14'] = ta.sma(df['bp'], length=14) / ta.sma(df['tr'], length=14)
                df['average28'] = ta.sma(df['bp'], length=28) / ta.sma(df['tr'], length=28)
                df['uo']= 100 * ((4 * df['average7']) + (2 * df['average14']) + df['average28']) / (4 + 2 + 1)
    
                
                
                ema_5_signal=[]
                sma_5_signal=[]
                ema_10_signal=[]
                sma_10_signal=[]
                ema_20_signal=[]
                sma_20_signal=[]
                ema_30_signal=[]
                sma_30_signal=[]
                ema_50_signal=[]
                sma_50_signal=[]
                ema_100_signal=[]
                sma_100_signal=[]
                ema_200_signal=[]
                sma_200_signal=[]
                wvma_signal=[]
                hma_signal=[]
                
                
                RSI_signal = []
                STOCH_K_signal = []
                CCI_signal = []
                ADX_signal = []
                AO_signal = []
                Momentum_signal= []
                MACD_signal = []
                stochfastk_signal=[]
                WPR_signal = []
                Bear_Bull_Power_signal = []
                uo_signal = []
                
                
                for i in range(len(df)):
                    #ema_5_signal
                    if df['Close'][i]>df['ema_5'][i]:
                        ema_5_signal.append("Buy")
                    elif df['Close'][i]<df['ema_5'][i]:
                        ema_5_signal.append('Sell')
                    else:
                        ema_5_signal.append('Neutral')
                
                    
                       # sma_5_signal
                    if df['Close'][i]>df['sma_5'][i]:
                        sma_5_signal.append("Buy")
                    elif df['Close'][i]<df['sma_5'][i]:
                        sma_5_signal.append('Sell')
                    else:
                        sma_5_signal.append('Neutral')
                
                
                
                    # ema_10_signal
                    if df['Close'][i]>df['ema_10'][i]:
                        ema_10_signal.append("Buy")
                    elif df['Close'][i]<df['ema_10'][i]:
                        ema_10_signal.append('Sell')
                    else:
                        ema_10_signal.append('Neutral')
                
                    
                
                    
                    # sma_10_signal
                    if df['Close'][i]>df['sma_10'][i]:
                        sma_10_signal.append("Buy")
                    elif df['Close'][i]<df['sma_10'][i]:
                        sma_10_signal.append('Sell')
                    else:
                        sma_10_signal.append('Neutral')
                
                
                
                    # ema_20_signal
                    if df['Close'][i]>df['ema_20'][i]:
                        ema_20_signal.append("Buy")
                    elif df['Close'][i]<df['ema_20'][i]:
                        ema_20_signal.append('Sell')
                    else:
                        ema_20_signal.append('Neutral')
                
                    
                
                    
                    # sma_20_signal
                    if df['Close'][i]>df['sma_20'][i]:
                        sma_20_signal.append("Buy")
                    elif df['Close'][i]<df['sma_20'][i]:
                        sma_20_signal.append('Sell')
                    else:
                        sma_20_signal.append('Neutral')
                
                
                
                
                 # ema_30_signal
                    if df['Close'][i]>df['ema_30'][i]:
                        ema_30_signal.append("Buy")
                    elif df['Close'][i]<df['ema_30'][i]:
                        ema_30_signal.append('Sell')
                    else:
                        ema_30_signal.append('Neutral')
                
                    
                
                    
                    # sma_30_signal
                    if df['Close'][i]>df['sma_30'][i]:
                        sma_30_signal.append("Buy")
                    elif df['Close'][i]<df['sma_30'][i]:
                        sma_30_signal.append('Sell')
                    else:
                        sma_30_signal.append('Neutral')
                
                
                
                
                 # ema_50_signal
                    if df['Close'][i]>df['ema_50'][i]:
                        ema_50_signal.append("Buy")
                    elif df['Close'][i]<df['ema_50'][i]:
                        ema_50_signal.append('Sell')
                    else:
                        ema_50_signal.append('Neutral')
                
                    
                
                    
                    # sma_50_signal
                    if df['Close'][i]>df['sma_50'][i]:
                        sma_50_signal.append("Buy")
                    elif df['Close'][i]<df['sma_50'][i]:
                        sma_50_signal.append('Sell')
                    else:
                        sma_50_signal.append('Neutral')
                
                
                # ema_100_signal
                    if df['Close'][i]>df['ema_100'][i]:
                        ema_100_signal.append("Buy")
                    elif df['Close'][i]<df['ema_100'][i]:
                        ema_100_signal.append('Sell')
                    else:
                        ema_100_signal.append('Neutral')
                
                    
                
                    
                    # sma_100_signal
                    if df['Close'][i]>df['sma_100'][i]:
                        sma_100_signal.append("Buy")
                    elif df['Close'][i]<df['sma_100'][i]:
                        sma_100_signal.append('Sell')
                    else:
                        sma_100_signal.append('Neutral')
                
                
                
                # ema_200_signal
                    if df['Close'][i]>df['ema_200'][i]:
                        ema_200_signal.append("Buy")
                    elif df['Close'][i]<df['ema_200'][i]:
                        ema_200_signal.append('Sell')
                    else:
                        ema_200_signal.append('Neutral')
                
                    
                
                    
                    # sma_200_signal
                    if df['Close'][i]>df['sma_200'][i]:
                        sma_200_signal.append("Buy")
                    elif df['Close'][i]<df['sma_200'][i]:
                        sma_200_signal.append('Sell')
                    else:
                        sma_200_signal.append('Neutral')
                
                
                  
                    # wvma_signal
                    if df['Close'][i]>df['wvma'][i]:
                        wvma_signal.append("Buy")
                    elif df['Close'][i]<df['wvma'][i]:
                        wvma_signal.append('Sell')
                    else:
                        wvma_signal.append('Neutral')
                
                    
                  # hma_signal
                    if df['Close'][i]>df['hma'][i]:
                        hma_signal.append("Buy")
                    elif df['Close'][i]<df['hma'][i]:
                        hma_signal.append('Sell')
                    else:
                        hma_signal.append('Neutral')
                            
                            
                
                
                
                    
                
                     # rsi_signal
                    if df['rsi'][i]<30 and (df['rsi'][i]>df['rsi'][i-1]):
                        RSI_signal.append("Buy")
                    elif df['rsi'][i]>70 and (df['rsi'][i]<df['rsi'][i-1]):
                        RSI_signal.append('Sell')
                    else:
                        RSI_signal.append('Neutral')
                
                
                    # stoch_k_signal
                    if df['%D'][i]<20 and (df['%K'][i]<df['%D'][i]) and (df['%K'][i]>df['%K'][i-1]):
                        STOCH_K_signal.append("Buy")
                    elif df['%D'][i]>80 and (df['%K'][i]<df['%D'][i]) and (df['%K'][i]<df['%K'][i-1]):
                        STOCH_K_signal.append('Sell')
                    else:
                        STOCH_K_signal.append('Neutral')
                 
                    
                     # CCI_signal
                    if df['CCI'][i]<-100  and (df['CCI'][i]>df['CCI'][i-1]):
                        CCI_signal.append("Buy")
                    elif df['CCI'][i]>100 and (df['CCI'][i]<df['CCI'][i-1]):
                        CCI_signal.append('Sell')
                    else:
                        CCI_signal.append('Neutral')
                
                    
                
                    
                    # ADX_signal
                    if df['ADX'][i]>20 and (df['DMP_14'][i]>df['DMN_14'][i]) :
                        ADX_signal.append("Buy")
                    elif df['ADX'][i]<20 and (df['DMP_14'][i]<df['DMN_14'][i]) :
                        ADX_signal.append('Sell')
                    else:
                        ADX_signal.append('Neutral')
                
                     
                    # AO_signal
                    if df['AO'][i]>0 and df['AO'][i-2]>df['AO'][i-1]<df['AO'][i]:
                        AO_signal.append("Buy")
                  
                    elif df['AO'][i]>0 and df['AO'][i-2]<df['AO'][i-1]>df['AO'][i]:
                        AO_signal.append('Sell')
                
                
                    elif df['AO'][i]<0 and df['AO'][i-2]>df['AO'][i-1]<df['AO'][i]:
                        AO_signal.append('Sell')
                    
                    elif df['AO'][i]<0 and df['AO'][i-2]<df['AO'][i-1]>df['AO'][i]:
                        AO_signal.append('Sell')
                
                    else:
                        AO_signal.append('Neutral')
                
                    
                
                    
                    # Momentum_signal
                    if (df['Momentum'][i]>df['Momentum'][i-1]):
                        Momentum_signal.append("Buy")
                    else:
                        Momentum_signal.append("Sell")
                  
                
                
                    # MACD_line_signal
                    if df['MACD_Level'][i]>0:
                        MACD_signal.append("Buy")
                    elif df['MACD_Level'][i]<0:
                        MACD_signal.append('Sell')   
                    else:
                        MACD_signal.append('Neutral')
                
                
                    
                    # stocastic fast %killbgscripts
                
                    if df['fastd'][i]<20 and df['fastk'][i]>df['fastd'][i] and df['fastk'][i]>df['fastk'][i-1]:
                        stochfastk_signal.append("Buy")
                    elif df['fastd'][i]>80 and df['fastk'][i]<df['fastd'][i] and df['fastk'][i]<df['fastk'][i-1]:
                        stochfastk_signal.append('Sell')   
                    else:
                        stochfastk_signal.append('Neutral')
                
                
                    
                    # WPR_signal
                    if df['WPR'][i]<-80 and df['WPR'][i]>df['WPR'][i-1]:
                        WPR_signal.append("Buy")
                    elif df['WPR'][i]>-20 and df['WPR'][i]<df['WPR'][i-1]:
                        WPR_signal.append('Sell')
                    else:
                        WPR_signal.append('Neutral')
                
                
                    
                    #uo_siganl
                    if df['uo'][i]<30 and df['uo'][i]>df['uo'][i-1]:
                        uo_signal.append("Buy")
                    elif df['uo'][i]>70 and df['uo'][i]<df['uo'][i-1]:
                        uo_signal.append('Sell')
                    else:
                        uo_signal.append('Neutral')
    
                
                
                df['ema_5_signal'] = ema_5_signal
                df['sma_5_signal']= sma_5_signal
                df['ema_10_signal'] = ema_10_signal
                df['sma_10_signal'] = sma_10_signal
                df['ema_20_signal'] = ema_20_signal
                df['sma_20_signal'] = sma_20_signal
                df['ema_30_signal'] = ema_30_signal
                df['sma_30_signal'] = sma_30_signal
                df['ema_50_signal'] = ema_50_signal
                df['sma_50_signal'] = sma_50_signal
                df['ema_100_signal'] = ema_100_signal
                df['sma_100_signal'] = sma_100_signal
                df['ema_200_signal'] = ema_200_signal
                df['sma_200_signal'] = sma_200_signal
                df['wvma_signal'] = wvma_signal
                df['hma_signal'] = hma_signal
                
                
                
                df['RSI_signal'] =RSI_signal
                df['STOCH_K_signal'] = STOCH_K_signal
                df['CCI_signal'] = CCI_signal
                df['AO_signal'] = AO_signal
                df['Momentum_signal']= Momentum_signal
                df['MACD_signal'] = MACD_signal
                df['stochfastk_signal']=stochfastk_signal
                df['WPR_signal'] =WPR_signal
                df['uo_signal'] = uo_signal
                
                
                
                moving_signal= df[['ema_5_signal','sma_5_signal','ema_10_signal', 'sma_10_signal', 'ema_20_signal', 'sma_20_signal', 'ema_30_signal', 
                'sma_30_signal', 'ema_50_signal', 'sma_50_signal', 'ema_100_signal', 'sma_100_signal', 
                'ema_200_signal', 'sma_200_signal', 'wvma_signal', 'hma_signal']].tail(1)
                
                osc_signal= df[['RSI_signal', 'STOCH_K_signal', 'CCI_signal', 'AO_signal', 'Momentum_signal', 'MACD_signal','stochfastk_signal','WPR_signal','uo_signal']].tail(1)
                
                moving_values=df[['ema_5', 'sma_5', 'ema_10', 'sma_10', 'ema_20', 'sma_20', 'ema_30', 
                'sma_30', 'ema_50', 'sma_50', 'ema_100', 'sma_100', 'ema_200', 'sma_200', 'wvma', 'hma']].tail(1)
                # moving_values
                
                osc_values=df[['rsi','%K','%D','CCI','AO','Momentum','MACD_Line','MACD_Level','fastk','WPR','uo']].tail(1)
                # osc_values
                
                m=moving_signal.values.tolist()[0]
                m_buy_count=m.count('Buy')
                m_sell_count=m.count('Sell')
                m_neutral_count=m.count('Neutral')
                
                m_total=m_buy_count+m_sell_count
                m_buy_percent=(m_buy_count/m_total)*100
                # m_buy_percent
                
                
                moving_averages_signal=[]
                if m_buy_percent<=20:
                    moving_averages_signal.append('Strong Sell')
                elif 20< m_buy_percent <=40:
                    moving_averages_signal.append('Sell')
                elif 40<m_buy_percent<=60:
                    moving_averages_signal.append('Neutral')
                elif 60<m_buy_percent<=80:
                    moving_averages_signal.append('Buy')
                elif m_buy_percent>80:
                    moving_averages_signal.append('Strong Buy')
                
                
                # moving_averages_signal
                
                
                s=osc_signal.values.tolist()[0]
                s_buy_count=s.count('Buy')
                s_sell_count=s.count('Sell')
                s_neutral_count=s.count('Neutral')
                
                
                value=50
                s_buy_value=s_buy_count*15
                s_sell_value=s_sell_count*15
                
                value=value+s_buy_value-s_sell_value
                # value
                
                
                osc_signal=[]
                if value<20:
                    osc_signal.append('Strong Sell')
                elif 20<=value<40:
                    osc_signal.append('Sell')
                elif 40<=value<60:
                    osc_signal.append('Neutral')
                elif 60<=value<80:
                    osc_signal.append('Buy')
                
                elif value>=80:
                    osc_signal.append('Strong Buy')
                
                
                
                
                if moving_averages_signal[0]=='Strong Sell':
                    moving_percent=0
                elif moving_averages_signal[0]=='Sell':
                    moving_percent=25
                elif moving_averages_signal[0]=='Neutral':
                    moving_percent=50
                elif moving_averages_signal[0]=='Buy':
                    moving_percent=75
                elif moving_averages_signal[0]=='Strong Buy':
                    moving_percent=100 
                    
                
                
                if osc_signal[0]=='Strong Sell':
                    osc_percent=0
                elif osc_signal[0]=='Sell':
                    osc_percent=25
                elif osc_signal[0]=='Neutral':
                    osc_percent=50
                elif osc_signal[0]=='Buy':
                    osc_percent=75
                elif osc_signal[0]=='Strong Buy':
                    osc_percent=100
                
                
                
                final_value=(moving_percent+osc_percent)/2
                final_value=int(final_value)
                # final_value
                
                
                final_weighted_signal=[]
                if final_value<=20:
                    final_weighted_signal.append('Strong Sell')
                elif 20<final_value<=40:
                    final_weighted_signal.append('Sell')
                elif 40<final_value<=60:
                    final_weighted_signal.append('Neutral')
                elif 60<final_value<=80:
                    final_weighted_signal.append('Buy')
                
                elif final_value>80:
                    final_weighted_signal.append('Strong Buy')
                
                
                
                
                
                # print(osc_signal,moving_averages_signal)
                # print(final_weighted_signal)
            
            
            
            
                
                dict[f"{ticker}"]=osc_signal[0],moving_averages_signal[0],final_weighted_signal[0]
            except:
                continue
                
    
     
    
        z=pd.DataFrame(dict)
        q=z.transpose()
        q.reset_index(inplace=True)
        q.rename(columns={'index':'ticker',0:'osc_signal',1:'moving_average_signal',2:'final_weighted_signal'},inplace=True)
        # x=q.loc[(q['final_weighted_signal']=='Strong Buy')]
        x = q.loc[(q['final_weighted_signal'] == 'Strong Buy') | (q['final_weighted_signal'] == 'Buy')]
    
    
        # x.to_csv(f"signal_{current_date}.csv")
    
        s=x['ticker'].to_list()
        for elm in s:
            overall_indicators_list.append(elm)
        
    
        
    
            
    
    
    
    # #Large and mid cap stocs across nse
    # ticker_list= [
    #     "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "INFY.NS", "HDFC.NS",
    #     "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "LICI.NS", "LT.NS",
    #     "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "DMART.NS",
    #     "ULTRACEMCO.NS", "BAJAJFINSV.NS", "WIPRO.NS", "ADANIENT.NS", "ONGC.NS", "NTPC.NS", "JSWSTEEL.NS",
    #     "POWERGRID.NS", "M&M.NS", "LTIM.NS", "TATAMOTORS.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    #     "TATASTEEL.NS", "HINDZINC.NS", "PIDILITIND.NS", "SIEMENS.NS", "ADANITRANS.NS", "SBILIFE.NS", "IOC.NS",
    #     "BAJAJ-AUTO.NS", "GRASIM.NS", "TECHM.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "VEDL.NS", "GODREJCP.NS",
    #     "DABUR.NS", "ATGL.NS", "SHREECEM.NS", "HAL.NS", "HINDALCO.NS", "VBL.NS", "DLF.NS", "BANKBARODA.NS",
    #     "INDUSINDBK.NS", "EICHERMOT.NS", "DRREDDY.NS", "DIVISLAB.NS", "BPCL.NS", "HAVELLS.NS", "ADANIPOWER.NS",
    #     "INDIGO.NS", "CIPLA.NS", "AMBUJACEM.NS", "SRF.NS", "ABB.NS", "BEL.NS", "SBICARD.NS", "GAIL.NS",
    #     "BAJAJHLDNG.NS", "TATACONSUM.NS", "ICICIPRULI.NS", "CHOLAFIN.NS", "MARICO.NS", "APOLLOHOSP.NS",
    #     "TATAPOWER.NS", "BOSCHLTD.NS", "BERGEPAINT.NS", "JINDALSTEL.NS", "MCDOWELL-N.NS", "UPL.NS", "AWL.NS",
    #     "ICICIGI.NS", "TORNTPHARM.NS", "CANBK.NS", "PNB.NS", "TVSMOTOR.NS", "ZYDUSLIFE.NS", "TIINDIA.NS",
    #     "TRENT.NS", "IDBI.NS", "NAUKRI.NS", "SHRIRAMFIN.NS", "HEROMOTOCO.NS", "INDHOTEL.NS", "PIIND.NS",
    #     "IRCTC.NS", "CGPOWER.NS", "UNIONBANK.NS", "MOTHERSON.NS", "CUMMINSIND.NS", "SCHAEFFLER.NS", "LODHA.NS",
    #     "ZOMATO.NS", "PGHH.NS", "YESBANK.NS", "POLYCAB.NS", "MAXHEALTH.NS", "IOB.NS", "PAGEIND.NS", "COLPAL.NS",
    #     "ASHOKLEY.NS", "ALKEM.NS", "NHPC.NS", "PAYTM.NS", "PFC.NS", "JSWENERGY.NS", "MUTHOOTFIN.NS", "AUBANK.NS",
    #     "INDUSTOWER.NS", "BALKRISIND.NS", "UBL.NS", "ABCAPITAL.NS", "TATAELXSI.NS", "DALBHARAT.NS", "HDFCAMC.NS",
    #     "INDIANB.NS", "ASTRAL.NS", "BHARATFORG.NS", "LTTS.NS", "MRF.NS", "TATACOMM.NS", "NYKAA.NS", "CONCOR.NS",
    #     "PERSISTENT.NS", "PATANJALI.NS", "IRFC.NS", "LINDEINDIA.NS", "IDFCFIRSTB.NS", "PETRONET.NS",
    #     "SOLARINDS.NS", "SAIL.NS", "MPHASIS.NS", "HINDPETRO.NS", "APLAPOLLO.NS", "FLUOROCHEM.NS", "NMDC.NS",
    #     "HONAUT.NS", "SUPREMEIND.NS", "GUJGASLTD.NS", "BANDHANBNK.NS", "ACC.NS", "OBEROIRLTY.NS", "BANKINDIA.NS",
    #     "RECLTD.NS", "AUROPHARMA.NS", "STARHEALTH.NS", "IGL.NS", "LUPIN.NS", "UCOBANK.NS", "JUBLFOOD.NS",
    #     "POLICYBZR.NS", "GODREJPROP.NS", "M&MFIN.NS", "IDEA.NS", "OFSS.NS", "FEDERALBNK.NS", "MANYAVAR.NS",
    #     "UNOMINDA.NS", "AIAENG.NS", "THERMAX.NS", "OIL.NS", "VOLTAS.NS", "3MINDIA.NS", "COROMANDEL.NS",
    #     "SUNDARMFIN.NS", "KPITTECH.NS", "DEEPAKNTR.NS", "ESCORTS.NS", "BIOCON.NS", "TATACHEM.NS", "TORNTPOWER.NS",
    #     "GMRINFRA.NS", "BHEL.NS", "SONACOMS.NS", "DELHIVERY.NS", "SYNGENE.NS", "CRISIL.NS", "GICRE.NS",
    #     "COFORGE.NS", "PHOENIXLTD.NS", "JKCEMENT.NS", "POONAWALLA.NS", "GLAXO.NS", "MFSL.NS", "METROBRAND.NS",
    #     "MSUMI.NS", "SUMICHEM.NS", "RELAXO.NS", "NAVINFLUOR.NS", "SKFINDIA.NS", "CENTRALBK.NS", "GLAND.NS",
    #     "KANSAINER.NS", "GRINDWELL.NS", "TIMKEN.NS", "IPCALAB.NS", "SUNDRMFAST.NS", "ATUL.NS", "ZEEL.NS",
    #     "L&TFH.NS", "ABFRL.NS", "APOLLOTYRE.NS","KPRMILL.NS", "ZFCVINDIA.NS", "FORTIS.NS", "AARTIIND.NS", "HATSUN.NS", "CARBORUNIV.NS", "CROMPTON.NS",
    #     "VINATIORGA.NS", "IIFL.NS", "BATAINDIA.NS", "BDL.NS", "LICHSGFIN.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS",
    #     "ENDURANCE.NS", "DEVYANI.NS", "PSB.NS", "DIXON.NS", "KAJARIACER.NS", "WHIRLPOOL.NS", "MAHABANK.NS",
    #     "SUNTV.NS", "PEL.NS", "PRESTIGE.NS", "NIACL.NS", "RADICO.NS", "PFIZER.NS", "NH.NS", "EMAMILTD.NS",
    #     "LAURUSLABS.NS", "FIVESTAR.NS", "AJANTPHARM.NS", "INDIAMART.NS", "360ONE.NS", "KEI.NS", "JBCHEPHARM.NS",
    #     "LALPATHLAB.NS", "JSL.NS", "IRB.NS", "EXIDEIND.NS", "PVR.NS", "GSPL.NS", "BLUEDART.NS", "NATIONALUM.NS",
    #     "RVNL.NS", "CREDITACC.NS", "TRIDENT.NS", "POWERINDIA.NS", "MEDANTA.NS", "GILLETTE.NS", "RATNAMANI.NS",
    #     "ELGIEQUIP.NS", "ISEC.NS", "CGCL.NS", "GODREJIND.NS", "CLEAN.NS", "MAZDOCK.NS", "MAHINDCIE.NS",
    #     "AEGISCHEM.NS", "FACT.NS", "BLUESTARCO.NS", "SANOFI.NS", "FINEORG.NS", "AFFLE.NS", "GLENMARK.NS",
    #     "NAM-INDIA.NS", "SJVN.NS", "REDINGTON.NS", "AAVAS.NS", "IDFC.NS", "FINCABLES.NS", "NUVOCO.NS",
    #     "BAJAJELEC.NS", "APTUS.NS", "SUVENPHAR.NS", "ASTERDM.NS", "RHIM.NS", "KEC.NS", "SONATSOFTW.NS",
    #     "AETHER.NS", "DCMSHRIRAM.NS", "IEX.NS", "HAPPSTMNDS.NS", "KIMS.NS", "ALKYLAMINE.NS", "CYIENT.NS",
    #     "CHAMBLFERT.NS", "ASAHIINDIA.NS", "CASTROLIND.NS", "BRIGADE.NS", "KALYANKJIL.NS", "TTML.NS", "VGUARD.NS",
    #     "NLCINDIA.NS", "LAXMIMACH.NS", "TRITURBINE.NS", "FINPIPE.NS", "AKZOINDIA.NS", "MANAPPURAM.NS",
    #     "EIHOTEL.NS", "CENTURYPLY.NS", "NATCOPHARM.NS", "KIOCL.NS", "CHOLAHLDNG.NS", "CAMPUS.NS", "CAMS.NS",
    #     "AMARAJABAT.NS", "ZYDUSWELL.NS", "BASF.NS", "TEJASNET.NS", "APLLTD.NS", "MGL.NS", "GRINFRA.NS",
    #     "ANGELONE.NS", "SFL.NS", "TTKPRESTIG.NS", "APARINDS.NS", "HINDCOPPER.NS", "CDSL.NS", "GODFRYPHLP.NS",
    #     "RENUKA.NS", "CUB.NS", "JKLAKSHMI.NS", "ANURAS.NS", "MRPL.NS", "GESHIP.NS", "POLYMED.NS", "NSLNISP.NS",
    #     "BIKAJI.NS", "MOTILALOFS.NS", "ABSLAMC.NS", "CESC.NS", "TATAINVEST.NS", "ALLCARGO.NS", "KALPATPOWR.NS",
    #     "PNBHOUSING.NS", "HUDCO.NS", "ITI.NS", "ROUTE.NS", "RITES.NS", "VTL.NS", "RBLBANK.NS", "HFCL.NS",
    #     "KARURVYSYA.NS", "CERA.NS", "EIDPARRY.NS", "INGERRAND.NS", "GALAXYSURF.NS", "PPLPHARMA.NS", "UTIAMC.NS",
    #     "KRBL.NS", "RAYMOND.NS", "ASTRAZEN.NS", "VIPIND.NS", "ACI.NS", "BALRAMCHIN.NS", "SUZLON.NS", "GODREJAGRO.NS",
    #     "GNFC.NS", "ERIS.NS", "PGHL.NS", "MEDPLUS.NS", "SAPPHIRE.NS", "DATAPATTNS.NS", "SUNCLAYLTD.NS", "JBMA.NS",
    #     "EASEMYTRIP.NS", "CCL.NS", "EQUITASBNK.NS", "CHALET.NS", "RAINBOW.NS", "PNCINFRA.NS", "FSL.NS", "KSB.NS",
    #     "BSOFT.NS", "KNRCON.NS", "SHOPERSTOP.NS", "SYMPHONY.NS", "CENTURYTEX.NS", "CANFINHOME.NS", "GRANULES.NS",
    #     "TANLA.NS", "JYOTHYLAB.NS", "SPLPETRO.NS", "DEEPAKFERT.NS", "CRAFTSMAN.NS", "BIRLACORPN.NS", "BLS.NS",
    #     "SHYAMMETL.NS", "NCC.NS", "GMMPFAUDLR.NS", "LATENTVIEW.NS", "USHAMART.NS", "HOMEFIRST.NS", "JKPAPER.NS",
    #     "TMB.NS", "JINDWORLD.NS", "METROPOLIS.NS", "SAREGAMA.NS", "NBCC.NS", "ECLERX.NS", "BALAMINES.NS",
    #     "WELSPUNIND.NS", "PRAJIND.NS", "COCHINSHIP.NS", "ZENSARTECH.NS", "AMBER.NS", "LEMONTREE.NS",
    #     "PRINCEPIPE.NS", "TRIVENI.NS", "GARFIBRES.NS", "LXCHEM.NS", "STLTECH.NS", "CEATLTD.NS", "BSE.NS",
    #     "SPARC.NS", "ALOKINDS.NS", "ORIENTELEC.NS", "INDIACEM.NS", "JUBLINGREA.NS", "KIRLOSENG.NS", "TCIEXP.NS",
    #     "JMFINANCIL.NS", "NETWORK18.NS", "BBTC.NS", "SWANENERGY.NS", "GPPL.NS", "KAYNES.NS", "VRLLOG.NS",
    #     "INTELLECT.NS", "SWSOLAR.NS", "CHEMPLASTS.NS", "QUESS.NS", "ROLEXRINGS.NS", "MAHLIFE.NS", "ESABINDIA.NS",
    #     "MHRIL.NS", "GOCOLORS.NS", "HGS.NS", "BORORENEW.NS", "GAEL.NS", "MAPMYINDIA.NS", "PRSMJOHNSN.NS",
    #     "RUSTOMJEE.NS", "IRCON.NS", "RCF.NS", "WELCORP.NS", "BEML.NS", "GRSE.NS", "EPL.NS", "MINDACORP.NS",
    #     "GRAPHITE.NS", "HGINFRA.NS", "OLECTRA.NS", "RELINFRA.NS", "JUSTDIAL.NS", "RAIN.NS", "IONEXCHANG.NS"
    #  ]
    # ticker_list=['BAJAJFINSV.NS']
    
    # ticker_list = [
    #     "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "INFY.NS", "HDFC.NS", "ITC.NS", "SBIN.NS", 
    #     "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "LICI.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", 
    #     "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "DMART.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "WIPRO.NS", "ADANIENT.NS", 
    #     "ONGC.NS", "NTPC.NS", "JSWSTEEL.NS", "POWERGRID.NS", "M&M.NS", "LTIM.NS", "TATAMOTORS.NS", "ADANIGREEN.NS", 
    #     "ADANIPORTS.NS", "COALINDIA.NS", "TATASTEEL.NS", "HINDZINC.NS", "PIDILITIND.NS", "SIEMENS.NS", "ADANITRANS.NS", 
    #     "SBILIFE.NS", "IOC.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "TECHM.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "VEDL.NS", 
    #     "GODREJCP.NS", "DABUR.NS", "ATGL.NS", "SHREECEM.NS", "HAL.NS", "HINDALCO.NS", "VBL.NS", "DLF.NS", "BANKBARODA.NS", 
    #     "INDUSINDBK.NS", "EICHERMOT.NS", "DRREDDY.NS", "DIVISLAB.NS", "BPCL.NS", "HAVELLS.NS", "ADANIPOWER.NS", "INDIGO.NS", 
    #     "CIPLA.NS", "AMBUJACEM.NS", "SRF.NS", "ABB.NS", "BEL.NS", "SBICARD.NS", "GAIL.NS", "BAJAJHLDNG.NS", "TATACONSUM.NS", 
    #     "ICICIPRULI.NS", "CHOLAFIN.NS", "MARICO.NS", "APOLLOHOSP.NS", "TATAPOWER.NS", "BOSCHLTD.NS", "BERGEPAINT.NS", 
    #     "JINDALSTEL.NS", "MCDOWELL-N.NS", "UPL.NS", "AWL.NS", "ICICIGI.NS", "TORNTPHARM.NS", "CANBK.NS", "PNB.NS", 
    #     "TVSMOTOR.NS", "ZYDUSLIFE.NS", "TIINDIA.NS", "TRENT.NS", "IDBI.NS", "NAUKRI.NS", "SHRIRAMFIN.NS", "HEROMOTOCO.NS", 
    #     "INDHOTEL.NS", "PIIND.NS", "IRCTC.NS", "CGPOWER.NS", "UNIONBANK.NS", "MOTHERSON.NS", "CUMMINSIND.NS", "SCHAEFFLER.NS", 
    #     "LODHA.NS", "ZOMATO.NS", "PGHH.NS", "YESBANK.NS", "POLYCAB.NS", "MAXHEALTH.NS", "IOB.NS", "PAGEIND.NS", "COLPAL.NS", 
    #     "ASHOKLEY.NS", "ALKEM.NS", "NHPC.NS", "PAYTM.NS", "PFC.NS", "JSWENERGY.NS", "MUTHOOTFIN.NS", "AUBANK.NS", 
    #     "INDUSTOWER.NS", "BALKRISIND.NS", "UBL.NS", "ABCAPITAL.NS", "TATAELXSI.NS", "DALBHARAT.NS", "HDFCAMC.NS", 
    #     "INDIANB.NS", "ASTRAL.NS", "BHARATFORG.NS", "LTTS.NS", "MRF.NS", "TATACOMM.NS", "NYKAA.NS", "CONCOR.NS", 
    #     "PERSISTENT.NS", "PATANJALI.NS", "IRFC.NS", "LINDEINDIA.NS", "IDFCFIRSTB.NS", "PETRONET.NS", "SOLARINDS.NS", 
    #     "SAIL.NS", "MPHASIS.NS", "HINDPETRO.NS", "APLAPOLLO.NS", "FLUOROCHEM.NS", "NMDC.NS", "HONAUT.NS", "SUPREMEIND.NS", 
    #     "GUJGASLTD.NS", "BANDHANBNK.NS", "ACC.NS", "OBEROIRLTY.NS", "BANKINDIA.NS", "RECLTD.NS", "AUROPHARMA.NS", 
    #     "STARHEALTH.NS", "IGL.NS", "LUPIN.NS", "UCOBANK.NS", "JUBLFOOD.NS", "POLICYBZR.NS", "GODREJPROP.NS", "M&MFIN.NS", 
    #     "IDEA.NS", "OFSS.NS", "FEDERALBNK.NS", "MANYAVAR.NS", "UNOMINDA.NS", "AIAENG.NS", "THERMAX.NS", "OIL.NS", 
    #     "VOLTAS.NS", "3MINDIA.NS", "COROMANDEL.NS", "SUNDARMFIN.NS", "KPITTECH.NS", "DEEPAKNTR.NS", "ESCORTS.NS", 
    #     "BIOCON.NS", "TATACHEM.NS", "TORNTPOWER.NS", "GMRINFRA.NS", "BHEL.NS", "SONACOMS.NS", "DELHIVERY.NS", "SYNGENE.NS", 
    #     "CRISIL.NS", "GICRE.NS", "COFORGE.NS", "PHOENIXLTD.NS", "JKCEMENT.NS", "POONAWALLA.NS", "GLAXO.NS", "MFSL.NS", 
    #     "METROBRAND.NS", "MSUMI.NS","SUMICHEM.NS", "RELAXO.NS", "NAVINFLUOR.NS", "SKFINDIA.NS", "CENTRALBK.NS", "GLAND.NS", "KANSAINER.NS", "GRINDWELL.NS",
    #     "TIMKEN.NS", "IPCALAB.NS", "SUNDRMFAST.NS", "ATUL.NS", "ZEEL.NS", "L&TFH.NS", "ABFRL.NS", "APOLLOTYRE.NS", "KPRMILL.NS",
    #     "ZFCVINDIA.NS", "FORTIS.NS", "AARTIIND.NS", "HATSUN.NS", "CARBORUNIV.NS", "CROMPTON.NS", "VINATIORGA.NS", "IIFL.NS",
    #     "BATAINDIA.NS", "BDL.NS", "LICHSGFIN.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "ENDURANCE.NS", "DEVYANI.NS", "PSB.NS",
    #     "DIXON.NS", "KAJARIACER.NS", "WHIRLPOOL.NS", "MAHABANK.NS", "SUNTV.NS", "PEL.NS", "PRESTIGE.NS", "NIACL.NS", "RADICO.NS",
    #     "PFIZER.NS", "NH.NS", "EMAMILTD.NS", "LAURUSLABS.NS", "FIVESTAR.NS", "AJANTPHARM.NS", "INDIAMART.NS", "360ONE.NS", 
    #     "KEI.NS", "JBCHEPHARM.NS", "LALPATHLAB.NS", "JSL.NS", "IRB.NS", "EXIDEIND.NS", "PVR.NS", "GSPL.NS", "BLUEDART.NS", 
    #     "NATIONALUM.NS", "RVNL.NS", "CREDITACC.NS", "TRIDENT.NS", "POWERINDIA.NS", "MEDANTA.NS", "GILLETTE.NS", "RATNAMANI.NS", 
    #     "ELGIEQUIP.NS", "ISEC.NS", "CGCL.NS", "GODREJIND.NS", "CLEAN.NS", "MAZDOCK.NS", "MAHINDCIE.NS", "AEGISCHEM.NS", "FACT.NS", 
    #     "BLUESTARCO.NS", "SANOFI.NS", "FINEORG.NS", "AFFLE.NS", "GLENMARK.NS", "NAM-INDIA.NS", "SJVN.NS", "REDINGTON.NS", "AAVAS.NS", 
    #     "IDFC.NS", "FINCABLES.NS", "NUVOCO.NS", "BAJAJELEC.NS", "APTUS.NS", "SUVENPHAR.NS", "ASTERDM.NS", "RHIM.NS", "KEC.NS", 
    #     "SONATSOFTW.NS", "AETHER.NS", "DCMSHRIRAM.NS", "IEX.NS", "HAPPSTMNDS.NS", "KIMS.NS", "ALKYLAMINE.NS", "CYIENT.NS", 
    #     "CHAMBLFERT.NS", "ASAHIINDIA.NS", "CASTROLIND.NS", "BRIGADE.NS", "KALYANKJIL.NS", "TTML.NS", "VGUARD.NS", "NLCINDIA.NS", 
    #     "LAXMIMACH.NS", "TRITURBINE.NS", "FINPIPE.NS", "AKZOINDIA.NS", "MANAPPURAM.NS", "EIHOTEL.NS", "CENTURYPLY.NS", 
    #     "NATCOPHARM.NS", "KIOCL.NS", "CHOLAHLDNG.NS", "CAMPUS.NS", "CAMS.NS", "AMARAJABAT.NS", "ZYDUSWELL.NS", "BASF.NS", 
    #     "TEJASNET.NS", "APLLTD.NS", "MGL.NS", "GRINFRA.NS", "ANGELONE.NS", "SFL.NS", "TTKPRESTIG.NS", "APARINDS.NS", 
    #     "HINDCOPPER.NS", "CDSL.NS", "GODFRYPHLP.NS", "RENUKA.NS", "CUB.NS", "JKLAKSHMI.NS", "ANURAS.NS", "MRPL.NS", 
    #     "GESHIP.NS", "POLYMED.NS", "NSLNISP.NS", "BIKAJI.NS", "MOTILALOFS.NS", "ABSLAMC.NS", "CESC.NS", "TATAINVEST.NS", 
    #     "ALLCARGO.NS", "KALPATPOWR.NS", "PNBHOUSING.NS", "HUDCO.NS", "ITI.NS", "ROUTE.NS", "RITES.NS", "VTL.NS", "RBLBANK.NS", 
    #     "HFCL.NS", "KARURVYSYA.NS", "CERA.NS", "EIDPARRY.NS", "INGERRAND.NS", "GALAXYSURF.NS", "PPLPHARMA.NS", "UTIAMC.NS", 
    #     "KRBL.NS", "RAYMOND.NS", "ASTRAZEN.NS", "VIPIND.NS", "ACI.NS", "BALRAMCHIN.NS", "SUZLON.NS", "GODREJAGRO.NS", 
    #     "GNFC.NS", "ERIS.NS", "PGHL.NS", "MEDPLUS.NS", "SAPPHIRE.NS", "DATAPATTNS.NS", "SUNCLAYLTD.NS", "JBMA.NS", 
    #     "EASEMYTRIP.NS", "CCL.NS", "EQUITASBNK.NS", "CHALET.NS", "RAINBOW.NS", "PNCINFRA.NS", "FSL.NS", "KSB.NS", 
    #     "BSOFT.NS", "KNRCON.NS", "SHOPERSTOP.NS", "SYMPHONY.NS", "CENTURYTEX.NS", "CANFINHOME.NS", "GRANULES.NS", 
    #     "TANLA.NS", "JYOTHYLAB.NS", "SPLPETRO.NS","DEEPAKFERT.NS", "CRAFTSMAN.NS", "BIRLACORPN.NS", "BLS.NS", "SHYAMMETL.NS", "NCC.NS", "GMMPFAUDLR.NS", "LATENTVIEW.NS",
    #     "USHAMART.NS", "HOMEFIRST.NS", "JKPAPER.NS", "TMB.NS", "JINDWORLD.NS", "METROPOLIS.NS", "SAREGAMA.NS", "NBCC.NS",
    #     "ECLERX.NS", "BALAMINES.NS", "WELSPUNIND.NS", "PRAJIND.NS", "COCHINSHIP.NS", "ZENSARTECH.NS", "AMBER.NS", "LEMONTREE.NS",
    #     "PRINCEPIPE.NS", "TRIVENI.NS", "GARFIBRES.NS", "LXCHEM.NS", "STLTECH.NS", "CEATLTD.NS", "BSE.NS", "SPARC.NS", "ALOKINDS.NS",
    #     "ORIENTELEC.NS", "INDIACEM.NS", "JUBLINGREA.NS", "KIRLOSENG.NS", "TCIEXP.NS", "JMFINANCIL.NS", "NETWORK18.NS", "BBTC.NS",
    #     "SWANENERGY.NS", "GPPL.NS", "KAYNES.NS", "VRLLOG.NS", "INTELLECT.NS", "SWSOLAR.NS", "CHEMPLASTS.NS", "QUESS.NS",
    #     "ROLEXRINGS.NS", "MAHLIFE.NS", "ESABINDIA.NS", "MHRIL.NS", "GOCOLORS.NS", "HGS.NS", "BORORENEW.NS", "GAEL.NS",
    #     "MAPMYINDIA.NS", "PRSMJOHNSN.NS", "RUSTOMJEE.NS", "IRCON.NS", "RCF.NS", "WELCORP.NS", "BEML.NS", "GRSE.NS", "EPL.NS",
    #     "MINDACORP.NS", "GRAPHITE.NS", "HGINFRA.NS", "OLECTRA.NS", "RELINFRA.NS", "JUSTDIAL.NS", "RAIN.NS", "IONEXCHANG.NS",
    #     "EDELWEISS.NS", "UJJIVANSFB.NS", "TV18BRDCST.NS", "GPIL.NS", "MTARTECH.NS", "TCI.NS", "RTNINDIA.NS", "VSTIND.NS",
    #     "SAFARI.NS", "ACE.NS", "MAHSCOOTER.NS", "DELTACORP.NS", "GLS.NS", "GHCL.NS", "INDIGOPNTS.NS", "MAHSEAMLES.NS", 
    #     "SUPRAJIT.NS", "KFINTECH.NS", "GSFC.NS", "J&KBANK.NS", "RELIGARE.NS", "MASTEK.NS", "SIS.NS", "JINDALSAW.NS", "TEGA.NS",
    #     "SYRMA.NS", "AVANTIFEED.NS", "STARCEMENT.NS", "IBULHSGFIN.NS", "RKFORGE.NS", "CAPLIPOINT.NS", "VAIBHAVGBL.NS", "RBA.NS",
    #     "JUBLPHARMA.NS", "SHARDACROP.NS", "NIITLTD.NS", "PCBL.NS", "MASFIN.NS", "SCI.NS", "PDSL.NS", "GUJALKALI.NS", "ELECON.NS",
    #     "CMSINFO.NS", "VMART.NS", "ICRA.NS", "JSWHL.NS", "FDC.NS", "CSBBANK.NS", "KTKBANK.NS", "MMTC.NS", "ENGINERSIN.NS",
    #     "SUNTECK.NS", "PRIVISCL.NS", "PARADEEP.NS", "SOBHA.NS", "FUSION.NS", "GMDCLTD.NS", "VIJAYA.NS", "JAMNAAUTO.NS",
    #     "ANANTRAJ.NS", "SANSERA.NS", "MFL.NS", "AHLUCONT.NS", "BSHSL.NS", "TATACOFFEE.NS", "TEAMLEASE.NS", "JKTYRE.NS",
    #     "VARROC.NS", "GREENLAM.NS", "JPPOWER.NS", "INFIBEAM.NS", "SPANDANA.NS", "HSCL.NS", "BHARATRAS.NS", "RAJRATAN.NS",
    #     "LAOPALA.NS", "SARDAEN.NS", "RALLIS.NS", "BOROLTD.NS", "RATEGAIN.NS", "SCHNEIDER.NS", "RPOWER.NS", "ARVINDFASN.NS",
    #     "TATVA.NS", "POWERMECH.NS", "HCG.NS", "NESCO.NS", "HEIDELBERG.NS", "TECHNOE.NS", "POLYPLEX.NS", "SURYAROSNI.NS",
    #     "AUTOAXLES.NS", "JWL.NS", "NFL.NS", "HEG.NS", "RAJRILTD.NS", "CHENNPETRO.NS", "WSTCSTPAPR.NS", "LUXIND.NS", "HIKAL.NS",
    #     "MIDHANI.NS", "HLEGLAS.NS", "SHAREINDIA.NS", "NOCIL.NS", "NAZARA.NS", "BANARISUG.NS", "ANANDRATHI.NS", "PRUDENT.NS",
    #     "GRAVITA.NS", "GREENPANEL.NS", "VESUVIUS.NS", "DCBBANK.NS", "ROSSARI.NS", "RESPONIND.NS", "TINPLATE.NS", "KIRLOSBROS.NS",
    #     "RAILTEL.NS", "AMIORG.NS", "ISGEC.NS", "NEOGEN.NS", "MARKSANS.NS", "NN.NS", "NEWGEN.NS", "BECTORFOOD.NS", "TWL.NS",
    #     "AARTIDRUGS.NS",    "UJJIVAN.NS", "GATEWAY.NS", "SULA.NS", "DAAWAT.NS", "SOUTHBANK.NS", "GET&D.NS", "HARSHA.NS", "PGEL.NS",
    #     "RSYSTEMS.NS", "INDOCO.NS", "MOLDTKPAC.NS", "IFBIND.NS", "SBCL.NS", "BCG.NS", "GREAVESCOT.NS", "MOIL.NS",
    #     "TATASTLLP.NS", "TARSONS.NS", "SHANTIGEAR.NS", "CHOICEIN.NS", "TIIL.NS", "DHANUKA.NS", "JCHAC.NS", "DODLA.NS",
    #     "DALMIASUG.NS", "VOLTAMP.NS", "ASTEC.NS", "SUDARSCHEM.NS", "KSCL.NS", "SUNFLAG.NS", "IBREALEST.NS", "THOMASCOOK.NS",
    #     "HBLPOWER.NS", "INOXWIND.NS", "NILKAMAL.NS", "ZENTEC.NS", "TCNSBRANDS.NS", "ADVENZYMES.NS", "STAR.NS", "FCL.NS",
    #     "KKCL.NS", "HINDWAREAP.NS", "MAHLOG.NS", "EMIL.NS", "JTEKTINDIA.NS", "MANINFRA.NS", "ITDC.NS", "APCOTEXIND.NS",
    #     "PRICOLLTD.NS", "PTC.NS", "AARTIPHARM.NS", "MBAPL.NS", "SAGCEM.NS", "TDPOWERSYS.NS", "JAICORPLTD.NS", "DBL.NS",
    #     "BARBEQUE.NS", "UNIPARTS.NS", "UFLEX.NS", "WONDERLA.NS", "PSPPROJECT.NS", "KIRLOSIND.NS", "IPL.NS", "DISHTV.NS",
    #     "TATAMETALI.NS", "PAISALO.NS", "PFOCUS.NS", "HEMIPROP.NS", "LGBBROSLTD.NS", "MAITHANALL.NS", "SSWL.NS", "NEULANDLAB.NS",
    #     "HATHWAY.NS", "THYROCARE.NS", "ORIENTCEM.NS", "DREAMFOLKS.NS", "ETHOSLTD.NS", "GLOBUSSPR.NS", "GANESHHOUC.NS",
    #     "ARVIND.NS", "ICIL.NS", "SHRIPISTON.NS", "WOCKPHARMA.NS", "DBREALTY.NS", "ISMTLTD.NS", "JINDALPOLY.NS", "WABAG.NS",
    #     "BAJAJCON.NS", "GENUSPOWER.NS", "BUTTERFLY.NS", "NAVNETEDUL.NS", "GOKEX.NS", "APOLLOPIPE.NS", "LANDMARK.NS",
    #     "IFCI.NS", "ATFL.NS", "EVEREADY.NS", "AGI.NS", "TI.NS", "ASHOKA.NS"
    # ]
    
    
    # ticker_list = [
    #     "ITC.NS", "SBIN.NS", 
    #     "BHARTIARTL.NS", "LT.NS", "HCLTECH.NS", "AXISBANK.NS", 
    #     "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", 
    #     "ONGC.NS", "NTPC.NS", "JSWSTEEL.NS", "POWERGRID.NS", "M&M.NS", "LTIM.NS", "TATAMOTORS.NS", 
    #     "ADANIPORTS.NS", "COALINDIA.NS", "TATASTEEL.NS", "PIDILITIND.NS", "SIEMENS.NS", 
    #     "SBILIFE.NS", "IOC.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "BRITANNIA.NS",
    #     "HAL.NS", "VBL.NS", "DLF.NS", "BANKBARODA.NS", 
    #     "INDUSINDBK.NS", "EICHERMOT.NS", "ADANIPOWER.NS", "INDIGO.NS", 
    #     "CIPLA.NS", "AMBUJACEM.NS", "ABB.NS", "BEL.NS", "GAIL.NS", "BAJAJHLDNG.NS", "TATACONSUM.NS", 
    #     "ICICIPRULI.NS", "CHOLAFIN.NS", "MARICO.NS", "APOLLOHOSP.NS", "TATAPOWER.NS", "BOSCHLTD.NS", "BERGEPAINT.NS", 
    #     "JINDALSTEL.NS", "MCDOWELL-N.NS", "ICICIGI.NS", "TORNTPHARM.NS", "CANBK.NS", "PNB.NS", 
    #     "TVSMOTOR.NS", "ZYDUSLIFE.NS", "TIINDIA.NS", "TRENT.NS", "IDBI.NS", "SHRIRAMFIN.NS", "HEROMOTOCO.NS", 
    #     "INDHOTEL.NS", "PIIND.NS", "CGPOWER.NS", "UNIONBANK.NS", "CUMMINSIND.NS", 
    #     "LODHA.NS", "ZOMATO.NS", "PGHH.NS", "POLYCAB.NS", "MAXHEALTH.NS", "COLPAL.NS",
    #     "ASHOKLEY.NS", "ALKEM.NS", "NHPC.NS", "PAYTM.NS", "PFC.NS", "JSWENERGY.NS", 
    #     "INDUSTOWER.NS", "UBL.NS", "TATAELXSI.NS", "DALBHARAT.NS",
    #     "INDIANB.NS", "BHARATFORG.NS", "MRF.NS", "TATACOMM.NS", 
    #     "PERSISTENT.NS", "LINDEINDIA.NS",
    #     "HINDPETRO.NS", "APLAPOLLO.NS", "SUPREMEIND.NS", 
    #     "OBEROIRLTY.NS","GODREJPROP.NS", 
    #     "IDEA.NS", "FEDERALBNK.NS", "UNOMINDA.NS", "AIAENG.NS", "THERMAX.NS", "OIL.NS","KPITTECH.NS","ESCORTS.NS", 
    #     "TORNTPOWER.NS", "BHEL.NS", 
    #     "CRISIL.NS","PHOENIXLTD.NS", "JKCEMENT.NS", "POONAWALLA.NS",
    #     "METROBRAND.NS", "CENTRALBK.NS", "GLAND.NS",
    #     "SUNDRMFAST.NS", "L&TFH.NS", "KPRMILL.NS",
    #     "ZFCVINDIA.NS", "FORTIS.NS", "CARBORUNIV.NS", "IIFL.NS","BDL.NS",
    #     "KAJARIACER.NS", "MAHABANK.NS", "PRESTIGE.NS", "RADICO.NS",
    #     "NH.NS", "FIVESTAR.NS", "AJANTPHARM.NS", "360ONE.NS",
    #     "KEI.NS", "JBCHEPHARM.NS", "JSL.NS", "IRB.NS", 
    #     "NATIONALUM.NS", "RVNL.NS", "CREDITACC.NS", "POWERINDIA.NS", "MEDANTA.NS", "RATNAMANI.NS", 
    #     "ELGIEQUIP.NS","CGCL.NS", "MAZDOCK.NS", "MAHINDCIE.NS", "AEGISCHEM.NS", "FACT.NS", 
    #     "BLUESTARCO.NS", "SJVN.NS",
    #     "IDFC.NS", "FINCABLES.NS", "ASTERDM.NS", "KEC.NS", 
    #     "SONATSOFTW.NS", "KIMS.NS", "CYIENT.NS",
    #     "ASAHIINDIA.NS","BRIGADE.NS", "KALYANKJIL.NS", "VGUARD.NS", "NLCINDIA.NS", 
    #     "LAXMIMACH.NS", "TRITURBINE.NS", "FINPIPE.NS", "AKZOINDIA.NS", "BASF.NS",
    #     "TEJASNET.NS", "ANGELONE.NS", "APARINDS.NS","CDSL.NS", "GODFRYPHLP.NS",
    #     "GESHIP.NS", "POLYMED.NS", "BIKAJI.NS", "MOTILALOFS.NS", "CESC.NS", "TATAINVEST.NS", 
    #     "PNBHOUSING.NS","RITES.NS", 
    #     "KARURVYSYA.NS", "CERA.NS", "INGERRAND.NS",
    #     "RAYMOND.NS", "ASTRAZEN.NS", "SUZLON.NS", 
    #     "SUNCLAYLTD.NS", "JBMA.NS",
    #     "CCL.NS", "EQUITASBNK.NS", "CHALET.NS", "RAINBOW.NS", "KSB.NS", 
    #     "SHOPERSTOP.NS","CANFINHOME.NS", 
    #     "JYOTHYLAB.NS", "SPLPETRO.NS", "CRAFTSMAN.NS", "BLS.NS", "NCC.NS", "LATENTVIEW.NS",
    #     "USHAMART.NS", "JINDWORLD.NS",
    #     "ECLERX.NS", "WELSPUNIND.NS","COCHINSHIP.NS", "LEMONTREE.NS",
    #     "TRIVENI.NS", "CEATLTD.NS", "BSE.NS", 
    #     "INDIACEM.NS","KIRLOSENG.NS",
    #     "SWANENERGY.NS", "GPPL.NS", "KAYNES.NS", "VRLLOG.NS",
    #     "ROLEXRINGS.NS", "ESABINDIA.NS", "MHRIL.NS", "GAEL.NS",
    #     "IRCON.NS", "RCF.NS", "WELCORP.NS", "BEML.NS", "GRSE.NS",
    #     "MINDACORP.NS", "HGINFRA.NS", "RELINFRA.NS", "IONEXCHANG.NS",
    #     "GPIL.NS", "MTARTECH.NS", "TCI.NS", "RTNINDIA.NS",
    #     "SAFARI.NS", "ACE.NS", "MAHSCOOTER.NS", "MAHSEAMLES.NS", 
    #     "KFINTECH.NS", "GSFC.NS", "J&KBANK.NS", "RELIGARE.NS", "JINDALSAW.NS", "TEGA.NS",
    #     "SYRMA.NS", "STARCEMENT.NS", "RKFORGE.NS",
    #     "PCBL.NS", "MASFIN.NS", "PDSL.NS", "GUJALKALI.NS", "ELECON.NS",
    #     "CMSINFO.NS","ICRA.NS","FDC.NS", "CSBBANK.NS", "KTKBANK.NS",
    #     "VIJAYA.NS",
    #     "ANANTRAJ.NS","AHLUCONT.NS", "TATACOFFEE.NS","JKTYRE.NS",
    #     "HSCL.NS",
    #     "LAOPALA.NS", "SARDAEN.NS","BOROLTD.NS", "RATEGAIN.NS", "SCHNEIDER.NS", "ARVINDFASN.NS",
    #     "POWERMECH.NS", "HCG.NS", "TECHNOE.NS", "SURYAROSNI.NS",
    #     "AUTOAXLES.NS", "JWL.NS", "CHENNPETRO.NS", "WSTCSTPAPR.NS",
    #     "SHAREINDIA.NS", "ANANDRATHI.NS", "PRUDENT.NS",
    #     "GRAVITA.NS","VESUVIUS.NS", "RESPONIND.NS", "KIRLOSBROS.NS",
    #     "RAILTEL.NS", "ISGEC.NS", "MARKSANS.NS", "NEWGEN.NS", "BECTORFOOD.NS",
    #     "UJJIVAN.NS", "GATEWAY.NS", "SULA.NS", "SOUTHBANK.NS", "GET&D.NS", "PGEL.NS",
    #     "RSYSTEMS.NS","SBCL.NS", "MOIL.NS",
    #     "SHANTIGEAR.NS", "CHOICEIN.NS", "TIIL.NS",
    #     "VOLTAMP.NS", "SUNFLAG.NS", "THOMASCOOK.NS",
    #     "HBLPOWER.NS", "INOXWIND.NS", "FCL.NS",
    #     "KKCL.NS", "HINDWAREAP.NS", "EMIL.NS", "JTEKTINDIA.NS", "MANINFRA.NS", "APCOTEXIND.NS",
    #     "PRICOLLTD.NS", "PTC.NS", "AARTIPHARM.NS", "TDPOWERSYS.NS", "JAICORPLTD.NS",
    #     "WONDERLA.NS", "PSPPROJECT.NS", "KIRLOSIND.NS",
    #     "PFOCUS.NS", "LGBBROSLTD.NS", "NEULANDLAB.NS",
    #     "ORIENTCEM.NS", "ETHOSLTD.NS", "GANESHHOUC.NS",
    #     "ARVIND.NS", "ICIL.NS", "DBREALTY.NS", "ISMTLTD.NS",
    #     "GOKEX.NS","LANDMARK.NS",
    #     "AGI.NS", "TI.NS"
    # ]
    
    
    ticker_list=['UNOMINDIA.NS','BDL.NS','PCBL.NS','MASFIN.NS']
    
    # len(ticker_list)
    
    import datetime
    
    current_date = datetime.date.today()
    increased_date = current_date + datetime.timedelta(days=1)
    # print(current_date)
    # print(increased_date)
    
    start_date='2018-01-01'
    end_date=increased_date
    
    
    
    
    
    
    generate_signal(ticker_list,start_date=start_date,end_date=end_date)
    
    
    # print(overall_indicators_list)
    # len(overall_indicators_list)
    
    greenzone_list=[]
    def generate_green_signal(overall_indicators_list,start_date,end_date):
        dict={}
        for ticker in overall_indicators_list:
            try:
                df=yf.download(ticker,start_date,end_date)
                df['res'] = df['High'].rolling(window=3).max()
                df['sup'] = df['Low'].rolling(window=3).min()
                # Pre-calculate the shifted columns
                df['res_shifted'] = df['res'].shift(1)
                df['sup_shifted'] = df['sup'].shift(1)
    
                # Determine the direction of the close using pre-calculated shifted columns
                df['avd'] = df.apply(lambda row: 1 if row['Close'] > row['res_shifted'] else (-1 if row['Close'] < row['sup_shifted'] else 0), axis=1)
                # Get the last non-zero value of avd
                df['avn'] = df['avd'].where(df['avd'] != 0).ffill().fillna(0)
    
                df['tsl'] = df.apply(lambda row: row['sup'] if row['avn'] == 1 else (row['res'] if row['avn'] == -1 else 0), axis=1)
            
    
    
    
    #             df['green_red'] = 'red'  # Initialize to 'red'
    
    # # Iterate through DataFrame from the second row onwards
    #             for i in range(1, len(df)):
    #                 if df['Close'].iloc[i-1] < df['tsl'].iloc[i-1] and df['Close'].iloc[i] > df['tsl'].iloc[i]:
    #                     df.loc[i, 'green_red'] = 'green'
    
    
    
    
                df['green_red']=0
                for i in range(len(df)):
                    if df['Close'][i-1]<df['tsl'][i-1] and df['Close'][i]>df['tsl'][i]:
                        df['green_red'][i] = 'green'
                    else:
                        df['green_red'][i] = 'red'
    
                    
                
    
    
                # df['green_red'] = np.where(df['Close'] > df['tsl'], 'green', 'red')
    
            
    
            
     
                
    
                dict[f'{ticker}']={'green_red':df['green_red'].tail(1).values[0]}
    
          
            
            
                
            except:
                  continue
            
            
        
        c=pd.DataFrame(dict)
        
        d=c.transpose()
        # print(d)
      
        
        t=d.loc[(d[d.columns[0]]=='green')]
        t.reset_index(inplace=True)
        # print(t)
    
        # t.to_csv(f"strategy1_{end_date}.csv")
    
        x=t['index'].to_list()
        for elm in x:
            greenzone_list.append(elm)
        
    
           
         
    
            
    
    import warnings
    
    # Ignore the SettingWithCopyWarning
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
    
    
    generate_green_signal(overall_indicators_list,start_date=start_date,end_date=end_date)
    
    # print(greenzone_list)
    
    
    
    
    
    
    result=pd.DataFrame(greenzone_list)
    
    
    # z={}
    
    # for ticker in greenzone_list:
    #     df = yf.download(ticker, start=start_date, end=end_date)
    #     df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    #     df['SL']=1.5*df['ATR']
    #     z[f'{ticker}']=round(df['SL'].tail(1),2)
    
    columns = ['Ticker', 'SL_Value','Close']  # Adjust columns as needed
    recommendation = pd.DataFrame(columns=columns)
    
    for ticker in greenzone_list:
        df = yf.download(ticker, start=start_date, end=end_date)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['SL'] = 1.5 * df['ATR']
        sl_value = round(df['SL'].iloc[-1], 2)
        
        
    
        
        # Append a row to the DataFrame
        recommendation = recommendation.append({'Ticker': ticker, 'SL_Value': sl_value,'Close':df['Close'].tail(1)[0]}, ignore_index=True)
    
    
    
    recommendation['SL_Price']=recommendation['Close']-recommendation['SL_Value']
    
    recommendation['Qty(25k)']=25000/recommendation['Close']
    recommendation['Qty(20k)']=20000/recommendation['Close']
    recommendation['Qty(10k)']=10000/recommendation['Close']
    recommendation['Qty(5k)']=5000/recommendation['Close']
    recommendation['Qty(4k)']=4000/recommendation['Close']
    # recommendation=pd.DataFrame(z).transpose()
    
    # print(recommendation)

  
    
    st.dataframe(recommendation)
    
    
    
    
    
    
    
    
    
    
    # recommendation.to_csv(f"signal_green_{current_date}.csv")
    
    
    
    
    
    
    
    
    
    
    # greenzone_list2=greenzone_list[:15]
    
    # print(greenzone_list2)
    # len(greenzone_list2)
    
    # lstm_ticker_list=[]
    # def future_predictions(greenzone_list2, start_date, end_date):
    #     dict={}
    #     def create_sequence(data, time_step):
    #         x, y = [], []
    #         for i in range(len(data) - time_step):
    #             x.append(data[i:(i + time_step)])
    #             y.append(data[i + time_step])
    #         return np.array(x), np.array(y)
    
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     time_step = 60
    #     n_days = 11
    
    #     lstm_units = [40,44,46,48,52,56,60]
        
    #     batch_sizes = [16]
    
    
    #     for ticker in greenzone_list2:
    #         try:
    #             data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    #             scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    #             x_data, y_data = create_sequence(scaled_data, time_step)
    #             x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
    
    #             best_loss = float('inf')
    #             best_units = None
    #             best_batch_size = None
    
    #             for units in lstm_units:
    #                 for batch_size in batch_sizes:
    #                     model = Sequential([
    #                         LSTM(units, return_sequences=True, input_shape=(time_step, 1)),
    #                         LSTM(units, return_sequences=False),
    #                         Dense(1)
    #                     ])
    #                     optimizer = Adam(learning_rate=0.001)
    #                     model.compile(loss='mean_squared_error', optimizer=optimizer)
                        
    #                     early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    #                     history = model.fit(x_data, y_data, epochs=100, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    
    #                     if min(history.history['loss']) < best_loss:
    #                         best_loss = min(history.history['loss'])
    #                         best_units = units
    #                         best_batch_size = batch_size
    
    #             model = Sequential([
    #                 LSTM(best_units, return_sequences=True, input_shape=(time_step, 1)),
    #                 LSTM(best_units, return_sequences=False),
    #                 Dense(1)
    #             ])
    #             optimizer = Adam(learning_rate=0.001)
    #             model.compile(loss='mean_squared_error', optimizer=optimizer)
    #             model.fit(x_data, y_data, epochs=100, batch_size=best_batch_size, verbose=1, callbacks=[early_stop])
    
    #             future_predictions = []
    #             last_sequence = scaled_data[-time_step:].reshape(1, -1, 1)
    #             for _ in range(n_days):
    #                 future = model.predict(last_sequence)
    #                 future_predictions.append(future[0])
    #                 new_seq = np.append(last_sequence[0][1:], future)
    #                 last_sequence = new_seq.reshape(1, -1, 1)
    
    #             future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    #             future_df = pd.DataFrame(future_prices, columns=['future'])
    #             # future_df.to_csv(f"{ticker}.csv")
    #             print(future_df['future'])
    
    
    #             if future_df['future'].iloc[-1]>future_df['future'].iloc[0]:
    #                 dict[ticker]='Buy'
    #             else:
    #                 dict[ticker]='Sell'
                    
             
    
    #         except:
    #             continue
    
          
    #     c=pd.DataFrame(list(dict.items()), columns=['Ticker', 'Signal']) 
    #     d=c.loc[(c['Signal']=='Buy')]
    #     x=d['Ticker'].to_list()
    #     for elm in x:
    #         lstm_ticker_list.append(elm)
    
    # future_predictions(greenzone_list2, start_date=start_date, end_date=end_date)
    
    
    
    # ticker_list
    
    # overall_indicators_list
    
    # greenzone_list2
    
    # lstm_ticker_list2=lstm_ticker_list[:5]
    
    # # recommendation=pd.DataFrame(lstm_ticker_list)
    # # recommendation.rename(columns={0:'stocks'},inplace=True)
    
    # # recommendation
    # # recommendation['datetime']=datetime.datetime.now().replace(second=0, microsecond=0)
    
    # # recommendation
    
    # z={}
    # for ticker in lstm_ticker_list2:
    #     df = yf.download(ticker, start=start_date, end=end_date)
    #     df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    #     df['SL']=1.5*df['ATR']
    #     z[f'{ticker}']=round(df['SL'].tail(1),2)
    
    
    
    
    
    # recommendation=pd.DataFrame(z).transpose()
    # recommendation
    
    # recommendation.reset_index(inplace=True)
    
    # recommendation['CreatedAt']=end_date
    
    
    # recommendation.rename(columns={recommendation.columns.values[0]:'StockSymbol',recommendation.columns.values[1]:'Stoploss'},inplace=True)
    
    # recommendation['Signal']='Buy'
    
    # recommendation['UserId']=0
    # recommendation['StockName']=None
    # recommendation['Weightage']=0
    
    
    
    # recommendation=recommendation[['UserId','StockName','StockSymbol','Signal','CreatedAt','Stoploss','Weightage']]
    
    
    # recommendation['StockSymbol'] = recommendation['StockSymbol']
    
    
    # # import pandas as pd
    # # from sqlalchemy import create_engine
    
    # import pandas as pd
    # from sqlalchemy import create_engine
    
    # # Define the database connection details
    # db_url = 'postgresql://WMS:Neoquant%402023@192.168.1.245:5432/WMS_Development'
    # engine = create_engine(db_url)
    
    
    # path=f"C:\\Users\\NQE00512\\Desktop\\vs\\Final_pipeline\\schedulers\\buy_signal_at_{end_date}.csv"
    
    
    
    # recommendation.to_csv(path)
    
    # recommendation.to_sql('Stock_Recommendations', con=engine, if_exists='append', index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
