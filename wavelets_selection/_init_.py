#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pywt
import pywt.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


__path__ = __import__('pkwavelet').extend_path(__path__, __name__)


# In[10]:


def wavelets_selection(time_series):
   
    discrete_wavelets=pywt.wavelist(kind='discrete')

    numero_wavelet = len(discrete_wavelets)
    
    vetor_rmse = [0] * numero_wavelet

    vetor_pearson_r = [0] * numero_wavelet

    contador = 0

    np.set_printoptions(suppress=True)

    
    for waveletname in discrete_wavelets:
        
        coefficients_level1 = pywt.wavedec(time_series, waveletname, 'smooth', level=1)
        [cA1_l1, cD1_l1] = coefficients_level1
        
        approx_coeff_level1_only = [cA1_l1, None]


        rec_signal_cA_level1 = pywt.waverec(approx_coeff_level1_only, waveletname, 'smooth')

        sinal_aproximacao_wavelet = np.array(rec_signal_cA_level1)    
 
        vetor_rmse[contador] = round(mean_squared_error(time_series, sinal_aproximacao_wavelet, squared=False),7)
        
        vetor_pearson_r[contador] = round(pearsonr(time_series, sinal_aproximacao_wavelet)[0],7)
    
        contador = contador + 1
   
    menor_rmse = min(vetor_rmse)

    posicao_minimo_rmse = vetor_rmse.index(menor_rmse)

    select_name = discrete_wavelets[posicao_minimo_rmse]

    tabela_info_rmse_pearson = []
    for i in range(0,numero_wavelet):
        tabela_info_rmse_pearson.append([discrete_wavelets[i], vetor_rmse[i],vetor_pearson_r[i]])

    tabela_rmse_pearson = pd.DataFrame(tabela_info_rmse_pearson, columns =["Wavelet", "RMSE", "PEARSON"])

    wave_metrics = tabela_rmse_pearson.sort_values(by='RMSE').reset_index(drop=True)
    
    coefficients_level1 = pywt.wavedec(time_series, select_name, 'smooth', level=1)
    [cA1_l1, cD1_l1] = coefficients_level1
    approx_coeff_final = [cA1_l1, None]
        
    rec_signal_cA_final = pywt.waverec(approx_coeff_final, select_name, 'smooth')
    
      
    fig, ax = plt.subplots(figsize=(20,8))
    ax.plot(time_series, label='SINAL',color="blue")
    ax.plot(rec_signal_cA_level1, label='Nivel 1 reconstruído', linestyle='dotted',color="orange")
    ax.legend(loc='upper right')
    ax.set_title('Reconstrução simples ('+ select_name+ ')', fontsize=20)
    ax.set_xlabel('Tempo', fontsize=16)
    ax.set_ylabel('Amplitude', fontsize=16)
    plt.show()
    return  wave_metrics


# In[ ]:




