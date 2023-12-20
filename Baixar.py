def getEVEinfo(i):
    
    import pandas as pd
    from datetime import date
    
    df = pd.read_csv('TCC_DATA.CSV')
    
    bind = df['bind'][i]
    tstart = df[' TSTART [UT]']
    tfinal = df[' TFINAL [UT]']
    # i += 1
    DATA = str(tstart[i])

    ano_evento = DATA[1:5]
    mes_evento = DATA[6:8]
    dia_evento = DATA[9:11]
    s_RBN = df[' S_RBN [cm^2]'][i]
    
    startHOUR = DATA[12:14]
    stopHOUR = tfinal[i][12:14]

    
    day_of_year = date(int(ano_evento), int(mes_evento), int(dia_evento)).timetuple().tm_yday  
    day_of_year = str(day_of_year)
    DOY = str(day_of_year).rjust(3, '0')
    ano_evento = str(ano_evento)
    # data_final = ano_evento + str(HORA_INICIAL) + str(HORA_FINAL) +  day_of_year
    
    head = 'EVS_L2_'
    
    #EVS_L2_ANODIA_HORA_007.fit.gz

    file = [head + ano_evento + DOY + '_' + startHOUR + '_007_02.fit.gz']
    
    if startHOUR != stopHOUR:
        file2 = head + ano_evento + DOY + '_' + stopHOUR + '_007_02.fit.gz'
        file.append(file2)
        
    # print(file)
    # print(s_RBN)
    # print(bind)
    return file,ano_evento,DOY,startHOUR,stopHOUR

def baixar_arquivo(url, endereco_local):
    import requests
    # faz requisição ao servidor
    resposta = requests.get(url)
    if resposta.status_code == requests.codes.OK:
        with open(endereco_local, 'wb') as novo_arquivo:
            novo_arquivo.write(resposta.content)
        print("Donwload finalizado. Salvo em: {}".format(endereco_local))
    else:
        resposta.raise_for_status()

import re
import os

if __name__ == "__main__":
    direct = r"C:\Users\tvd\Documents\DanielNeri\TCC\Codigos"
    direct = direct.replace(os.sep, '/')

    for i in range(139):
        arquivos =[]
        BASE_URL = 'https://lasp.colorado.edu/eve/data_access/eve_data/products/level2/year/DOY/'
        file, ano_evento, DOY, startHOUR, stopHOUR = getEVEinfo(i)
        BASE_URL = re.sub(r'year',ano_evento,BASE_URL)
        BASE_URL = re.sub(r'DOY',DOY,BASE_URL)
        if startHOUR != stopHOUR:
            arquivo_1 = BASE_URL + str(file[0])
            arquivo_2 = BASE_URL + str(file[1])
            arquivos = [arquivo_2, arquivo_2]
        else:
            arquivo_1 = BASE_URL + str(file[0])
            arquivos = [arquivo_1]

        # print(arquivos)
        for j in range(len(arquivos)):
            baixar_arquivo(arquivos[j],os.path.join(direct, os.path.basename(arquivos[j])))
