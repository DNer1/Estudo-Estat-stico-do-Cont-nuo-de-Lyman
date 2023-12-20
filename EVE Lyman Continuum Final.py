# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:30:52 2022

@author: 1159275
"""

import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
# import sunpy

###############################################################################

def read_eve(files):
    """
    read_eve

    Parameters
    ----------
    files : EVE FITS file
        
    Returns
    -------
    wav : TYPE
        DESCRIPTION.
    sod : TYPE
        DESCRIPTION.
    irr : TYPE
        DESCRIPTION.

    """

    # import astropy.units as u
    from sunpy.time import parse_time
    # from astropy.time import TimeDelta
    
    sod = np.array([])
    irr = np.array([])
    # tim = np.array([])
    
    for i in files:
        hdu = fits.open(i)
        ## TAI=tempo (formato TAI)
        # taii = hdu[3].data['TAI']
        ## data
        # dat = hdu[3].data['YYYYDOY']
        # print(dat)
        ## SOD = seconds of day (tempo)
        sodi = hdu[3].data['SOD']
        ## array de wavelength (comprimento de onda)
        wav = hdu[1].data['WAVELENGTH']
        ## array do espectro (irradiance W/m2/nm)
        irri = hdu[3].data['IRRADIANCE'] # time,irr (360,5200)
        hdu.close()
        
        start_time = parse_time(hdu[3].header['DATE_OBS'])
        # timi = start_time + TimeDelta(hdu[1].data['SOD']*u.second)
        
        ## concatenando dados    
        # tim = np.vstack([tim,timi]) if tim.size else timi
        sod = np.vstack([sod,sodi]) if sod.size else sodi
        irr = np.vstack([irr,irri]) if irr.size else irri
        
        # *1e3 para mudar de W/m2/nm para mW/m2/nm
        
    return wav,sod,irr,start_time

###############################################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

###############################################################################
       
def simple_plot(wav,irr,val):
    idx = find_nearest(wav, val)
    fig = plt.figure()
    plt.plot(irr[:,idx])

###############################################################################

def pltspec(wav,irr):
    plt.semilogy(wav,irr[0,:],linewidth=1)
    plt.ylim(1e-7,1e-2)
    plt.xlim(32,105)
    plt.grid()
    plt.ylabel(r'Irradiância [W/m$^2$/nm]')
    plt.xlabel('Comprimento de onda [nm]')

###############################################################################


def planck(wav,Ts,b1):
    from astropy import constants as cte
    import astropy.units as u
      
    h = cte.h.cgs
    c = cte.c.cgs
    kb = cte.k_B.cgs
    w = wav/1e9 * u.m
    T = Ts * u.K
    I = 2*h*c**2/w**5/np.exp(h*c/w/kb/T-1)/u.sr
    # print(I)
    I = I.to(u.erg/u.cm/u.cm/u.s/u.Angstrom/u.sr)/b1
    # print(I[0].unit)
    
    return I.value

###############################################################################

def getEVEinfo(i):
    
    import pandas as pd
    from datetime import date
    
    # df = pd.read_csv('TCC_DATA.CSV')
    # df = pd.read_csv("Dados-TCC.csv")

    df_n = pd.read_csv("Relação-Neri-Kazachenko.csv")

    df_k = pd.read_csv("ribbondb_v1.0.csv")

    df = pd.merge(df_n, df_k, left_on='INDEX KAZACKENKO', right_on='INDEX', how='inner')
    
    bind = df['bind'][i]
    tstart = df[' TSTART [UT]_y']
    tfinal = df[' TFINAL [UT]_y']
    # i += 1
    DATA = str(tstart[i])

    ano_evento = DATA[1:5]
    mes_evento = DATA[6:8]
    dia_evento = DATA[9:11]
    s_RBN = df[' S_RBN [cm^2]'][i]
    
    startHOUR = DATA[12:14]
    stopHOUR = tfinal[i][12:14]

    inicioEVE = DATA[12:17]
    
    day_of_year = date(int(ano_evento), int(mes_evento), int(dia_evento)).timetuple().tm_yday  
    day_of_year = str(day_of_year)
    DOY = str(day_of_year).rjust(3, '0')
    ano_evento = str(ano_evento)
    data_final = ano_evento + '-' + mes_evento + '-' + dia_evento
    
    head = 'EVS_L2_'
    
    #EVS_L2_ANODIA_HORA_007.fit.gz

    file = [head + ano_evento + DOY + '_' + startHOUR + '_007_02.fit.gz']
    
    if startHOUR != stopHOUR:
        file2 = head + ano_evento + DOY + '_' + stopHOUR + '_007_02.fit.gz'
        file.append(file2)
        
    print(file)
    # print(s_RBN)
    # print(bind)
    return file, s_RBN, bind, inicioEVE, data_final


###############################################################################


import csv

header = ['ID','Data do Evento','Hora Inicial','Área Evento [cm^2]','Tc (Sol Calmo)','Incerteza de Tc','b1 (Sol Calmo)','Incerteza de b1',
          'Fluxo Médio em QS a 91 nm','Tc (flare)',
          'Incerteza de Tc','b1 (flare)','Incerteza de b1','Fluxo Médio em Flare a 91 nm','Irradiação Maxima','3 Sigma']
          

erro = []
##=============================================================================
##=============================================================================
##=============================================================================
Machado = 0

## EVENTO
# i = -5
with open('complete_tcc_sim_fixed_selected_data_24_11.csv','w',newline='') as f:
    escrever = csv.writer(f)
    escrever.writerow(header)
    # range(139)
    # [9, 21, 30, 37, 40, 42, 44, 45, 55, 60, 62, 64, 65, 
    # 67, 69, 74, 75, 77, 78, 79, 83, 93, 104, 107, 112, 135, 136]
    for i in [9, 21, 30, 37, 40, 42, 44, 45, 55, 60, 62, 64, 65, 
    67, 69, 74, 75, 77, 78, 79, 83, 93, 104, 107, 112, 135, 136]:

        try:
            i = i - 1
            infos_eve = []
            file,s_RBN,bind, inicioEVE, data_final = getEVEinfo(i) ## retorna indices do background e FITS
            print(s_RBN)
            n_eve = str(i+1)
            infos_eve.append(n_eve)
            ## SEXTO EVENTO DO MACHADO
            # path0 = r'C:\Users\1159275\Dropbox\Mackenzie\Dell\IC\EVE\Neri\eve_test_click\Teste'
            # files = [path0+'\EVS_L2_2017249_11_007_02.fit.gz'
            #           ,path0+'\EVS_L2_2017249_12_007_02.fit.gz']
            # bind = [0,250]
            
            ##=============================================================================
            ##=============================================================================
            ##=============================================================================
            
            wav,sod,irr,start_time = read_eve(file)

            # start_time_str = str(start_time)
            # data_eve = start_time_str[0:10]
            # hora_ini = start_time_str[11:23]
            
            infos_eve.append(data_final)
            infos_eve.append(inicioEVE)
            infos_eve.append(np.round(s_RBN,2))
            
            # print('====================================')
            # print(inicioEVE)
            # print('====================================')
            
            from astropy import constants as cte
            # import astropy.units as u
            ## conversao F (W/m2/nm) p/ I (erg/s/cm2/A/ster)
            # flare_area = s_RBN ## suposição!
            # flare_area = 3.5e18 ## suposição!
            flare_area = 1e18 ## suposição!
            fac = 100/(flare_area/cte.au.cgs.value**2)
            facsun = np.pi*100/(np.pi*cte.R_sun.cgs.value**2/cte.au.cgs.value**2)
            
            val1 = 91
            # val2 = 97.7
            # simple_plot(wav,irr,val)
            
            ## intervalo do pre-flare (background), escolhido 
            bind = bind[1:-1].split()
            bkgt = [int(bind[0]),int(bind[1])] #[0,250]
            bkgt = np.array(bkgt)
            # print(len(irr))
            # print(bind)
            # print(bkgt)
            # bkgt = getBKGindex(event_index)
           
            
            bkg = irr[bkgt[0]:bkgt[1],:].mean(axis=0)     # <--------Dando erro
            sig = irr[bkgt[0]:bkgt[1],:].std(axis=0)
            sig_f = np.std(irr[bkgt[0]:bkgt[1],:])
            # print(bkg.shape)
            # print(bkg)
            print(sig)

            idx1 = find_nearest(wav, val1)
            # idx2 = find_nearest(wav, val2)
            ## encontra o máximo do tempo do Ly-C
            maxidx = irr[:,idx1].argmax()
            width = 2
            
            ## LIGHTCURVE
            fig = plt.figure()
            
            plt.plot(irr[:,idx1]*1e3,label=r'$\lambda = $'+str(wav[idx1])+'nm',linewidth=1)
            plt.plot([0,sod.size],[bkg[idx1]*1e3,bkg[idx1]*1e3],'--k',label='F quiet Sun')
            
            plt.plot((irr[:,idx1]-bkg[idx1])*1e3,label=r'$\lambda = $'+str(wav[idx1])+'nm',linewidth=1)
            plt.ylim(0,irr[:,idx1].max()*1e3)
            
            # plt.plot(irr[:,idx2]*1e3,label=r'$\lambda = $'+str(wav[idx2])+'nm',linewidth=1)
            # plt.plot([0,sod.size],[bkg[idx2]*1e3,bkg[idx2]*1e3],'--k',label='F quiet Sun')
            
            plt.grid()
            plt.legend()
            plt.ylabel(r'Irradiância [mW/m$^2$/nm]')
            plt.xlabel('tempo [índices]')
            plt.axvline(maxidx-width,alpha=0.5,color='red')
            plt.axvline(maxidx+width,alpha=0.5,color='red')
            # plt.savefig(r'C:/Users/tvd/Documents/DanielNeri/TCC/Imagens/Lightcurve/'+ n_eve + '_Flare.png')
            plt.close(fig)
            
            ## SPECTRA
            fig = plt.figure()
            plt.semilogy(wav,bkg,linewidth=1,label='quiet Sun')
            # plt.semilogy(wav,irr[maxidx,:]-bkg,linewidth=1,label='excesso max. flare (10s)')
            plt.semilogy(wav,irr[maxidx-width:maxidx+width,:].mean(axis=0)-bkg,linewidth=1,label='excesso max. flare (50s)')
            # plt.ylim(1e-7,1e-2)
            # plt.ylim(1e-7,1e-2)
            # plt.xlim(60,94)
            plt.grid()
            plt.legend()
            plt.ylabel(r'Irradiância [W/m$^2$/nm]')
            plt.xlabel('Comprimento de onda [nm]')
            plt.axvline(wav[idx1],alpha=0.5,color='red')
            # plt.savefig(r'C:/Users/tvd/Documents/DanielNeri/TCC/Imagens/Spectra/'+ n_eve + '_Spectra.png')
            plt.close(fig)
            
            #########################################
            
            
            id1 = find_nearest(wav, 60)
            id2 = find_nearest(wav, 94)
            pp = np.arange(id1,id2)
            
            # flare
            flr = (irr[maxidx-width:maxidx+width,pp].mean(axis=0)-bkg[pp])*fac
            sigf = sig[pp]*fac
            
            wav = wav[pp]
            bkg = bkg[pp]*facsun
            sigs = sig[pp]*facsun
            
            flr_med = sum(flr)/len(flr)
            bkg_med = sum(bkg)/len(flr)
            
            ## apenas continuo
            f1 = find_nearest(wav, 72.37)
            f2 = find_nearest(wav, 74.74)
            p0 = np.arange(f1,f2)
            
            f1 = find_nearest(wav, 79.2)
            f2 = find_nearest(wav, 83)
            p1 = np.arange(f1,f2)
            
            f3 = find_nearest(wav, 83.75)
            f4 = find_nearest(wav, 90.22)
            p2 = np.arange(f3,f4)
            
            f5 = find_nearest(wav, 90.70)
            f6 = find_nearest(wav, 91.40)
            p3 = np.arange(f5,f6)
            
            if Machado == 1:
                pf = np.concatenate([p1,p2,p3]) ## range do Machado
            else:
                pf = np.concatenate([p0,p1,p2,p3]) ## range extendida
                
            
            ###################################################
            if(np.isnan(flr[pf]).any()):
                continue
            else:
            
                def fitPlanck(w,I,sigI):
                    from scipy.optimize import curve_fit
                    x = curve_fit(planck, xdata = w, ydata = I,p0 = [1e4,100],sigma=sigI)
                    return x
                
                # flare
                fpar,fcov = fitPlanck(wav[pf],flr[pf],sigf[pf])
                ferr = np.sqrt(np.diag(fcov))
                # pre-flare
                spar,scov = fitPlanck(wav[pf],bkg[pf],sigs[pf])
                serr = np.sqrt(np.diag(scov))
                
                xlim = (60,94)
                #################### QUIET SUN ##################
                fig = plt.figure()
                plt.semilogy(wav,bkg,linewidth=1,label='quiet Sun',color='k')
                plt.fill_between(wav, bkg-sigs, bkg+sigs,color='gray')
                plt.ylim(1,1e3)
                plt.xlim(xlim)
                plt.grid()
                plt.ylabel(r'Intensidade específica [erg/s/cm$^2$/$\AA$/ster]')
                plt.xlabel('Comprimento de onda [nm]')
                
                c1 = 'C0'
                plt.plot(wav,planck(wav, spar[0], spar[1]),c1,alpha=0.5,linewidth=5)
                plt.plot(wav,planck(wav, spar[0], spar[1]),c1,alpha=0.9,linewidth=1,label='model')
                # plt.plot(wav,bkg,'r',alpha=0.5)
                
                plt.plot(wav,planck(wav, spar[0], spar[1]-serr[1]),c1,alpha=0.9,linewidth=1)
                plt.plot(wav,planck(wav, spar[0], spar[1]+serr[1]),c1,alpha=0.7,linewidth=1)
                plt.title(start_time.value)
                
                if Machado == 1: 
                    pf = [p1,p2,p3]
                else:
                    pf = [p0,p1,p2,p3]
                    
                    
                for i in pf:
                    plt.plot(wav[i],bkg[i],'r',alpha=0.5)
                    
                plt.text(82,2,'T='+str(np.round(spar[0]))+r'$\pm$'+str(np.round(serr[0]))+'K'+'\n'+'b1='+str(np.round(spar[1]))+r'$\pm$'+str(np.round(serr[1])))
                plt.legend(loc='upper left')
                # plt.savefig(r'C:/Users/danin/Documents/Daniel/IC/Novo Script/Imagens/'+ n_eve + '_QS.png')
                plt.close(fig)
                
                infos_eve.append(str(np.round(spar[0],2)))
                infos_eve.append(str(np.round(serr[0],2)))
                
                infos_eve.append(str(np.round(spar[1],2)))
                infos_eve.append(str(np.round(serr[1],2)))
                
                infos_eve.append(np.round(bkg_med,2))
                
                #################### FLARE ##################
                fig = plt.figure()
                plt.semilogy(wav,flr,linewidth=1,label='flare',color='k')
                plt.fill_between(wav, flr-sigf, flr+sigf,color='gray')
                plt.ylim(1e3,1e8)
                plt.xlim(xlim)
                plt.grid()
                plt.ylabel(r'Intensidade específica [erg/s/cm$^2$/$\AA$/ster]')
                plt.xlabel('Comprimento de onda [nm]')
                plt.title(start_time.value)
                # pf = [p0,p1,p2,p3]
                for i in pf:
                    plt.plot(wav[i],flr[i],'r',alpha=0.5)
                
                
                plt.plot(wav,planck(wav, fpar[0], fpar[1]),c1,alpha=0.5,linewidth=5)
                plt.plot(wav,planck(wav, fpar[0], fpar[1]),c1,alpha=0.9,linewidth=1,label='model')
                
                plt.plot(wav,planck(wav, fpar[0], fpar[1]-ferr[1]),c1,alpha=0.9,linewidth=1)
                plt.plot(wav,planck(wav, fpar[0], fpar[1]+ferr[1]),c1,alpha=0.7,linewidth=1)
                
                plt.text(72,2e7,'T='+str(np.round(fpar[0]))+r'$\pm$'+str(np.round(ferr[0]))+'K'+'\n'+'b1='+str(np.round(fpar[1],2))+r'$\pm$'+str(np.round(ferr[1],2)))
                plt.legend(loc='upper left')
                plt.savefig(r'C:/Users/tvd/Documents/DanielNeri/TCC/Imagens/Espectro_Flare/'+ n_eve + '_Flare.png')
                plt.close(fig)
                
                infos_eve.append(str(np.round(fpar[0],2)))
                infos_eve.append(str(np.round(ferr[0],2)))
                
                infos_eve.append(str(np.round(fpar[1],2)))
                infos_eve.append(str(np.round(ferr[1],2)))
                
                infos_eve.append(np.round(flr_med,2))
                infos_eve.append(np.round(max(flr),2))
                infos_eve.append(np.round(3.0*sig_f,2))

                # print(s_RBN)
                print('---------------------------')              
                escrever.writerow(infos_eve)       
        except:
            erro.append(i)
            # print(erro)