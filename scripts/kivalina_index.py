'''
Created on Jun 16, 2023

@author: simon
'''
import pandas as pd
import numpy as np
import datetime

from forcing import load_forcing_merra_subset, parse_dates, ind_TDD_exceedance

def kivalina_forcing(folder_forcing, year, remove_last=True):
    df = load_forcing_merra_subset(folder_forcing)
    d0 = {2019: '2019-05-14', 2018: '2018-05-18'}[year]
    d1 = {2019: '2019-09-15', 2018: '2018-09-15'}[year]
    datesstr = {2019:
                 ('20190606', '20190618', '20190630', '20190712', '20190724', '20190805',
                  '20190817', '20190829', '20190910'),
                2018:
                 ('20180611', '20180623', '20180705', '20180717', '20180729', '20180810', '20180822', 
                  '20180903')}
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    if remove_last: ind_scenes = ind_scenes[:-1]    
    dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    return dailytemp, ind_scenes

def indranges_kivalina(dailytemp, ind_scenes, TDD=[900, 1000]):
    inds0 = ind_TDD_exceedance(dailytemp, TDD)
    inds0n = [f'TDD{tdd}' for tdd in TDD]
    inds1 = (len(dailytemp) - 1, ind_scenes[-1])
    inds1n = ['lastday', 'lastscene']
            
    indranges_dict = {f'{ind0n}_{ind1n}': (ind0, ind1) 
                      for ind0, ind0n in zip(inds0, inds0n) for ind1, ind1n in zip(inds1, inds1n)}
    
    return indranges_dict



if __name__ == '__main__':
    scenarios = {'2019': (2019, False), '2019r': (2019, True), '2018': (2018, False)}
    TDD = (850, 950)
    # folder_forcing = '/10TBstorage/Work/gie/forcing/kivalina'
    folder_forcing = '/home/simon/Work/gie/forcing/kivalina'

    scenario = '2019r'
    year, remove_last = scenarios[scenario]
    
    dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year, remove_last=remove_last)
    indranges_dict = indranges_kivalina(dailytemp, ind_scenes, TDD=TDD)
    indranges_names, indranges = tuple(indranges_dict.keys()), tuple(indranges_dict.values())
    
    # save indranges_names in textfile
    
    # read data
    # 2019: unw_corr
    
    # apply nugget
    # add new covariance model: two options; all reference point or number 10 (lon: -164.56111)
    # check whether 2018 needs different references: store corrected unw_corr
    # include indrange in inference
    