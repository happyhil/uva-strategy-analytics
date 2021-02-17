#!/usr/bin/env python


import pandas as pd


def main():
    """
    Sources:
    - dfcountyresult: https://www.kaggle.com/unanimad/us-election-2020
    - dfcountydemos: https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-detail.html
    """

    # read in
    dfcountyresult = pd.read_csv('datasets/usa_president_county_candidate_2020.csv')
    dfcountydemos = pd.read_csv('datasets/usa_county_demographics_2019.csv')

    dimensions = ['statename', 'countyname']
    agg_dict_totals = {
        'tot_pop': 'sum',
        'tot_male': 'sum',
        'tot_female': 'sum',
        'wa_male': 'sum',
        'wa_female': 'sum',
        'ba_male': 'sum',
        'ba_female': 'sum'
    }

    # basis preps
    dfcountyresult = dfcountyresult.rename(columns={
        'state': 'statename',
        'county': 'countyname',
    })

    dfcountydemos.columns = [c.lower() for c in dfcountydemos.columns]
    dfcountydemos = dfcountydemos.rename(columns={
        'stname': 'statename',
        'ctyname': 'countyname',
    })
    dfcountydemos19 = dfcountydemos.loc[lambda x: x['year']==12].copy() # 12 = 7/1/2019 population estimate
    demos_counties = list(set(dfcountydemos19['countyname'].values))
    result_counties = list(set(dfcountyresult['countyname'].values))
    not_matching_counties = [x for x in result_counties if x not in demos_counties]
    print(f'-- number of non matching counties: {len(not_matching_counties)}')
    dfcountydemos19_totals = dfcountydemos19.loc[lambda x: (x['agegrp']>=4)].groupby(dimensions, as_index=False).agg(agg_dict_totals)
    dfcountydemos19_young = dfcountydemos19.loc[lambda x: (x['agegrp']>=4) & (x['agegrp']<=8)].groupby(dimensions, as_index=False).agg(agg_dict_totals)
    dfcountydemos19_young.columns = [f'{y}_young' if y not in dimensions else y for y in dfcountydemos19_young.columns]
    dfcountydemos19_full = dfcountydemos19_totals.merge(dfcountydemos19_young, how='inner', on=dimensions)

    # features creation, merge & selection
    dfcountydemos19_full['yougn'] = dfcountydemos19_full['tot_pop_young'] / dfcountydemos19_full['tot_pop']
    dfcountydemos19_full['female'] = dfcountydemos19_full['tot_female'] / dfcountydemos19_full['tot_pop']
    dfcountydemos19_full['black'] = (dfcountydemos19_full['ba_male'] + dfcountydemos19_full['ba_female']) / dfcountydemos19_full['tot_pop']
    dfcountydemos19_select = dfcountydemos19_full[['statename', 'countyname', 'tot_pop', 'yougn', 'female', 'black']].copy()
    dfcountyresult_select = dfcountyresult.loc[lambda x: x['party'].isin(['DEM', 'REP'])].drop(columns=['party'])
    dfcountyresult_wintrump = dfcountyresult_select.loc[lambda x: (x['won']==True) & x['candidate'].str.contains('Trump')]
    dfcountyresult_winbiden = dfcountyresult_select.loc[lambda x: (x['won']==True) & x['candidate'].str.contains('Biden')]
    dfcounty_winner = pd.concat([dfcountyresult_wintrump, dfcountyresult_winbiden])
    dfcounty_winner = dfcounty_winner[['statename', 'countyname', 'candidate']].rename(columns={'candidate': 'winner'})
    dfcountyresult_votes = dfcountyresult.groupby(dimensions, as_index=False)[['total_votes']].sum()
    dfcountyresult_final = dfcountyresult_votes.merge(dfcounty_winner)
    dfsubfinal = dfcountyresult_final.merge(dfcountydemos19_select, how='inner', on=dimensions)
    dfsubfinal['turnout'] = dfsubfinal['total_votes'] / dfsubfinal['tot_pop']
    dffinal = dfsubfinal.drop(columns=['total_votes'])

    # store result
    dffinal.to_csv('datasets/usa_election_dataset.csv', index=False)

    # selection diff check
    df_statetotal = dfcountydemos19_totals.groupby('statename', as_index=False)[['tot_pop']].sum()
    dffinal_statetotal = dffinal.groupby('statename', as_index=False)[['tot_pop']].sum()
    dffinal_statetotal = dffinal_statetotal.rename(columns={'tot_pop': 'tot_pop_select'})
    dfpopdiff = df_statetotal.merge(dffinal_statetotal, how='left').fillna(0)
    dfpopdiff['select_percent'] = dfpopdiff['tot_pop_select'] / dfpopdiff['tot_pop']
    print(f"-- population in final: {round(dfpopdiff['tot_pop_select'].sum() / dfpopdiff['tot_pop'].sum(), 3)*100}%")
    print('-- counties with less than 99% of population selected:')
    print(dfpopdiff.loc[lambda x: x['select_percent']<.99])

    # states diff check
    cperstate_demos = dfcountydemos19_select.groupby('statename', as_index=False)[['countyname']].nunique()
    cperstate_demos.columns = ['statename', 'n_counties_dems']
    cperstate_results = dfcountyresult_select.groupby('statename', as_index=False)[['countyname']].nunique()
    cperstate_results.columns = ['statename', 'n_counties_results']
    cperstate_total = cperstate_demos.merge(cperstate_results)
    cperstate_total['diff'] = cperstate_total['n_counties_results'] - cperstate_total['n_counties_dems']
    print('-- states with more than 2 countries difference with regards to count of counties / state:')
    print(cperstate_total.loc[lambda x: x['diff']>2])


if __name__ == '__main__':
    main()
