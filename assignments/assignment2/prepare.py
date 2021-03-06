#!/usr/bin/env python3


import csv
import os
import pandas as pd


def main():

    for _, _, files in os.walk('data/raw/'):
        pass

    dfs = []
    for y in [x.split('-')[-1][:2] for x in files]:
        subdf = pd.read_csv(f'data/raw/dataset-of-{y}s.csv')
        subdf['year'] = f'{y}s'
        dfs.append(subdf)

    dffull = pd.concat(dfs)
    dffull.to_csv('data/prepped/spotify_full.csv',
                  index=False,
                  quoting=csv.QUOTE_NONNUMERIC,
                  quotechar='"')

    features = [
        'target',
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms',
        'time_signature',
        'chorus_hit',
        'sections',
        'n_artist_tracks'
    ]
    dffull['n_artist_tracks'] = dffull.groupby('artist')['track'].transform('count')
    dfselect = dffull[features]
    dfselect.to_csv('data/prepped/spotify_featured.csv',
                    index=False,
                    quoting=csv.QUOTE_NONNUMERIC,
                    quotechar='"')


if __name__ == '__main__':
    main()
