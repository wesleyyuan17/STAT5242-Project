import pandas as pd
import os

from torch.utils.data import IterableDataset

LOCAL_PATH_TO_DIR = '~/Documents/Academics/Columbia/2021/2021 Fall/Advanced ML/Final Project'

class CryptoFeed(IterableDataset):
    def __init__(self, df, seq_len=5, technicals=None):
        """
        Creates an iterable feed of crypto market states increasing in time given an input df

        Args:
            df: pandas Dataframe, contains price data on crypto assets. Assumes 
            technicals: dict, string (key) mapped to function (value) that calculates technical indicator from df
        """
        df.sort_values(['timestamp', 'Asset_ID'], inplace=True)
        
        self.id_to_name = dict(zip(df['Asset_ID'], df['Asset_Name']))
        self.data = [df[df['Asset_ID'] == i].copy() for i in sorted(df['Asset_ID'].unique())]
        self.targets = pd.concat([tdf.set_index('timestamp')['Target'] for tdf in self.data], axis=1)

        if technicals is not None:
            for tdf in self.data:
                for k, v in technicals.items():
                    tdf[k] = v(tdf)
        for tdf in self.data:
            tdf.set_index('timestamp', inplace=True)
        for col in ['timestamp', 'Asset_ID', 'Asset_Name', 'Target']:
            for tdf in self.data:
                if col in tdf.columns:
                    tdf.drop(col, axis=1, inplace=True)

        self.features = pd.concat(self.data, axis=1)

        # only use last 100k
        self.features = self.features.iloc[-100000:]
        self.targets = self.targets.iloc[-100000:]
                        
        self.seq_len = seq_len
        self.valid_dates = list(self.features.index)
        # self.num_valid_starts = self.features.shape[0] - self.seq_len
    
    def __len__(self):
        return self.features.shape[0]
    
    def __iter__(self):
        for i in range(self.seq_len, len(self.valid_dates)):
            dates_idx = self.valid_dates[i-self.seq_len:i]
            features = self.features.loc[dates_idx].values
            target = self.targets.loc[dates_idx[-1]].values # target is target of end of window
            adj = self.targets.loc[dates_idx].corr().fillna(value=0).values # correlation matrix between previous seq_len target values
            yield features, target, adj


def get_crypto_dataset(seq_len=5, technicals=None):
    file_path = os.path.join(LOCAL_PATH_TO_DIR, 'data/g-research-crypto-forecasting/train.csv')
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    file_path = os.path.join(LOCAL_PATH_TO_DIR, 'data/g-research-crypto-forecasting/asset_details.csv')
    asset_details = pd.read_csv(file_path)
    id_to_names = dict(zip(asset_details['Asset_ID'], asset_details['Asset_Name']))
    data['Asset_Name'] = [id_to_names[a] for a in data['Asset_ID']]
    data.fillna(method='ffill', inplace=True)
    data.fillna(value=0, inplace=True)

    dataset = CryptoFeed(data, seq_len, technicals)
    return dataset