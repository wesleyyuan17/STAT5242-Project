import pandas as pd

from torch.utils.data import IterableDataset


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
                        
        self.seq_len = seq_len
        self.valid_dates = list(self.features.index)
        self.num_valid_starts = self.features.shape[0] - self.seq_len
    
    def __len__(self):
        return self.df.shape[0]
    
    def __iter__(self):
        for i in range(self.num_valid_starts):
            dates_idx = self.valid_dates[i:i+self.seq_len]
            features = self.features.loc[dates_idx].values
            target = self.targets.loc[dates_idx[-1]].values # target is target of end of window
            yield features, target


def get_crypto_dataset():
    data = pd.read_csv('../data/g-research-crypto-forecasting/train.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    asset_details = pd.read_csv('../../data/g-research-crypto-forecasting/asset_details.csv')
    id_to_names = dict(zip(asset_details['Asset_ID'], asset_details['Asset_Name']))
    data['Asset_Name'] = [id_to_names[a] for a in data['Asset_ID']]

    dataset = CryptoFeed(data)
    return dataset