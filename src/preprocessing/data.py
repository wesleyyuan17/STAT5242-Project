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
        
        self.features = df.copy()
        if technicals is not None:
            for k, v in technicals.items():
                self.features[k] = v(df)
        self.features.set_index(['timestamp', 'Asset_ID'], inplace=True)
        for col in ['timestamp', 'Asset_ID', 'Asset_Name', 'Target']:
            if col in self.features.columns:
                self.features.drop(col, axis=1, inplace=True)
                
        self.targets = df.set_index(['timestamp', 'Asset_ID'])['Target']
        
        self.seq_len = seq_len
        self.dates = sorted(df['timestamp'].unique())
        self.num_valid_starts = len(self.dates) - self.seq_len
    
    def __len__(self):
        return self.df.shape[0]
    
    def __iter__(self):
        for i in range(self.num_valid_starts):
            date_idx = self.dates[i:i+self.seq_len]
            features = self.features.loc[date_idx].values
            target = self.targets.loc[date_idx].values
            yield features, target


def get_crypto_dataset():
    data = pd.read_csv('../data/g-research-crypto-forecasting/train.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    asset_details = pd.read_csv('../../data/g-research-crypto-forecasting/asset_details.csv')
    id_to_names = dict(zip(asset_details['Asset_ID'], asset_details['Asset_Name']))
    data['Asset_Name'] = [id_to_names[a] for a in data['Asset_ID']]

    dataset = CryptoFeed(data)
    return dataset