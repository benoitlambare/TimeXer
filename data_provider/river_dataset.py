import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class RiverForecastDataset(Dataset):
    def __init__(self, df, input_window, horizon):
        arr = df[
            [
                "tp",
                "t2m",
                "e",
                "sro",
                "swvl1",
                "Value",
            ]
        ].to_numpy()
        self.data = (arr - arr.mean(axis=0)) / arr.std(axis=0)
        self.W = input_window
        self.H = horizon

    def __len__(self):
        return len(self.data) - self.W - self.H + 1

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.W, :-1]
        y = self.data[idx + self.W : idx + self.W + self.H, -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def data_provider(args):
        df = pd.read_csv(args.data_path, parse_dates=["Time"])
        dataset = RiverForecastDataset(
            df, input_window=args.input_len, horizon=args.pred_len
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return loader
