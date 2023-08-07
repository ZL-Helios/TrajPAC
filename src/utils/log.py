
import os
import pandas as pd


class Logger:
    
    def __init__(self, dir, name='log', ext='results', **kwargs):
        
        file = name + '_' + ext + '.xlsx'
        self.path = os.path.join(dir, file)
        if os.path.exists(self.path):
            try:
                self.df = pd.read_excel(self.path, **kwargs)
            except:
                self.df = pd.DataFrame()
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.df = pd.DataFrame()


    def append(self, df, **kwargs):
        self.df = pd.concat([self.df, df], **kwargs)


    def update(self, df, **kwargs):
        self.df = pd.concat([self.df, df], **kwargs)
        self.df = self.df.round(15)     # excel has max float precision 15
        self.df.sort_values(by=list(self.df.columns.values), inplace=True, ascending=True)
        self.df.drop_duplicates(inplace=True, ignore_index=True)
        #for col in self.df.columns.values:
        #    print(self.df[col].duplicated())


    def save(self, **kwargs):
        if os.path.exists(self.path):
            with pd.ExcelWriter(self.path, mode='a', if_sheet_exists='replace') as writer:
                self.df.to_excel(writer, index=False, **kwargs)
        else:
            self.df.to_excel(self.path, index=False, **kwargs)
