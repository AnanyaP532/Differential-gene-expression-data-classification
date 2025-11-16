#load data and pre-processing
#python src/preprocess.py --input data/GEO_dataset.csv --output results/processed.csv

import pandas as pd
import argparse
import os

def load_and_preview(path, n=5):
    df = pd.read_csv(path)
    print(df.head(n))
    print('\nShape:', df.shape)
    return df
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        load_and_preview(sys.argv[1])
    else:
        print('Provide path to GEO_dataset.csv')
        
def preprocess(infile, outfile):
    df = pd.read_csv(infile)
    df = df.dropna(axis=1, how='all')
    if 'target' not in df.columns:
        raise ValueError('Expected a "target" column (0/1) in the input CSV.')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.lower().map({'male':1,'m':1,'female':0,'f':0}).fillna(-1).astype(int)
    if 'Samples' in df.columns:
        df = df.drop(columns=['Samples'])
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f'Processed data written to {outfile}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess GEO dataset CSV')
    parser.add_argument('--input', required=True, help='Input CSV path')
    parser.add_argument('--output', required=True, help='Output processed CSV path')
    args = parser.parse_args()
    preprocess(args.input, args.output)
