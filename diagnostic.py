import numpy as np
import pandas as pd
import pickle

alpha = np.load('models/alpha.npy')
print('Spectral Radius:', np.max(np.real(np.linalg.eigvals(alpha))))

df = pd.read_csv('processed_data/events.csv')
with open('processed_data/airports.pkl', 'rb') as f:
    ports = pickle.load(f)

for port in ['ATL', 'ORD', 'SEA', 'LAX']:
    idx = ports.index(port)
    count = len(df[df['NODE'] == idx])
    print(f'{port} count:', count)

mask = np.load('processed_data/adj.npy')
print('First 6 columns density:', np.mean(mask[:, :6]))
