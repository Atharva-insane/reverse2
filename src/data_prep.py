import pandas as pd
import numpy as np
import os
import sys
import pickle
from pathlib import Path

def prepare_continuous_data(package_path=r"E:\Downloads\hawkes_event_log_package", data_dir='../processed_data'):
    print(f"Loading data from advanced external package: {package_path}")
    
    if package_path not in sys.path:
        sys.path.append(package_path)
        
    try:
        from load_hawkes_event_log import load_event_log, load_network_mask_top50, load_airport_index
    except ImportError:
        print(f"Error: Could not import load_hawkes_event_log from {package_path}.")
        return

    print("Loading continuous-time event log chunks (Top 50)...")
    df_raw = load_event_log(Path(package_path), top50=True)
    
    print("Loading proper airport index mapping...")
    airport_df = load_airport_index(Path(package_path), top50=True)
    # The true ground-truth list of airports sorted by `top_node` ID (0 to 49)
    airports = airport_df['airport'].tolist()
    
    print("Loading strict route-based Adjacency Mask...")
    mask_df = load_network_mask_top50(Path(package_path))
    
    print("Realigning Mask to match ground-truth node IDs...")
    # This is the critical fix. We force the rows and columns of the mask to 
    # strictly follow the 0-49 order defined by the airport index.
    mask_df = mask_df.reindex(index=airports, columns=airports, fill_value=0)
    
    package_mask = mask_df.to_numpy() # shape: (target, source)
    adj_matrix = package_mask.T       # shape: (source, target)
    
    print("Adapting data for the EM algorithm...")
    df = pd.DataFrame()
    df['TIME_HOURS'] = df_raw['t_hours']
    df['NODE'] = df_raw['top_node'].astype(int)
    
    if 'departure_delay' in df_raw.columns:
        df['DEPARTURE_DELAY'] = df_raw['departure_delay']
    elif 'DEPARTURE_DELAY' in df_raw.columns:
        df['DEPARTURE_DELAY'] = df_raw['DEPARTURE_DELAY']
    else:
        df['DEPARTURE_DELAY'] = 20.0 
        
    df = df.sort_values(by='TIME_HOURS')
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Saving strictly aligned processed data artifacts...")
    with open(os.path.join(data_dir, 'airports.pkl'), 'wb') as f:
        pickle.dump(airports, f)
        
    np.save(os.path.join(data_dir, 'adj.npy'), adj_matrix)
    
    out_path = os.path.join(data_dir, 'events.csv')
    df.to_csv(out_path, index=False)
    
    print(f"Success! Integrated external package data into {data_dir}.")
    print("Alignment verified. You can now safely run 'python train.py'!")

if __name__ == "__main__":
    prepare_continuous_data()
