import pandas as pd
import numpy as np
import os
import sys
import pickle
from pathlib import Path

def prepare_continuous_data(package_path=r"E:\Downloads\hawkes_event_log_package", data_dir='../processed_data'):
    print(f"Loading data from advanced external package: {package_path}")
    
    # Dynamically inject the package into the Python path
    if package_path not in sys.path:
        sys.path.append(package_path)
        
    try:
        from load_hawkes_event_log import load_event_log, load_network_mask_top50
    except ImportError:
        print(f"Error: Could not import load_hawkes_event_log from {package_path}.")
        print("Please ensure the path is correct and the package exists.")
        return

    # 1. Load the perfectly chronological UTC event log
    print("Loading continuous-time event log chunks (Top 50)...")
    df_raw = load_event_log(Path(package_path), top50=True)
    
    # 2. Load the pre-computed rigorous Adjacency mask
    print("Loading strict route-based Adjacency Mask...")
    mask_df = load_network_mask_top50(Path(package_path))
    
    # Extract airports from the mask columns
    airports = list(mask_df.columns)
    num_nodes = len(airports)
    
    # The mask defines allowed Hawkes edges: mask[target, source] = 1
    # Our EM expects alpha[source, target], so we transpose the mask if necessary.
    # Wait, the package says: "mask[target, source] = 1 means delay at source excites target".
    # This means the DataFrame rows are targets, columns are sources.
    # In our train.py, alpha[u, v] is influence of u (source) on v (target).
    # So we need adj_matrix[u, v] = 1 if source u infects target v.
    # This means adj_matrix should be the transpose of the package's mask matrix!
    package_mask = mask_df.to_numpy() # shape: (target, source)
    adj_matrix = package_mask.T       # shape: (source, target)
    
    # 3. Format the Event Log for our train.py
    print("Adapting data for the EM algorithm...")
    df = pd.DataFrame()
    
    # Convert t_hours into TIME_MINUTES (what our EM expects)
    df['TIME_MINUTES'] = df_raw['t_hours'] * 60.0
    
    # Map the node index
    df['NODE'] = df_raw['top_node'].astype(int)
    
    # We can keep the delay magnitude just in case, but EM only strictly needs TIME_MINUTES and NODE
    if 'departure_delay' in df_raw.columns:
        df['DEPARTURE_DELAY'] = df_raw['departure_delay']
    elif 'DEPARTURE_DELAY' in df_raw.columns:
        df['DEPARTURE_DELAY'] = df_raw['DEPARTURE_DELAY']
    else:
        df['DEPARTURE_DELAY'] = 20.0 # dummy value if missing
        
    # Sort just to be absolutely certain (though load_event_log already sorts)
    df = df.sort_values(by='TIME_MINUTES')
    
    # 4. Save to processed_data/ so the rest of our pipeline works seamlessly
    os.makedirs(data_dir, exist_ok=True)
    
    print("Saving processed data artifacts...")
    with open(os.path.join(data_dir, 'airports.pkl'), 'wb') as f:
        pickle.dump(airports, f)
        
    np.save(os.path.join(data_dir, 'adj.npy'), adj_matrix)
    
    out_path = os.path.join(data_dir, 'events.csv')
    df.to_csv(out_path, index=False)
    
    print(f"Success! Integrated external package data into {data_dir}.")
    print(f"Total UTC Events Loaded: {len(df)}")
    print(f"Adjacency Sparsity: {adj_matrix.mean() * 100:.2f}% connections exist.")
    print("You can now safely run 'python train.py'!")

if __name__ == "__main__":
    prepare_continuous_data()
