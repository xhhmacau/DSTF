import numpy as np
import pandas as pd
from numpy.linalg import eigh
import os
import argparse

def generate_q_matrix(file_path, save_path='./dataset', train_ratio=0.7):
    """
    Generate Q-matrices (eigenvectors of covariance matrix) for OLinear model.
    """
    if not os.path.exists(save_path):
        print(f'{save_path} is created.')
        os.makedirs(save_path, exist_ok=True)
        
    # check
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename_with_ext = os.path.basename(file_path)
    filename, ext = os.path.splitext(filename_with_ext)

    print(f"Processing {filename_with_ext}...")

    if ext == '.csv':
        data = pd.read_csv(file_path, header=0)
        data = data.dropna(axis=1, how='all')  # in case there is a column with all nans
        data = data.values
    else:
        try:
            data = pd.read_excel(file_path, header=0).values
        except Exception as e:
            print(f"Error reading excel file: {e}")
            return

    train_length = int(data.shape[0]*train_ratio)

    # base_ratio logic from original script, seemingly always 1.0
    for base_ratio in [1.0]:  
        ratio = train_ratio * base_ratio
        
        # Data slicing logic from original script
        # Adjusted to handle both CSV and potential other formats consistently if needed,
        # but keeping original logic: CSV skips first column (date?), others don't?
        # Original script:
        # if ext == '.csv': A = data[... , 1:] (Assuming 1st col is date)
        # else: A = data[..., 0:]
        
        if ext == '.csv':
            # Assuming first column is date for CSVs
            A = data[train_length-int(data.shape[0]*ratio):train_length, 1:].astype(np.float32)
        else:
            A = data[train_length-int(data.shape[0]*ratio):train_length, 0:].astype(np.float32)

        print(f'Data Shape for Covariance: {A.shape}')

        # Standard time lags for TSF benchmarks
        time_lags = [24, 48, 96, 192, 336, 720]
        
        for time_lag in time_lags:
            # Initialize a list to store covariance matrices for all features
            Sigma_list = []

            # Loop through all features
            # Original script loop: for feature_idx in range(int(A.shape[1]/1)):
            for feature_idx in range(A.shape[1]):
                # Construct the lagged matrix for the current feature
                lagged_matrix = np.array([
                    A[i:A.shape[0]-time_lag+i+1, feature_idx]
                    for i in range(time_lag)
                ])
                
                if np.isnan(lagged_matrix).any():
                    lagged_matrix = np.nan_to_num(lagged_matrix)
                    print(f'Warning: NaN found in lagged matrix for feature {feature_idx}, time_lag {time_lag}')
                    
                # Compute the covariance matrix for the lagged matrix
                cov_matrix = np.cov(lagged_matrix)
                diag_vec = np.diag(cov_matrix)
        
                if (diag_vec < 1e-4).any():
                    continue
                
                cov_matrix = cov_matrix / diag_vec  # make sure the diagonal entries are 1

                Sigma_list.append(np.array(cov_matrix, dtype=np.float32))

            if not Sigma_list:
                print(f"Skipping time_lag {time_lag}: No valid features found.")
                continue

            # Average over all features to get the final Sigma
            Sigma = np.mean(Sigma_list, axis=0)

            # Compute eigenvalues and eigenvectors of Sigma
            eigenvalues, eigenvectors = eigh(Sigma)

            q_mat = np.flip(eigenvectors.T, axis=0)

            save_file = os.path.join(save_path, f'{filename}_{time_lag}_ratio{ratio:.1f}.npy')
            np.save(save_file, q_mat)

            # Display the result
            print(f'  [Saved] {save_file} (Shape: {q_mat.shape})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Q-matrices used by OLinear model')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='Root directory of datasets')
    parser.add_argument('--data_path', type=str, default=None, help='Filename of the dataset (e.g., ETTh1.csv)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (e.g. ETTh1, ECL) to lookup in settings.py')
    parser.add_argument('--file_path', type=str, default=None, help='Full path to dataset file. Overrides root_path/data_path if provided.')
    parser.add_argument('--save_path', type=str, default='./dataset/', help='Where to save the .npy files')
    parser.add_argument('--train_ratio', type=str, default='0.7', help='Training ratio (default: 0.7). can be "0.7" or "0.6,0.7"')
    
    args = parser.parse_args()

    full_path = None
    if args.file_path:
        full_path = args.file_path
    elif args.dataset:
        # Import settings from parent directory
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from settings import data_settings
        
        if args.dataset in data_settings:
            data_file = data_settings[args.dataset]['data']
            full_path = os.path.join(args.root_path, data_file)
        else:
             print(f"Error: Dataset '{args.dataset}' not found in settings.py")
             # Fallback: assume dataset name is filename sans extension? No, better safe.
             # Check if data_path was provided as fallback
             if args.data_path:
                 full_path = os.path.join(args.root_path, args.data_path)
             else:
                 sys.exit(1)
    elif args.data_path:
        full_path = os.path.join(args.root_path, args.data_path)
    else:
        print("Error: Please provide --dataset, --data_path or --file_path")
        sys.exit(1)
    
    ratios = [float(r) for r in args.train_ratio.split(',')]
    
    for r in ratios:
        print(f"\nGeneratring for train_ratio: {r}")
        generate_q_matrix(full_path, args.save_path, train_ratio=r)

    print("\nAll Done!")
