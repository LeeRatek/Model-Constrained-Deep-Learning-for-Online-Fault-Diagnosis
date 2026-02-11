import pickle
import numpy as np
import torch
import os
import io


# Helper to load pickled torch tensors on CPU
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False
            )
        return super().find_class(module, name)


def load_pkl(path):
    print(f"Loading {path}...")
    try:
        with open(path, "rb") as f:
            data = CPUUnpickler(f).load()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


base_dir = r"c:\Users\이승민\Desktop\Offices\개발\BMS\Model-Constrained-Deep-Learning-for-Online-Fault-Diagnosis\data"
dti_path_vin2 = os.path.join(base_dir, "DTI", "0", "vin_2.pkl")

print("\n=== Finding Convergence Point in DTI vin_2 Column 0 ===")
dti_vin2 = load_pkl(dti_path_vin2)

if dti_vin2 is not None:
    if isinstance(dti_vin2, torch.Tensor):
        data = dti_vin2.detach().cpu().numpy()
    else:
        data = np.array(dti_vin2)

    col0 = data[:, 0]

    # Calculate absolute differences
    diffs = np.abs(np.diff(col0))

    # Thresholds to test
    thresholds = [1.0, 0.1, 0.05]

    suggested_idx = 0

    for thr in thresholds:
        valid_indices = []
        # Find first index i where ALL diffs[i:] < thr
        for i in range(len(diffs)):
            if np.all(diffs[i:] < thr):
                valid_indices.append(i + 1)  # +1 because diff is between i and i+1
                break

        if valid_indices:
            idx = valid_indices[0]
            val_at_idx = col0[idx]
            print(
                f"Threshold < {thr}: Statically Stable from Index {idx}. Value: {val_at_idx:.4f}"
            )
            if thr == 0.1:  # Heuristic choice
                suggested_idx = idx

    print(f"\nValues around suggested index {suggested_idx}:")
    start = max(0, suggested_idx - 5)
    end = min(len(col0), suggested_idx + 10)
    for i in range(start, end):
        d_val = diffs[i - 1] if i > 0 else 0
        mark = "<-- Suggested Cut" if i == suggested_idx else ""
        print(f"Index {i}: {col0[i]:.4f} (Prev Diff: {d_val:.4f}) {mark}")

    # Check validity for other columns
    print("\nChecking stability of Column 1 at this index...")
    col1 = data[:, 1]
    col1_diffs = np.abs(np.diff(col1))

    # Check if col1 is also relatively stable or at least not jumping wildly
    print(f"Col 1 Value at {suggested_idx}: {col1[suggested_idx]:.4f}")
    if suggested_idx > 0:
        print(f"Col 1 Diff at {suggested_idx}: {col1_diffs[suggested_idx-1]:.4f}")
