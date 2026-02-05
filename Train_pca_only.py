import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
from Function_ import *
from Class_ import *
import torch

# from sklearn.datasets import load_boston
import argparse
from monitering import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')
# VIN_data = pd.read_excel(r'{args.source_data_dir}/Name_list.xls')

parser = argparse.ArgumentParser(description="Run diagnostics plotting with CLI options")
parser.add_argument("--battery-type", choices=["QAS","DTI"], default="QAS", help="배터리 유형 선택 (DTI fault index range = [77~79], QAS fault index range = [335~392])")
parser.add_argument("--models-folder", type=str, default="260203_163103") # 260123_082222 & 260126_100030
parser.add_argument("--models-dir", type=str, default="./models" if os.environ.get("MODEL_DIR") is None else os.environ.get("MODEL_DIR"))
parser.add_argument("--source-data-dir", type=str, default="./data" if os.environ.get("SOURCE_DIR") is None else os.environ.get("SOURCE_DIR"))
args = parser.parse_args()

vals = read_values_from_sim_config(
    f"{args.models_dir}/{args.models_folder}",
    names_to_find=["learning_case","vehicle_start","vehicle_end","normalize_dx","normalize_val",
                   "ae_u_scale","ae_u_shift","ae_x_scale","ae_x_shift"],
    strict=False, # True: 키가 하나라도 없을 시 예외, False: 없는 키는 None
)
learning_case = vals["learning_case"]
vehicle_start = vals["vehicle_start"] if vals["vehicle_start"] is not None else 0
vehicle_end = vals["vehicle_end"] if vals["vehicle_end"] is not None else -1
normalize_dx = vals["normalize_dx"] if vals["normalize_dx"] is not None else False
normalize_val = vals["normalize_val"] if vals["normalize_val"] is not None else 4

## ================== For backward compatibility =================== ##
ae_u_scale = vals["ae_u_scale"] if vals["ae_u_scale"] is not None else 1.8
ae_u_shift = vals["ae_u_shift"] if vals["ae_u_shift"] is not None else 2.5
ae_x_scale = vals["ae_x_scale"] if vals["ae_x_scale"] is not None else 1.0
ae_x_shift = vals["ae_x_shift"] if vals["ae_x_shift"] is not None else 0


BATTERY_TYPE = args.battery_type # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(learning_case)
# SKIP_CHARGE_READY = False
dim_dict = get_input_dimensions(BATTERY_TYPE)

## =================== Load training data  =================== ##
if PREPROCESSING:
    net = CombinedAE(input_size=dim_dict["x"], encode2_input_size=dim_dict["q"], output_size=dim_dict["y"],
                        activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
else:
    net = CombinedAE(input_size=dim_dict["x"], encode2_input_size=dim_dict["q"], output_size=dim_dict["y"],
                        activation_fn=CustomSigmoidFunc(scale=ae_u_scale, shift=ae_u_shift), use_dx_in_forward=True).to(device)
netx = CombinedAE(input_size=dim_dict["x2"], encode2_input_size=dim_dict["q2"], output_size=dim_dict["y2"], activation_fn=CustomSigmoidFunc(scale=ae_x_scale, shift=ae_x_shift),
                        use_dx_in_forward=True).to(device)
net_state_dict = torch.load(os.path.join(f"{args.models_dir}/{args.models_folder}/artifact/", 'net.pth'))
net.load_state_dict(net_state_dict)
netx_state_dict = torch.load(os.path.join(f"{args.models_dir}/{args.models_folder}/artifact/", 'netx.pth'))
netx.load_state_dict(netx_state_dict)


data_path = make_model_path_based_timestamp(base=args.models_dir)
train_list = np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_train.npy").astype(np.int64).tolist()
FIRST_LOAD = True

#----------------------------------------Hyper Parameter Setting---------------------

train_list = train_list[vehicle_start:vehicle_end+1] if vehicle_end != -1 else train_list[vehicle_start:]
count = 0
for i in train_list:
    count += 1
    if count % 10 == 0:
        print(f"Processing vehicle {count}/{len(train_list)}")
    VEHICLE_ID = f'{i}'

    #----------------------------------------Data loading for MC-AE (customized) ------------------------------
    tensor = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl', verbose=False)
    tensorx = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl', verbose=False)
    
    #----------------------------------------Preprocessing ------------------------------#
    tensor, tensorx = preprocess_combined_tensor(tensor, tensorx, dim_dict, 
                                                BATTERY_TYPE, PREPROCESSING, SKIP_CHARGE_READY, 
                                                normalize_dx=normalize_dx, normalize_val=normalize_val)
    
    if FIRST_LOAD:
        FIRST_LOAD = False
        combined_tensor = tensor
        combined_tensorx = tensorx
        break
    else:
        combined_tensor = torch.cat((combined_tensor, tensor), dim=0)
        combined_tensorx = torch.cat((combined_tensorx, tensorx), dim=0)
    


#----------------------------------------Training for MC-AE--------------------------
x_recovered = combined_tensor[:, :dim_dict["x"]] # 0~1
y_recovered = combined_tensor[:, dim_dict["x"]:dim_dict["x"] + dim_dict["y"]] # 2~111
z_recovered = combined_tensor[:, dim_dict["x"] + dim_dict["y"]: dim_dict["x"] + dim_dict["y"] + dim_dict["z"]] # 112~221
q_recovered = combined_tensor[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"]:] # 222~

x_recovered2 = combined_tensorx[:, :dim_dict["x2"]]
y_recovered2 = combined_tensorx[:, dim_dict["x2"]:dim_dict["x2"] + dim_dict["y2"]]
z_recovered2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"]: dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"]]
q_recovered2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"]:]


AE_U_BATCHSIZE = 100
class Dataset(Dataset):
    def __init__(self, x, y, z, q):
        self.x = x.to(torch.double)
        self.y = y.to(torch.double)
        self.z = z.to(torch.double)
        self.q = q.to(torch.double)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]

## ================= Training MC-AE for voltage reconstruction ================= ##
train_loader = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=len(x_recovered),
                        shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loader):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    net = net.double()
    recon_imtest, recon = net(x, z, q)
AA = recon_imtest.cpu().detach().numpy()
yTrainU = y_recovered.cpu().detach().numpy()
ERRORU = np.abs(AA - yTrainU)


## ================= Training MC-AE for SOC reconstruction ================= ##
train_loaderx = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=len(x_recovered2), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loaderx):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    netx = netx.double()
    recon_imtestx, z = netx(x, z, q)

BB = recon_imtestx.cpu().detach().numpy()
yTrainX = y_recovered2.cpu().detach().numpy()
ERRORX = np.abs(BB - yTrainX)

df_data, _ = DiagnosisFeature(ERRORU,ERRORX)
results = Custom_PCA(df_data,0.99,0.99)

save_pca_results(f"{data_path}/", results)
