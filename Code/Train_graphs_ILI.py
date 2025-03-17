import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from accelerate import Accelerator

import torch

import ili
from ili.inference import InferenceRunner

import pickle

from ili.dataloaders import TorchLoader
from torch_geometric.loader.dataloader import Collater

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from tqdm import tqdm

import argparse

from utils import GraphNN, GraphCreator, sample_with_timeout


parser = argparse.ArgumentParser(description='Train and evaluate the model.')
parser.add_argument('mode', choices=['train', 'sample', 'sampletest'], help='Mode to run the script in: train or sample, or sample test set only')
parser.add_argument('sim', choices=['AURIGA', 'NIHAO'], help='Simulation type')
parser.add_argument('test_long', type=int, choices=[0, 1], help='Test long flag (0 or 1)')
parser.add_argument('hlr_std', type=int, choices=[0, 1], help='Use pre-calculated hlr and std flag (0 or 1)')
parser.add_argument('N_stars', type=int, help='Number of stars to be used for creating graphs')
parser.add_argument('GraphNN_type', choices=['Cheb', 'GCN', 'GAT'])

args = parser.parse_args()

sim = args.sim
test_long = bool(args.test_long)

capsize_num = 5
markersize_num = 3

#################################################################################################
# CUDA CONFIGURATION
#################################################################################################

# Accelerator takes care of CUDA configuration. Set CUDA_VISIBLE_DEVICES before running to restrict devices
accelerator = Accelerator()
device = accelerator.device

Nstars = args.N_stars

N_proj_per_gal = 64

stellar_features = {
    'positions': [],
    'velocities': [],
}

labels = []

estimator_masses = []

seed = 22133
np.random.seed(seed)


data_folder = '/net/debut/scratch/jsarrato/Paper-GraphSimProfiles/work/arrs_NIHAO_and_AURIGA_PCAfilt_1000/'

label_file = pd.read_csv('/net/debut/scratch/jsarrato/Paper-GraphSimProfiles/work/proj_data_NIHAO_and_AURIGA_PCAfilt_samplearr.csv')
label_file = label_file.drop_duplicates(subset=['name']).reset_index(drop = True)
label_file['sim'] = label_file['name'].apply(lambda x: 1 if x.startswith('halo') else 0) # 1 for AURIGA, 0 for NIHAO

AU_indices = np.array(label_file[label_file['sim'] == 1]['i_index'], dtype = int)
NH_indices = np.array(label_file[label_file['sim'] == 0]['i_index'], dtype = int)
print(len(AU_indices), len(NH_indices))

if sim == 'AURIGA':
	existing_files = [data_folder + f'masses_arr{idx}.npy' for idx in AU_indices]
	len_train_files = len(existing_files)
	if test_long:
		existing_files = existing_files	+ [data_folder + f'masses_arr{idx}.npy' for idx in NH_indices]
		len_test_files = len([data_folder + f'masses_arr{idx}.npy' for idx in NH_indices])
	else:
		existing_files = existing_files	+ [data_folder + f'masses_arr{idx}.npy' for idx in NH_indices[:100]]
		len_test_files = len([data_folder + f'masses_arr{idx}.npy' for idx in NH_indices[:100]])

if sim == 'NIHAO':
	existing_files = [data_folder + f'masses_arr{idx}.npy' for idx in NH_indices]
	len_train_files = len(existing_files)
	if test_long:
		existing_files = existing_files	+ [data_folder + f'masses_arr{idx}.npy' for idx in AU_indices]
		len_test_files = len([data_folder + f'masses_arr{idx}.npy' for idx in AU_indices])
	else:
		existing_files = existing_files	+ [data_folder + f'masses_arr{idx}.npy' for idx in AU_indices[:100]]
		len_test_files = len([data_folder + f'masses_arr{idx}.npy' for idx in AU_indices[:100]])

# Generate random number of stars for each projection
nstars_arr = np.random.poisson(Nstars, size=(len_train_files + len_test_files) * N_proj_per_gal)
#nstars_arr = np.array([Nstars]*((len_train_files + len_test_files) * N_proj_per_gal))

print('Reading Data')
file_indices = []
for i, file_name in enumerate(tqdm(existing_files)):
    # Extract the index from the file name
    idx = int(file_name.split('_arr')[-1].split('.npy')[0])

    masses_name = data_folder + f'masses_arr{idx}.npy'
    hlrstd_name = data_folder + f'hlrstd_ar{idx}.npy'
    posvel_name = data_folder + f'posvel_{idx}.pkl'

    masses = np.load(masses_name)
    posvel = torch.load(posvel_name, weights_only=False)

    random_projs = np.random.choice(len(posvel), N_proj_per_gal)

    for j, proj_idx in enumerate(random_projs):
        nstars = nstars_arr[i * N_proj_per_gal + j]  # Use a different random number of stars for each projection

        posveldata = posvel[proj_idx]
        masses_idx = np.log10(masses[:-4, proj_idx])
            
        estim_masses = np.array([masses[9, proj_idx], masses[-2, proj_idx], masses[16, proj_idx], masses[17, proj_idx], masses[17, proj_idx]])
          
        if np.any(np.isnan(posveldata)) or np.any(np.isinf(posveldata)):
            continue
          
        if np.any(np.isnan(masses_idx)) or np.any(np.isinf(masses_idx)):
            continue

        estimator_masses.append(estim_masses)
        stellar_features['positions'].append(posveldata[:nstars, :2])
        stellar_features['velocities'].append(posveldata[:nstars, -1])
        labels.append(masses_idx)
        file_indices.append(i)  # Keep track of the file index
            
    if i == len_train_files:
        train_and_val_size = len(labels)
		

labels2 = np.array(labels)

labels2_train_val = labels2[:train_and_val_size]
labels2_test = labels2[train_and_val_size:]

mask_test = np.ones_like(labels2_test[:,0], dtype=bool)

print('Original number of test points:', np.sum(mask_test))

for i in range(labels2_test.shape[1]):
    mask_test = mask_test & (labels2_test[:,i] > np.min(labels2_train_val[:,i])) & (labels2_test[:,i] < np.max(labels2_train_val[:,i]))

print('Number of test points:', np.sum(mask_test))

labels2_test_filtered = labels2_test[mask_test]

highs = np.max(labels2, axis=0)
lows = np.min(labels2, axis=0)

k = np.min([Nstars, 20])

transform = GraphCreator(
    graph_type="KNNGraph",
    graph_config={
        "k": k, 
        "force_undirected": True, 
        "loop": True
    },
    use_log_radius=True,
)

stds = []
hlrs = []

print('Transforming Data')
data_list = []
for i in tqdm(range(len(labels))):
	data = transform(
		positions=stellar_features['positions'][i],
		velocities=stellar_features['velocities'][i],
		labels=labels[i],
	)
	data.hlr = torch.tensor(torch.quantile(data.x[:,0], 0.5), dtype = torch.float32).reshape((1,1))
	data.std = torch.tensor(torch.std(data.x[:,1]), dtype = torch.float32).reshape((1,1))
	stds.append(float(data.std.numpy()))
	hlrs.append(float(data.hlr.numpy()))
	#data.hlr = torch.tensor([0], dtype = torch.float32).reshape((1,1))
	#data.std = torch.tensor([0], dtype = torch.float32).reshape((1,1))
	data_list.append(data)

stds = np.array(stds)
hlrs = np.array(hlrs)
estim_masses = np.array(estimator_masses)

kmtom = 10**3

kpctom = 3.086*10**19

kgtomsun = (2*10**30)**-1

G = 6.6743*10**-11	
    
C_Amorisco = 5.8
C_Errani = 3.5*1.8
C_Wolf = 4
C_Campbell = 6
C_Walker = 5/2

M_Wolf = C_Wolf * (G**-1) * (stds**2) * (kmtom**2) * hlrs * kpctom * kgtomsun #Wolf
M_Errani= C_Errani * (G**-1) * (stds**2) * (kmtom**2) * hlrs * kpctom * kgtomsun #Errani
M_Campbell= C_Campbell * (G**-1) * (stds**2) * (kmtom**2) * hlrs * kpctom * kgtomsun #Campbell
M_Amorisco= C_Amorisco * (G**-1) * (stds**2) * (kmtom**2) * hlrs * kpctom * kgtomsun #Amorisco
M_Walker = C_Walker * (G**-1) * (stds**2) * (kmtom**2) * hlrs * kpctom * kgtomsun #Walker

Walker_rel = M_Walker/estim_masses[:,0]
Wolf_rel = M_Wolf/estim_masses[:,1]
Amorisco_rel = M_Amorisco/estim_masses[:,2]
Errani_rel = M_Errani/estim_masses[:,3]
Campbell_rel = M_Campbell/estim_masses[:,4]
    
data_list_test = [data_list[train_and_val_size:][i] for i in range(len(mask_test)) if mask_test[i]]


data_list = data_list[:train_and_val_size]
file_indices = file_indices[:train_and_val_size]
    
collater = Collater(data_list)
    
def collate_fn(batch):
    batch = collater(batch)
    return batch, batch.y
    
main_file_folder = '/scratch/jsarrato/Wolf_for_FIRE/work/'
Model_str = f'{args.GraphNN_type}_{sim}_poisson{Nstars}_Nfiles{len_train_files}_hlrstd{args.hlr_std}'
    
model_folder = main_file_folder+'Graph+Flow_Mocks_NH/'+Model_str
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
    
    
# Get unique file indices
unique_file_indices = np.unique(file_indices)

# Randomly select 20% of the files for validation
num_val_files = int(0.2 * len(unique_file_indices))
val_file_indices = np.random.choice(unique_file_indices, num_val_files, replace=False)

# Create masks for training and validation sets
val_mask = np.isin(file_indices, val_file_indices)
train_mask = ~val_mask

# Split data into training and validation sets based on the masks
train_data_list = [data_list[i] for i in range(len(data_list)) if train_mask[i]]
val_data_list = [data_list[i] for i in range(len(data_list)) if val_mask[i]]

train_size = len(train_data_list)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_data_list, batch_size=64, shuffle=True, num_workers=1, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(
    val_data_list, batch_size=64, shuffle=False, num_workers=1, collate_fn=collate_fn)


loader = TorchLoader(train_loader, val_loader)


if args.GraphNN_type == 'Cheb':
    embedding = GraphNN(in_channels= 2,
    out_channels= 128,
    hidden_graph_channels=128,
    num_graph_layers= 3,
    hidden_fc_channels= 128,
    num_fc_layers= 2,
    hlr_std = bool(args.hlr_std),
    activation= "relu",
    graph_layer_name= "ChebConv",
    graph_layer_params={"K": 4,"normalization": "sym","bias": True}
    )
elif args.GraphNN_type == 'GCN':
    embedding = GraphNN(in_channels= 2,
    out_channels= 128,
    hidden_graph_channels=128,
    num_graph_layers= 3,
    hidden_fc_channels= 128,
    num_fc_layers= 2,
    hlr_std = bool(args.hlr_std),
    activation= "relu",
    graph_layer_name= "GCNConv",
    graph_layer_params={"normalize":True,"bias": True}
    )
elif args.GraphNN_type == 'GAT':
    embedding = GraphNN(in_channels= 2,
    out_channels= 128,
    hidden_graph_channels=128,
    num_graph_layers= 3,
    hidden_fc_channels= 128,
    num_fc_layers= 2,
    hlr_std = bool(args.hlr_std),
    activation= "relu",
    graph_layer_name= "GATConv",
    graph_layer_params={"heads": 1,"bias": True}
    )

# define training arguments
train_args = {
    'training_batch_size': 64,
    'learning_rate': 5e-4,
    'stop_after_epochs': 20,
    'max_epochs': 500
}



nets = [ili.utils.load_nde_lampe(model='maf', hidden_features=128, num_transforms=4, embedding_net=embedding, x_normalize=False, device=device)]

prior = ili.utils.Uniform(low=lows, high=highs, device=device)

# initialize the trainer
runner = InferenceRunner.load(
    backend='lampe',
    engine='NPE',
    prior=prior,
    nets=nets,
    device=device,
    train_args=train_args,
    proposal=None,
    out_dir=model_folder
)


if args.mode == 'train':
    posterior_ensemble, summaries = runner(loader=loader)
    
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = list(mcolors.TABLEAU_COLORS)
    for i, m in enumerate(summaries):
        ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
        ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
    ax.set_xlim(0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log probability')
    ax.legend()
    f.savefig(model_folder + '/loss.png', bbox_inches = 'tight')
else:
    posterior_ensemble = pickle.load(open(model_folder+'/posterior.pkl', 'rb'))
    
N_samples = 1000

n_min = 5  # Set the timeout duration in minutes
n_sec = n_min * 60  # Convert minutes to seconds

if args.mode in ['train', 'sample']:
    print('Sampling Training Set')
    samples_train = sample_with_timeout(posterior_ensemble, train_data_list, N_samples, device, n_sec)
    np.save(model_folder + '/samples_train', samples_train)

    print('Sampling Validation Set')
    samples_val = sample_with_timeout(posterior_ensemble, val_data_list, N_samples, device, n_sec)
    np.save(model_folder + '/samples_val', samples_val)
else:
    samples_train = np.load(model_folder + '/samples_train.npy')
    samples_val = np.load(model_folder + '/samples_val.npy')

print('Sampling Test Set')
samples_test = sample_with_timeout(posterior_ensemble, data_list_test, N_samples, device, n_sec)
np.save(model_folder + '/samples_test.npy', samples_test)

xmed = [-0.3765142150803461, -0.32533992583436344, -0.25339925834363414, -0.22224969097651423, -0.1473423980222497, -0.09320148331273177, -0.056860321384425205, 0.04029666254635353, 0.12484548825710756, 0.21384425216316444, 0.26872682323856617, 0.34956736711990116, 0.42076637824474666, 0.4912237330037083, 0.5305315203955501]
ymed = [0.9654545454545455, 0.9654545454545455, 0.9654545454545455, 0.9545454545454546, 0.9509090909090909, 0.9472727272727273, 0.9509090909090909, 0.9472727272727273, 0.9363636363636363, 0.9254545454545455, 0.9327272727272727, 0.94, 0.9472727272727273, 0.9436363636363636, 0.9363636363636363]

xup = [ -0.37725587144622996, -0.3542645241038319, -0.22373300370828184, -0.19035846724351052, -0.09245982694684796, -0.060568603213844247, 0.04029666254635353, 0.07070457354758963, 0.20494437577255875, 0.3681087762669963, 0.43337453646477136, 0.500123609394314, 0.5290482076637826]
yup = [1.5072727272727273, 1.4927272727272727, 1.3363636363636364, 1.3072727272727271, 1.2054545454545456, 1.2054545454545456, 1.1436363636363636, 1.1327272727272728, 1.1327272727272728, 1.1763636363636363, 1.2127272727272729, 1.230909090909091, 1.2236363636363636]

xdown = [-0.37873918417799757, -0.25636588380716935, -0.22447466007416564, -0.19110012360939432, -0.06205191594561185, 0.05364647713226206, 0.169344870210136, 0.2739184177997528, 0.39110012360939433, 0.4867737948084055, 0.5290482076637826]
ydown = [0.6272727272727272, 0.6890909090909091, 0.6854545454545454, 0.7036363636363636, 0.7545454545454545, 0.7872727272727273, 0.7909090909090909, 0.769090909090909, 0.74, 0.7109090909090909, 0.6927272727272727]

xmed = np.array(xmed)
xup = np.array(xup)
xdown = np.array(xdown)



truths_train = labels2[:train_size]
medians_train = 10**np.nanmedian(samples_train, axis = 0)/(10**truths_train)
r_medians_train = np.nanmedian(medians_train, axis = 0)
r_p16_train = np.nanpercentile(medians_train, 16, axis = 0)
r_p84_train = np.nanpercentile(medians_train, 84, axis = 0)


truths_val = labels2[train_size:train_and_val_size]
medians_val = 10**np.nanmedian(samples_val, axis = 0)/(10**truths_val)
r_medians_val = np.nanmedian(medians_val, axis = 0)
r_p16_val = np.nanpercentile(medians_val, 16, axis = 0)
r_p84_val = np.nanpercentile(medians_val, 84, axis = 0)

truths_test = labels2[train_and_val_size:train_and_val_size+len(data_list_test)]
medians_test = 10**np.nanmedian(samples_test, axis = 0)/(10**truths_test)
r_medians_test = np.nanmedian(medians_test, axis = 0)
r_p16_test = np.nanpercentile(medians_test, 16, axis = 0)
r_p84_test = np.nanpercentile(medians_test, 84, axis = 0)

print('VALIDATION')
print(r_medians_val)
print(r_p84_val - r_p16_val)
print('')

print('TRAINING')
print(r_medians_train)
print(r_p84_train - r_p16_train)
print('')

print('TESTING')
print(r_medians_test)
print(r_p84_test - r_p16_test)
print('')


x = np.arange(0.2,4,0.1) # Positions: 1.0 - 9, 1.8 - 17, 1.7 - 16
plt.figure()
plt.gca().axhline(y = 1, ls = '--', color = 'k')


plt.plot(x, r_medians_val, label = 'Validation')
plt.fill_between(x, r_p16_val, r_p84_val, alpha = 0.5)
plt.plot(x, r_medians_train, label = 'Training')
plt.fill_between(x, r_p16_train, r_p84_train, alpha = 0.5)
plt.plot(x, r_medians_test, label = 'Testing')
plt.fill_between(x, r_p16_test, r_p84_test, alpha = 0.5)
plt.plot(10**xmed, ymed, ls = ':', color = 'darkgray', label = 'Genina+20')
plt.plot(10**xup, yup, ls = ':', color = 'darkgray')
plt.plot(10**xdown, ydown, ls = ':', color = 'darkgray')
plt.errorbar(1,np.nanmedian(Walker_rel),np.array([np.nanmedian(Walker_rel)-np.nanpercentile(Walker_rel,16), np.nanpercentile(Walker_rel,84)-np.nanmedian(Walker_rel)]).reshape((2,1)),capsize = capsize_num,fmt='o', markersize = markersize_num, label = 'Walker')
plt.errorbar(4/3,np.nanmedian(Wolf_rel),np.array([np.nanmedian(Wolf_rel)-np.nanpercentile(Wolf_rel,16), np.nanpercentile(Wolf_rel,84)-np.nanmedian(Wolf_rel)]).reshape((2,1)),capsize = capsize_num,fmt='d', markersize = markersize_num, label = 'Wolf')
plt.errorbar(1.7,np.nanmedian(Amorisco_rel),np.array([np.nanmedian(Amorisco_rel)-np.nanpercentile(Amorisco_rel,16), np.nanpercentile(Amorisco_rel,84)-np.nanmedian(Amorisco_rel)]).reshape((2,1)),capsize = capsize_num,fmt='*', markersize = markersize_num, label = 'Amorisco')
plt.errorbar(1.8,np.nanmedian(Campbell_rel),np.array([np.nanmedian(Campbell_rel)-np.nanpercentile(Campbell_rel,16), np.nanpercentile(Campbell_rel,84)-np.nanmedian(Campbell_rel)]).reshape((2,1)),capsize = capsize_num,fmt='<', markersize = markersize_num, label = 'Campbell')
plt.errorbar(1.8,np.nanmedian(Errani_rel),np.array([np.nanmedian(Errani_rel)-np.nanpercentile(Errani_rel,16), np.nanpercentile(Errani_rel,84)-np.nanmedian(Errani_rel)]).reshape((2,1)),capsize = capsize_num,fmt='^', markersize = markersize_num, lw = 2, label = 'Errani')
plt.xlabel(r'r/R$_{\rm h}$')
plt.ylabel(r'M(<r)$_{\rm pred}$/M(<r)$_{\rm true}$')
plt.title(sim + ' ' + str(Nstars) +' stars')
plt.legend()
plt.savefig(model_folder + '/TrainingVsValidationVsTest.png', bbox_inches = 'tight')

# Data to be saved
plot_data = {
    'xmed': xmed,
    'ymed': ymed,
    'xup': xup,
    'yup': yup,
    'xdown': xdown,
    'ydown': ydown,
    'r_medians_train': r_medians_train,
    'r_p16_train': r_p16_train,
    'r_p84_train': r_p84_train,
    'r_medians_val': r_medians_val,
    'r_p16_val': r_p16_val,
    'r_p84_val': r_p84_val,
    'r_medians_test': r_medians_test,
    'r_p16_test': r_p16_test,
    'r_p84_test': r_p84_test,
    'sim': sim,
    'Nstars': Nstars,
    'actual_n': nstars_arr,
    'seed': seed,
    'N_files': len_train_files,
    'N_testing_files': len_test_files,
    'N_proj_per_gal': N_proj_per_gal,
    'model_folder': model_folder,
    'hlrs': hlrs,
    'stds': stds,
    'estim_masses': estim_masses,
    'M_Wolf': M_Wolf,
    'M_Errani': M_Errani,
	'M_Campbell': M_Campbell,
	'M_Amorisco': M_Amorisco,
	'M_Walker': M_Walker,
    'test_long': test_long
}

# Save data to a pkl file
with open(model_folder + '/plot_data.pkl', 'wb') as f:
    pickle.dump(plot_data, f)

print("Plot data saved to " + model_folder + '/plot_data.pkl')

