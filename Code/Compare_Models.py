import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib


xmed = [-0.3765142150803461, -0.32533992583436344, -0.25339925834363414, -0.22224969097651423, -0.1473423980222497, -0.09320148331273177, -0.056860321384425205, 0.04029666254635353, 0.12484548825710756, 0.21384425216316444, 0.26872682323856617, 0.34956736711990116, 0.42076637824474666, 0.4912237330037083, 0.5305315203955501]
ymed = [0.9654545454545455, 0.9654545454545455, 0.9654545454545455, 0.9545454545454546, 0.9509090909090909, 0.9472727272727273, 0.9509090909090909, 0.9472727272727273, 0.9363636363636363, 0.9254545454545455, 0.9327272727272727, 0.94, 0.9472727272727273, 0.9436363636363636, 0.9363636363636363]

xup = [ -0.37725587144622996, -0.3542645241038319, -0.22373300370828184, -0.19035846724351052, -0.09245982694684796, -0.060568603213844247, 0.04029666254635353, 0.07070457354758963, 0.20494437577255875, 0.3681087762669963, 0.43337453646477136, 0.500123609394314, 0.5290482076637826]
yup = [1.5072727272727273, 1.4927272727272727, 1.3363636363636364, 1.3072727272727271, 1.2054545454545456, 1.2054545454545456, 1.1436363636363636, 1.1327272727272728, 1.1327272727272728, 1.1763636363636363, 1.2127272727272729, 1.230909090909091, 1.2236363636363636]

xdown = [-0.37873918417799757, -0.25636588380716935, -0.22447466007416564, -0.19110012360939432, -0.06205191594561185, 0.05364647713226206, 0.169344870210136, 0.2739184177997528, 0.39110012360939433, 0.4867737948084055, 0.5290482076637826]
ydown = [0.6272727272727272, 0.6890909090909091, 0.6854545454545454, 0.7036363636363636, 0.7545454545454545, 0.7872727272727273, 0.7909090909090909, 0.769090909090909, 0.74, 0.7109090909090909, 0.6927272727272727]


xmed = np.array(xmed)
xup = np.array(xup)
xdown = np.array(xdown)

sim = 'NIHAO'

if sim == 'NIHAO':
    model_folder = '/home/jsarrato/Physics/PhD/Paper3/Debut/NewModelsSept_matchingdataset/Models_NI_and_AU_new2024_05rxy_05lim_trainNH/PDF+hlr+std+v98_STDREV98'
elif sim == 'AURIGA':
    model_folder = '/home/jsarrato/Physics/PhD/Paper3/Debut/NewModelsSept_matchingdataset/Models_NI_and_AU_new2024_05rxy_05lim_trainAU/PDF+hlr+std+v98_STDREV98'

inputs = np.load(os.path.join(model_folder, 'Part_output_train.npy'))
#labels = np.load(os.path.join(model_folder,'Y_train_scaled.npy')
labels = np.load(os.path.join(model_folder,'Y_train.npy'))

inputs_val = np.load(os.path.join(model_folder,'Part_output_test.npy'))
labels_val = np.load(os.path.join(model_folder,'Y_test.npy'))
#labels_val = np.load(os.path.join(model_folder,'Y_test_scaled.npy')

Y_scaler = joblib.load(os.path.join(model_folder,'scaler1.gz'))

N_samples = 1000

x_small = np.arange(0.6,2.6,0.2)

samples = np.load(os.path.join(model_folder,'samples_train.npy'))

# Get medians over samples and deescale them
medians = Y_scaler.inverse_transform(np.median(samples,axis=0))
labels_descaled = Y_scaler.inverse_transform(labels)


ratio = (10**medians)/(10**labels_descaled)

ratio_medians = np.median(ratio, axis = 0)
ratio_perc_16 = np.percentile(ratio, 16,axis = 0)
ratio_perc_84 = np.percentile(ratio, 84, axis = 0)



# Define the x-axis values
x = np.arange(0.2, 4, 0.1)

Ns = [100, 1000]

# List of folders
folders = [f'Cheb_{sim}_poisson{N}_jgnntransform' for N in Ns]

# Define colors for each folder
colors = ['blue', 'green', 'red', 'purple']

# Create a plot
plt.figure(figsize=(10, 6))

plt.gca().axhline(y = 1, color = 'black', linestyle = '--')

plt.gca().fill_between(x_small, ratio_perc_16, ratio_perc_84, color='gray', alpha=0.5, label=f'CNN trained on {sim}')

plt.plot(10**xmed, ymed, ls = '-', color = 'dimgray', label = 'Genina+20')
plt.plot(10**xup, yup, ls = '-', color = 'dimgray')
plt.plot(10**xdown, ydown, ls = '-', color = 'dimgray')

# Loop through each folder
for i, folder in enumerate(folders):
    # Load the data from the pickle file
    with open(os.path.join(folder, 'plot_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    # Extract the relevant keys
    r_medians_train = data['r_medians_train']
    r_p16_train = data['r_p16_train']
    r_p84_train = data['r_p84_train']
    
    # Extract the number from the folder name (e.g., 1000, 100, 20, 5)
    folder_number = folder.split('poisson')[-1].split('_')[0]
    
    # Plot the median line with a label
    plt.plot(x, r_medians_train, color=colors[i], label=f'{folder_number} Stars')
    
    # Plot the p16 and p84 lines with the same linestyle (dashed) and no label
    plt.plot(x, r_p16_train, color=colors[i], linestyle='--')
    plt.plot(x, r_p84_train, color=colors[i], linestyle='--')

# Add labels, title, and legend
plt.xlabel(r'r/R$_{\rm h}$')
plt.ylabel(r'M(<r)$_{\rm pred}$/M(<r)$_{\rm true}$')
plt.legend()

plt.savefig(f'/home/jsarrato/Physics/PhD/Paper-GraphSimProfiles/Images/CompareModels_{sim}_100_and_1000.png', bbox_inches = 'tight')
# Show the plot
plt.show()
