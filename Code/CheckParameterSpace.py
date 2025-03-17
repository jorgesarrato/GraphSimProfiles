import pickle
import glob
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


pkl_files = glob.glob('gal*.pkl')

scalar_data = []

class galaxy(object):
    pass

for pkl_file in pkl_files:
    galdata = []
    with open(pkl_file, 'rb') as inp:
        while True:
            try:
                galdata.append(pickle.load(inp))
            except:
                break


    # Loop through each object in the list
    for obj in galdata:
        # Extract scalar values using attribute access
        scalar_dict = {
            'path': obj.path,
            'name': obj.name,
            'Vmax_g': float(obj.Vmax_g),
            'Rmax_g': float(obj.Rmax_g),  # Assuming Rmax_g is a single-element array
            'K_rot': float(obj.K_rot),
            'K_rot_salva': float(obj.K_rot_salva),
            'Rvir': float(obj.Rvir),
            #'res_s': float(obj.Ms)/float(obj.Ns),
            #'res_dm': float(obj.Mvir)/float(obj.Ndm),
            #'res_g': float(obj.Mg)/float(obj.Ng),
            'Mvir': float(obj.Mvir),
            'Ms': float(obj.Ms),
            'Mh': float(obj.Mh),
            #'Mg': float(obj.Mg),
            'Rhl': float(obj.Rhl),
            'eps_s': float(obj.eps_s),
            'eps_dm': float(obj.eps_dm),
            #'angle': float(obj.angle)
        }
        # Append the dictionary to the list
        scalar_data.append(scalar_dict)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(scalar_data)
df = df.dropna().reset_index(drop = True)


df['simflag'] = df['name'].apply(lambda x: 0 if x.startswith('g') else 1)

# Drop the 'name' and 'simflag' columns for PCA
features = df.drop(columns=['path','name', 'simflag'])

#features = features.dropna().reset_index(drop = True)
# Standardize the features
features_standardized = (features - features.mean()) / features.std()

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_standardized)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['simflag'] = df['simflag']
pca_df['eps_s'] = features_standardized['eps_s']
pca_df['eps_dm'] = features_standardized['eps_dm']
#pca_df['res_s'] = features_standardized['res_s']
#pca_df['res_dm'] = features_standardized['res_dm']

# Plot the PCA results
plt.figure(figsize=(8, 6))
colors = {0: 'blue', 1: 'red'}
labels = {0: 'NIHAO', 1: 'AURIGA'}
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['simflag'].apply(lambda x: colors[x]), alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Galaxy Data')

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in colors]
plt.legend(handles, [labels[i] for i in colors], title="Simulation")

plt.show()

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['eps_s'], alpha=0.5)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Galaxy Data')
plt.show()

# Plot the PCA results
plt.figure(figsize=(8, 6))
colors = {0: 'blue', 1: 'red'}
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['eps_dm'], alpha=0.5)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Galaxy Data')
plt.show()


# Analyze the contribution of each feature to the PCA components
pca_components = pd.DataFrame(pca.components_, columns=features.columns, index=['PC1', 'PC2'])

# Print the contribution of each feature to the PCA components
print("Contribution of each feature to the PCA components:")
print(pca_components)

# Plot the contribution of each feature to the PCA components
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(pca_components.columns, pca_components.loc['PC1'])
plt.title('Contribution to PC1')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
plt.bar(pca_components.columns, pca_components.loc['PC2'])
plt.title('Contribution to PC2')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

mask = (pca_df['PC1'] < 1.5) & (pca_df['PC2'] < -0.85)

df_filtered = df[mask].reset_index(drop = True)

import numpy as np

df_filtered['Rvir'] = np.log10(df_filtered['Rvir'])
df_filtered['Mvir'] = np.log10(df_filtered['Mvir'])
df_filtered['Ms'] = np.log10(df_filtered['Ms'])
df_filtered['Mh'] = np.log10(df_filtered['Mh'])
df_filtered['Rhl'] = np.log10(df_filtered['Rhl'])
df_filtered['Rmax_g'] = np.log10(df_filtered['Rmax_g'])
df_filtered['Vmax_g'] = np.log10(df_filtered['Vmax_g'])

# Create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(df_filtered.drop(columns = 'simflag'), c=df_filtered['simflag'], figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

mask2 = (df_filtered['eps_dm'] < 0.7) & (df_filtered['eps_s'] > 0.1)

df_filtered2 = df_filtered[mask2].reset_index(drop = True)

# Create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(df_filtered2.drop(columns = 'simflag'), c=df_filtered2['simflag'].apply(lambda x: colors[x]), figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=10, alpha=.8)

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in colors]
plt.legend(handles, [labels[i] for i in colors], title="Simulation", loc = [1.05, 10])
plt.show()

df_filtered2.to_csv('/home/jsarrato/Physics/PhD/Paper-GraphSimProfiles/Data/file_labels_NIHAOandAURIGA_PCAfiltered.csv', index = False)