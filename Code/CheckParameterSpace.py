"""for pkl_file in pkl_files:
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
        scalar_data.append(scalar_dict)"""


import pickle
import glob
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import corner
import numpy as np

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

    for obj in galdata:
        scalar_dict = {
            'path': obj.path,
            'name': obj.name,
            'Vmax_g': float(obj.Vmax_g),
            'Rmax_g': float(obj.Rmax_g),
            'K_rot': float(obj.K_rot),
            'K_rot_ordered': float(obj.K_rot_salva),
            'Rvir': float(obj.Rvir),
            'Mvir': float(obj.Mvir),
            'Ms': float(obj.Ms),
            'Mh': float(obj.Mh),
            'Rhl': float(obj.Rhl),
            'eps_s': float(obj.eps_s),
            'eps_dm': float(obj.eps_dm),
        }
        scalar_data.append(scalar_dict)

df = pd.DataFrame(scalar_data)
df = df.dropna().reset_index(drop=True)

df['simflag'] = df['name'].apply(lambda x: 1 if x.startswith('halo') else 0)

df.to_csv('/home/jsarrato/Physics/PhD/Paper-GraphSimProfiles/Data/file_labels_NIHAOandAURIGA_PCAnofiltered.csv', index=False)


colors = {0: 'blue', 1: 'red'}
labels = {0: 'NIHAO', 1: 'AURIGA'}

df2 = df.copy()

df2['Rvir'] = np.log10(df2['Rvir'])
df2['Mvir'] = np.log10(df2['Mvir'])
df2['Ms'] = np.log10(df2['Ms'])
df2['Mh'] = np.log10(df2['Mh'])
df2['Rhl'] = np.log10(df2['Rhl'])
df2['Rmax_g'] = np.log10(df2['Rmax_g'])
df2['Vmax_g'] = np.log10(df2['Vmax_g'])

grr = pd.plotting.scatter_matrix(df2.drop(columns=['name', 'path', 'simflag']), c=df2['simflag'].apply(lambda x: colors[x]), figsize=(15, 15), marker='.', hist_kwds={'bins': 20}, alpha=.1)
plt.savefig('scatter_matrix.png')
plt.show()

data = df2.drop(columns=['name', 'path', 'simflag'])

data_simflag_0 = data[df2['simflag'] == 0]
data_simflag_1 = data[df2['simflag'] == 1]

figure = corner.corner(data_simflag_0, labels=data_simflag_0.columns, color='blue', bins=20, smooth=1.0, levels=[0.68, 0.95], plot_contours=True, fill_contours=True, hist_kwargs={'alpha': 0.6})
corner.corner(data_simflag_1, fig=figure, labels=data_simflag_1.columns, color='red', bins=20, smooth=1.0, levels=[0.68, 0.95], plot_contours=True, fill_contours=True, hist_kwargs={'alpha': 0.6})
plt.savefig('corner_plot.png')
plt.show()

features = df.drop(columns=['path', 'name', 'simflag'])
features_standardized = (features - features.mean()) / features.std()

pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_standardized)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['simflag'] = df['simflag']
pca_df['eps_s'] = features_standardized['eps_s']
pca_df['eps_dm'] = features_standardized['eps_dm']

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['simflag'].apply(lambda x: colors[x]), alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Galaxy Data')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in colors]
plt.legend(handles, [labels[i] for i in colors], title="Simulation")

# Highlight the region with PC1 < 1.5 and PC2 < -0.85
rect = plt.Rectangle((-2.16, -3.5), 1.5 - (-2.16), -0.85 - (-3.5), linewidth=1, edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)

plt.savefig('pca_simflag.png')
plt.show()

# Zoom-in plot
plt.figure(figsize=(8, 6))
zoom_mask = (pca_df['PC1'] < 1.5) & (pca_df['PC2'] < -0.85)
scatter_zoom = plt.scatter(pca_df[zoom_mask]['PC1'], pca_df[zoom_mask]['PC2'], c=pca_df[zoom_mask]['simflag'].apply(lambda x: colors[x]), alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Zoom-in PCA of Galaxy Data')
plt.legend(handles, [labels[i] for i in colors], title="Simulation")
plt.savefig('pca_simflag_zoom.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['eps_s'], alpha=0.5)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Galaxy Data')
plt.savefig('pca_eps_s.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['eps_dm'], alpha=0.5)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Galaxy Data')
plt.savefig('pca_eps_dm.png')
plt.show()

pca_components = pd.DataFrame(pca.components_, columns=features.columns, index=['PC1', 'PC2'])

print("Contribution of each feature to the PCA components:")
print(pca_components)

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
plt.savefig('pca_contributions.png')
plt.show()

mask = (pca_df['PC1'] < 1.5) & (pca_df['PC2'] < -0.85)
df_filtered = df[mask].reset_index(drop=True)

df_filtered['Rvir'] = np.log10(df_filtered['Rvir'])
df_filtered['Mvir'] = np.log10(df_filtered['Mvir'])
df_filtered['Ms'] = np.log10(df_filtered['Ms'])
df_filtered['Mh'] = np.log10(df_filtered['Mh'])
df_filtered['Rhl'] = np.log10(df_filtered['Rhl'])
df_filtered['Rmax_g'] = np.log10(df_filtered['Rmax_g'])
df_filtered['Vmax_g'] = np.log10(df_filtered['Vmax_g'])

grr = pd.plotting.scatter_matrix(df_filtered.drop(columns='simflag'), c=df_filtered['simflag'], figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.savefig('scatter_matrix_filtered.png')
plt.show()

mask2 = (df_filtered['eps_dm'] < 0.7) & (df_filtered['eps_s'] > 0.1)
df_filtered2 = df_filtered[mask2].reset_index(drop=True)

grr = pd.plotting.scatter_matrix(df_filtered2.drop(columns='simflag'), c=df_filtered2['simflag'].apply(lambda x: colors[x]), figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=10, alpha=.8)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in colors]
plt.legend(handles, [labels[i] for i in colors], title="Simulation", loc=[1.05, 10])
plt.savefig('scatter_matrix_filtered2.png')
plt.show()

df_filtered2.to_csv('/home/jsarrato/Physics/PhD/Paper-GraphSimProfiles/Data/file_labels_NIHAOandAURIGA_PCAfiltered.csv', index=False)

data = df_filtered2.drop(columns=['name', 'path', 'simflag'])

data_simflag_0 = data[df_filtered2['simflag'] == 0]
data_simflag_1 = data[df_filtered2['simflag'] == 1]

figure = corner.corner(data_simflag_0, labels=data_simflag_0.columns, color='blue', bins=20, smooth=1.0, levels=[0.68, 0.95], plot_contours=True, fill_contours=True, hist_kwargs={'alpha': 0.6})
corner.corner(data_simflag_1, fig=figure, labels=data_simflag_1.columns, color='red', bins=20, smooth=1.0, levels=[0.68, 0.95], plot_contours=True, fill_contours=True, hist_kwargs={'alpha': 0.6})
plt.savefig('corner_plot_filtered.png')
plt.show()
