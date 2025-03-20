import numpy as np
import pynbody
import sys
import os
import pandas as pd
import warnings
import math
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from torch_geometric.transforms import Distance

MyGraph = KNNGraph(k = 5, loop = False, force_undirected = True)
MyDistance = Distance(norm = False, cat = True)


warnings.filterwarnings("ignore")

def fibonacci_sphere(samples=100):

	xlist = np.zeros(samples)
	ylist = np.zeros(samples)
	zlist = np.zeros(samples)
	phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

	for i in range(samples):
		y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
		radius = math.sqrt(1 - y * y)  # radius at y

		theta = phi * i  # golden angle increment

		x = math.cos(theta) * radius
		z = math.sin(theta) * radius

		xlist[i] = x
		ylist[i] = y
		zlist[i] = z
	return xlist, ylist, zlist

def half_light_r_count_XYZ(X,Y,Z, cylindrical=False):


    half_l = int(len(X) * 0.5)

    if cylindrical:
        R = np.sqrt(X**2+Y**2)
    else:
        R = np.sqrt(X**2+Y**2+Z**2)
        
    max_high_r = np.max(R)
    
    test_r = 0.5 * max_high_r
    
    testrf = R<test_r
    
    min_low_r = 0.0
    
    test_l = int(len(X[testrf]))
    
    it = 0
    while ((np.abs(test_l - half_l) / half_l) > 0.01):
        it = it + 1
        if (it > 100):
            break

        if (test_l > half_l):
            test_r = 0.5 * (min_low_r + test_r)
        else:
            test_r = (test_r + max_high_r) * 0.5
        testrf = R<test_r
        test_l = int(len(X[testrf]))

        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r

    return test_r
	
#################################################################################################
# DEFINE WORKING PATH, READ DATA AND MASK IT
#################################################################################################

main_file_folder = '/scratch/jsarrato/Paper-GraphSimProfiles/work/'
#main_file_folder = '/home/jorge/Physics/PhD/Paper3/Work/'

gal_data = pd.read_csv(main_file_folder + 'file_labels_NIHAOandAURIGA_PCAfiltered.csv')

"""gal_data = gal_data[gal_data['name'].str.endswith('h1')]
gal_data = gal_data.sort_values('m_r3').reset_index(drop = True)"""

paths = gal_data['path']

#################################################################################################
# DEFINE VARIBALES
#################################################################################################

n_processes = int(sys.argv[1])
n_process = int(sys.argv[2])

#n_processes = 1
#n_process = 1

N_sphere_points = 64


x,y,z = fibonacci_sphere(N_sphere_points)

n_project = N_sphere_points

filename = 'proj_data_NIHAO_and_AURIGA_PCAfilt_samplearr' + str(n_process) + '.csv'

n_gals = len(paths)


startindex = int((n_process-1)*n_gals/n_processes)
print(startindex)
endindex = int(n_process*n_gals/n_processes)
if n_process == n_processes:
   endindex = n_gals+1

count = n_project * startindex
current_path = ''
current_halo = ''

stellar_quantities_to_save = ['age', 'HII', 'HeIII', 'hetot', 'hydrogen', 'ne', 'feh', 'ofe', 'u_mag', 'b_mag', 'v_mag', 'r_mag', 'i_mag', 'j_mag', 'h_mag', 'k_mag']


r_array = np.arange(0.2,4,0.1)
r_array = np.append(r_array,np.array([1.7])) # Amorisco & Evans
r_array = np.append(r_array,np.array([10000])) # e.g Errani
r_array = np.append(r_array,np.array([4/3])) # Wolf
r_array = np.append(r_array,np.array([1.04])) # Campbell
rheader_list = list(['r'+str(ii+1) for ii in range(np.size(r_array))])
mrheader_list_circ = list(['m_r_circ'+str(ii+1) for ii in range(np.size(r_array))])
dispsheader_list_circ = list(['disp_r_circ'+str(ii+1) for ii in range(np.size(r_array))])

headers = ['path','name','i_index','count','angx','angy','angz']

print(main_file_folder + filename)
                    
n_projs_computed = 0
if os.path.exists(main_file_folder + filename):
  print('existe')
  existing_csv = pd.read_csv(main_file_folder + filename)
  startindex = existing_csv['i_index']
  startindex = startindex[np.size(startindex)-1]+1
  count = existing_csv['count']
  count = count[np.size(count)-1]
  n_projs_computed = len(existing_csv[existing_csv['i_index']==startindex-1])
  
  N_projs_necessary = int(gal_data[gal_data['name']==existing_csv.iloc[-1]['name']]['count'])

  if n_projs_computed < N_projs_necessary:
    startindex-=1
  else:
    n_projs_computed = 0

print(startindex)
# For each copy of a galaxy
for i in range(startindex, endindex):
  N_sphere_points = 64

  data_list = []

  if N_sphere_points > 1:
    x,y,z = fibonacci_sphere(N_sphere_points)
  else:
    x = [1]
    y = [1]
    z = [1]

  n_project = N_sphere_points

  new_gal_data = pd.DataFrame(columns = headers)

  galname = gal_data['name'][i]

  galpath = paths[i]

  print(galname)

  galname_original = galname

  halonumstr = galname[galname.rfind('h')+1:]
  halonum = int(halonumstr)
  galname = galname[:galname.rfind('h')]
  

  # if not already in memory, charge the simulation and center the galaxy
  if galpath != current_path:

    s = pynbody.load(galpath)

    s.physical_units()
    h = s.halos()
    
    current_path = galpath
    current_halo = ''
    
    print('Charged '+galpath)

  if current_halo != halonumstr:
    print('Centering halo '+halonumstr)
    try:
      h1 = h[halonum]
      
      pynbody.analysis.halo.center(h1)

      try:
        h1 = h1[h1['r']<h1.properties['Rvir']]
      except:
        h1 = h1[h1['r']<h1.properties['Rhalo']]

      try:
        h1.s = h1.s[h1.s['aform']>0.]
      except:
        try:
          h1.s = h1.s[h1.s['age']>0.]
        except:
          print('Could not separate wind particles')


      try:
        pynbody.analysis.angmom.faceon(h1, use_stars = True)
      except:
        pynbody.analysis.angmom.faceon(h1, use_stars = False)


      current_halo = halonumstr

      centered = True
      print('Centering done')
    except:
       centered = False
       print('Centering failed')
  else:
    print('Halo '+halonumstr+'already centered')

  n_stars = 1000
  if n_stars > len(h1.s['x']):
    n_stars = len(h1.s['x'])
  if n_stars<50:
     centered = False
     

  p = pynbody.analysis.profile.Profile(h1, rmin = 0.001, rmax = 150, ndim = 3, type = 'lin',nbins = 100000)

  try:
    indeces2 = np.random.choice(np.arange(0,len(h1.s['x'])), size = n_stars, replace = False)
  except:
    print('less than 200 stars?')
    indeces = np.arange(0,len(h1.s['x']))

  hlrstd_arr = np.zeros((2,n_project))
  
  masses_arr = np.zeros((len(r_array),n_project))


  for project in range(n_projs_computed,n_project):

      print(n_stars, len(h1.s['x']))
      XYZ_0 = np.vstack((h1.s['x'],h1.s['y'],h1.s['z']))
      VXYZ_0 = np.vstack((h1.s['vx'],h1.s['vy'],h1.s['vz']))

      new_gal_data = pd.DataFrame(columns = headers)
      print(str(i+1)+'/'+str(endindex)+' , '+str(project+1)+'/'+str(n_project),count)
      if centered:

          count+=1

          nv = np.array([x[project], y[project], z[project]])  # Replace a, b, c with your normal vector components
          nv /= np.linalg.norm(nv)  # Normalize the normal vector

          rotmatrix = np.array([[nv[1]/np.sqrt(nv[0]**2 + nv[1]**2), -nv[0]/np.sqrt(nv[0]**2 + nv[1]**2) ,0],
                                [nv[0]*nv[2]/np.sqrt(nv[0]**2 + nv[1]**2), nv[1]*nv[2]/np.sqrt(nv[0]**2 + nv[1]**2), -np.sqrt(nv[0]**2 + nv[1]**2)],
                                [nv[0], nv[1], nv[2]]])



          XYZ = np.matmul(rotmatrix, XYZ_0)
          X = XYZ[0,:]
          Y = XYZ[1,:]
          Z = XYZ[2,:]

          RXY = np.sqrt(X**2+Y**2)

          VXYZ = np.matmul(rotmatrix, VXYZ_0)
          VX = VXYZ[0,:]
          VY = VXYZ[1,:]
          VZ = VXYZ[2,:]


          #try:
          hlr_true = half_light_r_count_XYZ(X,Y,Z,cylindrical=True)
          hlr = half_light_r_count_XYZ(X[indeces2],Y[indeces2],Z[indeces2],cylindrical=True)
          print(hlr)

          r_arr1 = hlr*r_array

          X = X[indeces2]
          Y = Y[indeces2]
          Z = Z[indeces2]
          VZ = VZ[indeces2]
            

      else:
          r_arr1 = np.ones_like(r_array)*-9999


          hfile = open(main_file_folder + '/Logs/'+galname+'_CenterError.dat','w')
          hfile.write(galname+'  '+str(halonum)+'  '+galpath+'  ')
          hfile.close()

      gal_result = [galpath, galname_original ,i, count, x[project], y[project], z[project]]
      gal_dict = {headers[i]: [gal_result[i]] for i in range(len(headers))}
      gal_line = pd.DataFrame(gal_dict)
      new_gal_data = pd.concat([new_gal_data, gal_line])

      if os.path.exists(main_file_folder + filename):
        new_gal_data.to_csv(main_file_folder + filename, index = False, mode = 'a', header = False)
      else:
        new_gal_data.to_csv(main_file_folder + filename, index = False, mode = 'w')

      if centered:
      
        r_proj = np.sqrt(X**2+Y**2)
        node_attrs = np.r_[np.array(r_proj).reshape(1,-1), np.array(VZ).reshape(1,-1)].T

        data = np.r_[np.array(X).reshape(1,-1), np.array(Y).reshape(1,-1), np.array(VZ).reshape(1,-1)].T

        print(data.shape)

        data_list.append(data)

        hlrstd_arr[0, project] = hlr
        hlrstd_arr[1, project] = np.std(np.array(VZ))
        
        torch.save(data_list, main_file_folder+f"arrs_NIHAO_and_AURIGA_PCAfilt_1000/posvel_{i}.pkl")

        np.save(main_file_folder+f"arrs_NIHAO_and_AURIGA_PCAfilt_1000/hlrstd_arr{i}", hlrstd_arr)
        
        np.savez(main_file_folder+f"arrs_NIHAO_and_AURIGA_PCAfilt_1000/mass_interp{i}.npz", x=p['rbins'], y=p['mass_enc'])

        stellar_quantities = np.zeros((len(stellar_quantities_to_save), n_stars))
        for j, quantity in enumerate(stellar_quantities_to_save):
            stellar_quantities[j] = h1.s[quantity][indeces2]

        np.save(main_file_folder+f"arrs_NIHAO_and_AURIGA_PCAfilt_1000/steallar_quantities{i}", stellar_quantities)

        np.save(main_file_folder+f"arrs_NIHAO_and_AURIGA_PCAfilt_1000/softenings{i}", np.array([min(h1.s['eps'][indeces2]), min(h1.d['eps'][indeces2])]))
          
  n_projs_computed = 0
