#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:06:18 2022

@author: dixinchen
"""
# %% import + sns_set
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import seaborn as sns
sns.set(style="white", font_scale=1.1)
plt.rcParams['font.family'], plt.rcParams['axes.linewidth'] = 'DejaVu Sans', 1.5
plt.rcParams['xtick.bottom'], plt.rcParams['ytick.left'] = True, True

# %% define address of the data files

# address = '/home/dixinchen/1e6/'
address = '/home/dixinchen/1e6-100bins/'

# %% method to load data, method to return the index of one number

def LoadData(filename):
    
    fp = open(filename)
    rdr = csv.reader(filter(lambda row: row[0]!='#', fp))
    data = []
    for row in rdr:
        data.append(row)
    fp.close()
    data = np.array([[float(y) for y in x] for x in data])

    return data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def getIndex(x, x_i):
    return np.where(x == find_nearest(x, x_i))[0][0]

# %% define the number of incident protons, scoring geometry and energy

n_p = 1e6

# binx = 50
# biny = 50
# binz = 50
# bin_energy = 50

binx = 100
biny = 100
binz = 100
bin_energy = 100

# binx = 3
# biny = 4
# binz = 5
# bin_energy = 6

abs_x = 50
abs_y = 50
abs_z = 50

min_energy = 0.01
max_energy = 260.01
energy = np.linspace(min_energy, max_energy, bin_energy)

# %% load dose

fn = address + 'DoseAtWaterTank.csv'
data_dose = LoadData(fn)

# %% create spacial arrays

x = abs_x / binx * data_dose[::binz * biny, 0] - abs_x/2
y = abs_y / biny * data_dose[:binz * biny-1:binz,1] - abs_y/2
z = abs_z / binz * data_dose[0:binz,2]

# check if the bins are correct
if binx != len(x):
    print("Wrong binx >_<")
    sys.exit()
if biny != len(y):
    print("Wrong biny >_<")
    sys.exit()
if binz != len(z):
    print("Wrong binz >_<")
    sys.exit()
    
    
# %% create treatment dose and its standard deviation (sd)

proton_dose = np.zeros((binx, biny, binz))
for i in range(len(data_dose)):
    proton_dose[int(data_dose[i,0]), int(data_dose[i,1]), int(data_dose[i,2])] = data_dose[i,3]
    
proton_dose_sd = np.zeros((binx, biny, binz))
for i in range(len(data_dose)):
    proton_dose[int(data_dose[i,0]), int(data_dose[i,1]), int(data_dose[i,2])] = data_dose[i,4]
    


# %% plot 2D heatmap of treatment dose (Gy)
fig_dosemap = plt.figure(figsize=(10,4))
plt.imshow(np.sum(proton_dose*1e9, axis=1), aspect="auto", 
                  extent=[min(z),max(z),min(x),max(x)], 
                  cmap='plasma')
cbar = plt.colorbar()
cbar.set_label('Proton dose (nGy)')
plt.xlabel('Depth [cm]')
plt.ylabel('X [cm]')
plt.tight_layout()
plt.show()



# %% create treatment depth dose (Gy)
# this is summing over x and y
# dose vs depth (z)

proton_pdd = np.sum(np.sum(proton_dose, axis=0), axis=0)
# proton_pdd = proton_pdd/max(proton_pdd)*100

# %% plot depth dose from depth = "probe_z" to the bottom of the water tank

probe_z = 0
probe_z_idx = np.where(z == find_nearest(z, probe_z))

fig_pdd = plt.figure(figsize=(6, 6/1.6))
plt.plot(z[probe_z_idx[0][0]:], proton_pdd[probe_z_idx[0][0]:],'k.')
plt.xlabel('Depth [cm]')
plt.ylabel('Proton PDD [%]')
plt.tight_layout()
plt.show()


# %% create lateral beam profile at the bragg peak
bragg_z_idx = np.where(proton_pdd == find_nearest(proton_pdd, max(proton_pdd)))[0][0]
lat_dose = np.sum(proton_dose[:,:,bragg_z_idx],axis=1)

fig_lat_prof_bragg = plt.figure(figsize=(6, 6/1.6))
plt.plot(x, lat_dose*1e9, 'k.')
plt.xlabel('X [cm]')
plt.ylabel('Proton dose [nGy]')
plt.tight_layout()
plt.show()


# %% add finer z bins

# fn = '/home/dixinchen/bin-z-1000/DoseAtWaterTank.csv'
# data_fine_z_dose = LoadData(fn)

# fbinx = 10
# fbiny = 10
# fbinz = 48
# fabs_x = 50
# fabs_y = 50
# fabs_z = 4.8
# fine_proton_dose = np.zeros((fbinx, fbiny, fbinz))
# for i in range(len(data_fine_z_dose)):
#     fine_proton_dose[int(data_fine_z_dose[i,0]), int(data_fine_z_dose[i,1]), int(data_fine_z_dose[i,2])] = data_fine_z_dose[i,3]
# fx = fabs_x / fbinx * data_fine_z_dose[::fbinz * fbiny, 0] - fabs_x/2
# fy = fabs_y / fbiny * data_fine_z_dose[:fbinz * fbiny-1:fbinz,1] - fabs_y/2
# fz = fabs_z / fbinz * data_fine_z_dose[0:fbinz,2] + 30.1

# plt.plot(fz,np.sum(np.sum(fine_proton_dose, axis=0), axis=0),'k.')

# fine_proton_dose = np.concatenate((np.sum(np.sum(proton_dose, axis=0), axis=0),
#                                       np.sum(np.sum(fine_proton_dose, axis=0), axis=0)))
# fine_proton_dose_pdd = fine_proton_dose/max(fine_proton_dose)*100

# fz = np.concatenate((z, fz))

# fig_fine_pdd = plt.figure(figsize=(6, 6/1.6))
# plt.plot(fz, fine_proton_dose_pdd,'k.')
# plt.xlabel('Depth [cm]')
# plt.ylabel('Proton PDD [%]')
# plt.tight_layout()
# plt.show()
# %% load neutron fluence data

fn = address + 'FluenceSpectra.csv'
data_fluence = LoadData(fn)

# %% create fluence and it sd (unit: numbers of particle per unit area, cm^(-2))
fluence = np.zeros((binx, biny, binz, bin_energy))
for i in range(len(data_fluence)):
    for j in range(bin_energy):
        fluence[int(data_fluence[i,0]), int(data_fluence[i,1]), int(data_fluence[i,2])][j] = data_fluence[i,5:-5:2][j]

fluence_sd = np.zeros((binx, biny, binz, bin_energy))
for i in range(len(data_fluence)):
    for j in range(bin_energy):
        fluence_sd[int(data_fluence[i,0]), int(data_fluence[i,1]), int(data_fluence[i,2])][j] = data_fluence[i,6:-4:2][j]

# %% normalizing the neutron fluence
norm_fluence = fluence / n_p

# %% [commented] probeFluenceSumXY
def probeFluenceSumXY(norm_fluence, probe_z):
    probe_z_idx = np.where(z == find_nearest(z, probe_z))[0][0]
    probe_fluence = np.sum(norm_fluence[:,:,probe_z_idx,:],axis=(0,1))
    return probe_fluence

# def probeFluenceAtVoxel(dim4data, probe_z, probe_x, probe_y):
#     probe_z_idx = np.where(z == find_nearest(z, probe_z))[0][0]
#     probe_x_idx = np.where(x == find_nearest(x, probe_x))[0][0]
#     dim4data = dim4data[x,:,z,:]

    
# %% create neutron fluence at the chosen depth (probe_z) summed along x and y
probe_z1 = 32
probe_fluence_z1 = probeFluenceSumXY(norm_fluence, probe_z1)

probe_z2 = 37
probe_fluence_z2 = probeFluenceSumXY(norm_fluence, probe_z2)

probe_z3 = 37+5
probe_fluence_z3 = probeFluenceSumXY(norm_fluence, probe_z3)

# %% calculate the average energy
def fluenceAvgEnergy(probe_fluence, energy):
    return np.sum(probe_fluence*energy)/np.sum(probe_fluence)

avg_e_z1 = fluenceAvgEnergy(probe_fluence_z1, energy)
avg_e_z2 = fluenceAvgEnergy(probe_fluence_z2, energy)
avg_e_z3 = fluenceAvgEnergy(probe_fluence_z3, energy)

# %% plot neutron fluence vs energy at depth z1
n_fluence = plt.figure(figsize=(7.5, 7.5/1.6))
plt.step(energy, probe_fluence_z1,"k", linewidth=1., label='At BP, $\\overline{E}_N$ = %.1f MeV' % (avg_e_z1))
# plt.step(energy, probe_fluence_z2,"k", linestyle='--', linewidth=0.7, label='5 cm below BP, $\\overline{E}_N$ = %.1f MeV' % (avg_e_z2)) #probe_z2, 
# plt.step(energy, probe_fluence_z3,"k", linestyle=':', linewidth=0.7, label='10 cm below BP, $\\overline{E}_N$ = %.1f MeV' % (avg_e_z3))
plt.xlim([0, 260.01])
plt.xlabel('Neutron energy [MeV]')
plt.ylabel('Neutron fluence / $10^6$ protons') #  / $10^5$ protons
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

# # plot neutron fluence vs energy at depth z1 and z2
# f = plt.figure(figsize=(6,12))
# ax1 = f.add_subplot(211)
# ax1.step(energy, probe_fluence_z1,"k")
# ax1.set_xlabel('Neutron energy [MeV]')
# ax1.set_ylabel('Neutron fluence / $10^6$ protons') 
# ax1.set_yscale("log")
# ax1.set_xlim([0, 260.01])
# ax2 = f.add_subplot(212)
# ax2.step(energy, probe_fluence_z2,"k")
# ax2.set_xlabel('Neutron energy [MeV]')
# ax2.set_ylabel('Neutron fluence / $10^6$ protons') 
# ax2.set_yscale("log")
# ax2.set_xlim([0, 260.01])
# plt.show()

# %% plot neutron fluence vs energy in one voxel


# v1 = [0,0,32+5] # in the center, 5 cm downstream from the bp
# v2 = [10,0,32+5] # 5 cm laterally from the bp, 5 cm downstream from the bp
# v3 = [15,0,32+5]
# # v1_fl = norm_fluence[getIndex(x,v1[0]),getIndex(y,v1[1]),getIndex(z,v1[2]),:]
# # v2_fl = norm_fluence[getIndex(x,v2[0]),getIndex(y,v2[1]),getIndex(z,v2[2]),:]

# a = 2
# v1_fl = norm_fluence[getIndex(x,v1[0])-a:getIndex(x,v1[0])+a,getIndex(y,v1[1])-a:getIndex(y,v1[1])+a,getIndex(z,v1[2])-a:getIndex(z,v1[2])+a,:]
# v2_fl = norm_fluence[getIndex(x,v2[0])-a:getIndex(x,v2[0])+a,getIndex(y,v2[1])-a:getIndex(y,v2[1])+a,getIndex(z,v2[2])-a:getIndex(z,v2[2])+a,:]
# v3_fl = norm_fluence[getIndex(x,v3[0])-a:getIndex(x,v3[0])+a,getIndex(y,v3[1])-a:getIndex(y,v3[1])+a,getIndex(z,v3[2])-a:getIndex(z,v3[2])+a,:]

# # v1_fl = norm_fluence[getIndex(x,v1[0])-a:getIndex(x,v1[0])+a,:,getIndex(z,v1[2])-a:getIndex(z,v1[2])+a,:]
# # v2_fl = norm_fluence[getIndex(x,v2[0])-a:getIndex(x,v2[0])+a,:,getIndex(z,v2[2])-a:getIndex(z,v2[2])+a,:]
# # v3_fl = norm_fluence[getIndex(x,v3[0])-a:getIndex(x,v3[0])+a,:,getIndex(z,v3[2])-a:getIndex(z,v3[2])+a,:]

# v1_fl = np.sum(v1_fl, axis=(0,1,2))
# v2_fl = np.sum(v2_fl, axis=(0,1,2))
# v3_fl = np.sum(v3_fl, axis=(0,1,2))

# # v1_fl2 = []
# # v2_fl2 = []
# # for i in range(int(len(v1_fl)/2)):
# #     v1_fl2.append(v1_fl[2*i]+v1_fl[2*i+1])
# #     v2_fl2.append(v2_fl[2*i]+v2_fl[2*i+1])

# v1_fl3 = []
# v2_fl3 = []
# v3_fl3 = []
# for i in range(int(len(v1_fl)/3)):
#     v1_fl3.append(v1_fl[3*i]+v1_fl[3*i+1]+v1_fl[3*i+2])
#     v2_fl3.append(v2_fl[3*i]+v2_fl[3*i+1]+v2_fl[3*i+2])
#     v3_fl3.append(v3_fl[3*i]+v3_fl[3*i+1]+v3_fl[3*i+2])


# avg_e_v1 = fluenceAvgEnergy(v1_fl3, energy[:-1:3])
# avg_e_v2 = fluenceAvgEnergy(v2_fl3, energy[:-1:3])
# avg_e_v3 = fluenceAvgEnergy(v3_fl3, energy[:-1:3])

# # v1_fl4 = []
# # v2_fl4 = []
# # for i in range(int(len(v1_fl)/4)):
# #     v1_fl4.append(v1_fl[4*i]+v1_fl[4*i+1]+v1_fl[4*i+2]+v1_fl[4*i+3])
# #     v2_fl4.append(v2_fl[4*i]+v2_fl[4*i+1]+v2_fl[4*i+2]+v2_fl[4*i+3])


# # plt.step(energy, v1_fl,"k", linewidth=1., label='x=0,z=37')
# # plt.step(energy, v2_fl,"k", linestyle='--', label='x=5,z=37')
# # plt.step(energy[::2], v1_fl2,"k", linewidth=1., label='x=0,z=37')
# # plt.step(energy[::2], v2_fl2,"k", linestyle='--', label='x=5,z=37')
# plt.step(energy[:-1:3], v1_fl3,"k", linewidth=1., label='x = 0 cm, $\\overline{E}_N$ = %.1f MeV' % (avg_e_v1))
# plt.step(energy[:-1:3], v2_fl3,"k",  linewidth=1.2, linestyle='--', label='x = 5 cm, $\\overline{E}_N$ = %.1f MeV' % (avg_e_v2))
# plt.step(energy[:-1:3], v3_fl3,"k", linewidth=1.7, linestyle=':', label='x = 10 cm, $\\overline{E}_N$ = %.1f MeV' % (avg_e_v3))
# plt.xlim([0, 260.01])
# plt.xlabel('Neutron energy [MeV]')
# plt.ylabel('Neutron fluence / $10^6$ protons') #  / $10^5$ protons
# plt.yscale("log")
# plt.legend()
# plt.tight_layout

# %% create voxel neutron fluence
def voxel_neutron_fluence(v1, norm_fluence, a, energy=energy, x=x, y=y, z=z):
    v1_fl = norm_fluence[getIndex(x,v1[0])-a:getIndex(x,v1[0])+a,getIndex(y,v1[1])-a:getIndex(y,v1[1])+a,getIndex(z,v1[2])-a:getIndex(z,v1[2])+a,:]
    v1_fl = np.sum(v1_fl, axis=(0,1,2))
    v1_fl3 = []
    for i in range(int(len(v1_fl)/3)):
        v1_fl3.append(v1_fl[3*i]+v1_fl[3*i+1]+v1_fl[3*i+2])
    if a == 3:
        avg_e_v1 = fluenceAvgEnergy(v1_fl3, energy[:-1:a])
        energy = energy[:-1:a]
    if a == 2:
        avg_e_v1 = fluenceAvgEnergy(v1_fl3, energy[::a])
        energy = energy[::a]
    return energy,v1_fl3,avg_e_v1,v1

v0 = np.sum(norm_fluence[getIndex(x,0),:,getIndex(z,32),:],axis=0)

v1 = voxel_neutron_fluence([0,0,32],norm_fluence,3)
v2 = voxel_neutron_fluence([0,0,32+5],norm_fluence,3)
v3 = voxel_neutron_fluence([0,0,32+10],norm_fluence,3)
v4 = voxel_neutron_fluence([0,0,32+5],norm_fluence,3)
v5 = voxel_neutron_fluence([5,0,32+5],norm_fluence,3)
v6 = voxel_neutron_fluence([10,0,32+5],norm_fluence,3)

# %% plot lateral plane and voxel neutron fluence together

f = plt.figure(figsize=(6,12))
ax2 = plt.subplot(313) #313

ax0 = plt.subplot(311,sharex=ax2) #311
ax0.step(energy, probe_fluence_z2,"k-", linewidth=1., label='5 cm below BP, $\\overline{E}_N$ = %.1f MeV' % (avg_e_z1))
# ax0.set_xlabel('Neutron energy [MeV]')
ax0.set_ylabel('Neutron fluence / $10^6$ protons') 
ax0.set_yscale("log")
ax0.set_xlim([0, 260.01])
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.tick_params(which='both', direction='in')
ax0.text(0.03, 0.06, '50$\\times$50 cm$^2$ lateral plane at depth = 37 cm', fontsize=12, transform=ax0.transAxes)
ax0.legend()

ax1 = plt.subplot(312,sharex=ax2) #312
ax1.step(v1[0], v1[1],"k", linewidth=1., linestyle='-', label='Depth = 32 cm, $\\overline{E}_N$ = %.1f MeV' % (v1[2]))
ax1.step(v2[0], v2[1],"k", linewidth=1.05, linestyle='--', label='Depth = 37 cm, $\\overline{E}_N$ = %.1f MeV' % (v2[2]))
ax1.step(v3[0], v3[1],"k", linewidth=1.7, linestyle=':', label='Depth = 42 cm, $\\overline{E}_N$ = %.1f MeV' % (v3[2]))
#ax1.set_xlabel('Neutron energy [MeV]')
ax1.set_ylabel('Neutron fluence / $10^6$ protons') 
ax1.set_yscale("log")
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.legend()
ax1.set_xlim([0, 260.01])
ax1.text(0.03, 0.06, '2 cm$^3$ voxel at x = 0 cm', fontsize=12, transform=ax1.transAxes)
ax1.tick_params(which='both', direction='in')

ax2.step(v4[0], v4[1],"k", linewidth=1., label='x = 0 cm, $\\overline{E}_N$ = %.1f MeV' % (v4[2]))
ax2.step(v5[0], v5[1],"k", linewidth=1.05, linestyle='--', label='x = 5 cm, $\\overline{E}_N$ = %.1f MeV' % (v5[2]))
ax2.step(v6[0], v6[1],"k", linewidth=1.7, linestyle=':', label='x = 10 cm, $\\overline{E}_N$ = %.1f MeV' % (v6[2]))
ax2.set_xlabel('Neutron energy [MeV]')
ax2.set_ylabel('Neutron fluence / $10^6$ protons') 
ax2.set_yscale("log")
ax2.tick_params(which='both', direction='in')
ax2.legend(loc=1)
ax2.text(0.03, 0.06, '2 cm$^3$ voxel at depth = 37 cm', fontsize=12, transform=ax2.transAxes)
ax2.set_xlim([0, 260.01])

plt.tight_layout()
plt.show()



# %% load neutron ambient dose equivalent data

fn = address + 'AmbientDoseNeutronPerSourceNeutron.csv'
data_n_dose = LoadData(fn)

# %% create dose equivalent and it sd (unit: Sv)

n_dose = np.zeros((binx, biny, binz, bin_energy))
for i in range(len(data_n_dose)):
    for j in range(bin_energy):
        n_dose[int(data_n_dose[i,0]), int(data_n_dose[i,1]), int(data_n_dose[i,2])][j] = data_n_dose[i,5:-5:2][j]

n_dose_sd = np.zeros((binx, biny, binz, bin_energy))
for i in range(len(data_n_dose)):
    for j in range(bin_energy):
        n_dose_sd[int(data_n_dose[i,0]), int(data_n_dose[i,1]), int(data_n_dose[i,2])][j] = data_n_dose[i,6:-4:2][j]

# %%
save_address = '/home/dixinchen/Plotting/mar-06-save/'

np.save(save_address + "proton_dose.npy", proton_dose)
np.save(save_address + "proton_dose_sd.npy", proton_dose_sd)

np.save(save_address + "fluence.npy", fluence)
np.save(save_address + "fluence_sd.npy", fluence_sd)

np.save(save_address + "n_dose.npy", n_dose)
np.save(save_address + "n_dose_sd.npy", n_dose_sd)

# %% normalize the neutron dose
norm_n_dose = n_dose / (proton_dose*1e9)

# %% create neutron dose at the chosen depth (probe_z) summed along x and y
probe_z1 = 20
probe_n_dose_z1 = probeFluenceSumXY(norm_n_dose, probe_z1)

probe_z2 = 35
probe_n_dose_z2 = probeFluenceSumXY(norm_n_dose, probe_z2)

# %% plot neutron fluence vs energy at depth z1
fig_n_dose = plt.figure(figsize=(7.5, 7.5/1.6))
plt.step(energy, probe_n_dose_z1,"k", linewidth=1., label='depth at %d cm' % (probe_z1))
plt.step(energy, probe_n_dose_z2,"k", linestyle='--', linewidth=0.7, label='depth at %d cm' % (probe_z2))
plt.xlim([0, 260.01])
plt.xlabel('Neutron energy [MeV]')
plt.ylabel('Neutron dose / treatment Gy [Sv/nGy]') #  / $10^5$ protons
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

# %% create neutron dose voxel at x=x_i, z=z_i, y=0
probe_x1 = 0
probe_x2 = 10
probe_z1 = 32+5

probe_x1_idx = getIndex(x,0)
probe_x2_idx = getIndex(x,10)
probe_z1_idx = getIndex(z,32+5)

nd_1 = np.sum(norm_n_dose[probe_x1_idx, probe_x1_idx-2:probe_x1_idx+2, probe_z1_idx, :], axis=0)
nd_2 = np.sum(norm_n_dose[probe_x2_idx, probe_x1_idx-2:probe_x1_idx+2, probe_z1_idx, :], axis=0)


# nd_1 = norm_n_dose[probe_x1_idx, probe_x1_idx, probe_z1_idx, :]
# nd_2 = norm_n_dose[probe_x2_idx, probe_x1_idx, probe_z1_idx, :]

# %%
plt.step(energy, nd_1, label='X = %d cm' % (probe_x1))
plt.step(energy, nd_2, label='X = %d cm' % (probe_x2))
plt.yscale("log")
plt.legend()

# %% plot neutron fluence at the chosen depth (probe_z) summed along x and y

def probeProtonDose(proton_dose, probe_z):
    probe_z_idx = np.where(z == find_nearest(z, probe_z))[0][0]
    probe_dose = np.sum(proton_dose[:,:,probe_z_idx],axis=1)
    return probe_dose

probe_z1 = 20
probe_pdose_z1 = probeProtonDose(proton_dose, probe_z1)

probe_z2 = 35
probe_pdose_z2 = probeProtonDose(proton_dose, probe_z2)

fig_probe_pdd = plt.figure(figsize=(6, 6/1.6))
# plt.plot(x, probe_pdose_z1*1e9, 'k--', label='20cm')
plt.plot(x, probe_pdose_z2*1e9, 'k-', label='At depth %d cm'% (probe_z2))
plt.plot(x, lat_dose*1e9, 'b--', label='At the Bragg peak')
plt.xlabel('X [cm]')
plt.ylabel('Proton dose [nGy]')
plt.legend()
plt.tight_layout()
plt.show()




# %% create NeutronDoseAtWaterTank
fn = address + 'NeutronDoseAtWaterTank.csv'
data_NeutronDoseAtWaterTank = LoadData(fn)

NeutronDoseAtWaterTank = np.zeros((binx, biny, binz))
for i in range(len(data_NeutronDoseAtWaterTank)):
    NeutronDoseAtWaterTank[int(data_NeutronDoseAtWaterTank[i,0]), int(data_NeutronDoseAtWaterTank[i,1]), int(data_NeutronDoseAtWaterTank[i,2])] = data_NeutronDoseAtWaterTank[i,3]
    
# %% plot neutron dose
fig_NeutronDoseAtWaterTank = plt.figure(figsize=(10,4))
plt.imshow(np.sum(NeutronDoseAtWaterTank*1e9, axis=1), aspect="auto", 
                  extent=[min(z),max(z),min(x),max(x)], 
                  cmap='plasma')
cbar = plt.colorbar()
cbar.set_label('Neutron dose [nGy]')
plt.xlabel('Depth [cm]')
plt.ylabel('X [cm]')
plt.tight_layout()
plt.show()





# %% create neutron spectra at one voxel

bragg_peak_z = 12+25

# coordinate for the probing voxel
probe_v1 = [0,0,bragg_peak_z+5]
probe_v2 = [5,0,bragg_peak_z+5]
probe_v3 = [10,0,bragg_peak_z+5]
probe_v4 = [5,0,bragg_peak_z]
probe_v5 = [10,0,bragg_peak_z]

probes = [probe_v1, probe_v2, probe_v3, probe_v4, probe_v5]

probe_voxel_fl = np.zeros((len(probes),bin_energy))
for i in range(len(probes)):
    for j in range(bin_energy):
        xind = np.where(x == find_nearest(x, probes[i][0]))
        yind = np.where(y == find_nearest(y, probes[i][1]))
        zind = np.where(z == find_nearest(z, probes[i][2]))
        probe_voxel_fl[i,j] = norm_fluence[xind,yind,zind,j]

# plt.plot(energy, probe_voxel_fl[0],'.',color='grey',alpha=1) 
alpha = np.linspace(1,0.1,5)
for i in range(len(probes)-2):
    plt.step(energy, probe_voxel_fl[i], color='grey',alpha=alpha[i])
plt.legend(['v1','v2','v3','v4','v5'])
plt.xlim([0, 260.01])
plt.xlabel('Neutron energy [MeV]')
plt.ylabel('Neutron fluence / $10^6$ protons') #  / $10^5$ protons
plt.yscale("log")
plt.tight_layout()



# %% calculate the uncertainty in noramlized neutron fluence at z = "probe_z"

def normFluenceError(fluence_sd, probe_fluence, n_p, probe_z):
    probe_z_idx = np.where(z == find_nearest(z, probe_z))[0][0]
    # the uncertainty of the sum in each voxel is the sd in that voxel divided by sqrt of the number of histories
    sigma_fl_2 = (fluence_sd/np.sqrt(n_p))**2
    
    # the uncertainty of the fluence at probe_z (i.e. summing along x and y) is the quadrature sum of the uncertainty for each voxel
    sigma_Fl = np.sum(np.sum(sigma_fl_2[:,:,probe_z_idx,:],axis=0),axis=0)
    
    # the uncertainty of normalized fluence at probe_z is calculated from error propagation
    # if a = b/c
    # then (sigma_a/a)^2 = (sigma_b/b)^2 + (sigma_c/c)^2
    # note that the sigma of n_p is sqrt(n_p) since it follows Poisson distribution
    probe_fl_err = np.sqrt((sigma_Fl/probe_fluence)**2+(np.sqrt(n_p)/n_p)**2) * probe_fluence

    return probe_fl_err



# plt.errorbar(energy, probe_fluence, yerr=probe_fl_err, fmt = 'C0.',
#               markersize = '4', capsize = 2, ecolor='C0',
#               elinewidth = 1, markeredgewidth = 1)


# Show the uncertainty on neutron fluence is negligible when number of incident protons is greater than 10000
xx = np.linspace(10000,1e6,100)
error = []
for i in range(len(xx)):
    c = normFluenceError(fluence_sd, probe_fluence_z1, xx[i], probe_z)
    error.append(100*np.mean(c[:-10]))
f = plt.figure(figsize=(6.5, 6.5/1.6))
plt.plot(xx, error,'-')
plt.xlabel("Number of histories")
plt.ylabel("Uncertainty in neutron fluence (%)")


