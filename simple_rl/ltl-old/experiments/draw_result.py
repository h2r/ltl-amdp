import os
import matplotlib.pyplot as plt
import numpy as np

""" Read a result file - env 1"""
contents_set={}
file = open("{}/results/result_time_random.txt".format(os.getcwd()), "r")
contents_set[2] = file.readlines()
file.close()

'''
file = open("{}/results/result_time_random_env1.txt".format(os.getcwd()), "r")
contents_set[1] = file.readlines()
file.close()
'''
analysis = {}
Ndata = 100
#for num_env in [1,2]:

for num_env in [2]:
    contents = contents_set[num_env]
    var_names = contents[2].split(', ')

    # initialize lists (extract name of variables)
    clist={}
    for jj in range(0, 13):
        clist[var_names[jj]] = []
    clist['ap_maps']=[]

    # Read data
    print('Length contents: ', len(contents))
    for ii in range(3, len(contents)):
        print(ii)
        print(contents[ii])
        cur_list = contents[ii].split(', ')
        if not(cur_list[6]=='-1' and cur_list[7]=='-1' and cur_list[8]=='-1'):
            for jj in range(0, 12):
                clist[var_names[jj]].append(float(cur_list[jj]))
            clist['LTL'].append(cur_list[12])
            clist['ap_maps'].append(eval(', '.join(cur_list[13:])))

            if len(clist['time_AMDP']) == Ndata:
                break

    n = len(clist['time_AMDP'])
    print(np.sum(clist['correct_AMDP']), np.sum(clist['correct_pMDP']))
    # analysis
    ratio_time = np.divide(clist['time_AMDP'], clist['time_pMDP'])
    ratio_backup = np.divide(clist['backup_AMDP'], clist['backup_pMDP'])
    min_level = []
    for ii in range(0, n):
        levels = [clist['ap_maps'][ii][ap][0] for ap in clist['ap_maps'][ii].keys()]
        min_level.append(min(levels))
    # histogram-ratio
    bin_edges_ratio = [ii*0.1 for ii in range(0, 18)]
    hist_time, bin_edges = np.histogram(ratio_time, bin_edges_ratio)
    hist_backup, _ = np.histogram(ratio_backup, bin_edges_ratio)

    # histogram -time
    nbin = np.linspace(0, max(max(clist['time_AMDP']), max(clist['time_pMDP'])), 10)
    hist_time_AMDP, bin_edges_time_AMDP = np.histogram(clist['time_AMDP'], nbin)
    hist_time_pMDP, bin_edges_time_pMDP = np.histogram(clist['time_pMDP'], bin_edges_time_AMDP)


    # histogram -backup
    nbin = np.linspace(0, max(max(clist['backup_AMDP']), max(clist['backup_pMDP'])), 10)
    hist_backup_AMDP, bin_edges_backup_AMDP = np.histogram(clist['backup_AMDP'], nbin)
    hist_backup_pMDP, bin_edges_backup_pMDP = np.histogram(clist['backup_pMDP'], nbin)

    analysis[num_env] = {'hist_time_ratio': hist_time, 'hist_backup_ratio': hist_backup, 'bin_edges_ratio': bin_edges_ratio,
                         'hist_time_AMDP': hist_time_AMDP, 'hist_time_pMDP': hist_time_pMDP,
                         'bin_edges_time_AMDP': bin_edges_time_AMDP, 'bin_edges_time_pMDP': bin_edges_time_pMDP,
                         'hist_backup_AMDP': hist_backup_AMDP, 'hist_backup_pMDP': hist_backup_pMDP,
                         'bin_edges_backup_AMDP': bin_edges_backup_AMDP, 'bin_edges_backup_pMDP': bin_edges_backup_pMDP}


# time histogram
ftsize = 13
f1 = plt.figure()
plt.plot(bin_edges[1:], np.cumsum(analysis[1]['hist_time_ratio'])/np.sum(analysis[1]['hist_time_ratio']), marker='o')
plt.plot(bin_edges[1:], np.cumsum(analysis[2]['hist_time_ratio'])/np.sum(analysis[2]['hist_time_ratio']), marker='^')
plt.grid(True)
plt.legend(['Small world', 'Large world'], fontsize=ftsize)
plt.xlabel('Ratio: time (AL-MDP) / time (P-MDP)', fontsize=ftsize)
plt.ylabel('Cumulative histogram', fontsize=ftsize)
plt.savefig("{}/results/ratio_time.png".format(os.getcwd()))
plt.show(block=False)

f2 = plt.figure()
plt.plot(bin_edges[1:], np.cumsum(analysis[1]['hist_backup_ratio'])/np.sum(analysis[1]['hist_backup_ratio']), marker='o')
plt.plot(bin_edges[1:], np.cumsum(analysis[2]['hist_backup_ratio'])/np.sum(analysis[2]['hist_backup_ratio']), marker='^')
plt.legend(['Small world', 'Large world'])
plt.xlabel('Ratio: backup (AL-MDP) / backup (P-MDP)')
plt.ylabel('Cumulative histogram')
plt.grid(True)
plt.savefig("{}/results/ratio_backup.png".format(os.getcwd()))
plt.show(block=False)

#
f3 = plt.figure()
_, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(analysis[1]['bin_edges_time_AMDP'][1:], np.cumsum(analysis[1]['hist_time_AMDP']), marker='o', color=color)
ax1.plot(analysis[1]['bin_edges_time_pMDP'][1:], np.cumsum(analysis[1]['hist_time_pMDP']), linestyle='--', marker='^', color=color)
ax1.set_xlabel('time (s)', fontsize=ftsize, color=color)
ax1.set_ylabel('Cumulative number of cases', fontsize=ftsize)
ax1.tick_params(labelsize='large')
ax1.legend(['time (AL-MDP)', 'time (pMDP)'], fontsize=ftsize-1, loc=(0.55, 0.27))

ax2 = ax1.twiny()
color = 'tab:blue'
ax2.plot(analysis[1]['bin_edges_backup_AMDP'][1:], np.cumsum(analysis[1]['hist_backup_AMDP']), marker='o', color=color)
ax2.plot(analysis[1]['bin_edges_backup_pMDP'][1:], np.cumsum(analysis[1]['hist_backup_pMDP']), linestyle='--', marker='^', color=color)
ax2.set_xlabel('Backup', fontsize=ftsize, color=color)
ax2.tick_params(labelsize='large')
ax2.legend(['backup (AL-MDP)', 'backup (pMDP)'], fontsize=ftsize-1, loc=(0.55, 0.1))


plt.grid(True, alpha=0.5)
plt.savefig("{}/results/time_hist_small.png".format(os.getcwd()))
plt.show(block=False)

#
f4 = plt.figure()
_, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(analysis[2]['bin_edges_time_AMDP'][1:], np.cumsum(analysis[2]['hist_time_AMDP']), marker='o', color=color)
ax1.plot(analysis[2]['bin_edges_time_pMDP'][1:], np.cumsum(analysis[2]['hist_time_pMDP']), linestyle='--', marker='^', color=color)
ax1.set_xlabel('time (s)', fontsize=ftsize, color=color)
ax1.set_ylabel('Cumulative number of cases', fontsize=ftsize)
ax1.tick_params(labelsize='large')
ax1.legend(['time (AL-MDP)', 'time (pMDP)'], fontsize=ftsize-1, loc=(0.55, 0.27))

ax2 = ax1.twiny()
color = 'tab:blue'
ax2.plot(analysis[2]['bin_edges_backup_AMDP'][1:], np.cumsum(analysis[2]['hist_backup_AMDP']), marker='o', color=color)
ax2.plot(analysis[2]['bin_edges_backup_pMDP'][1:], np.cumsum(analysis[2]['hist_backup_pMDP']), linestyle='--', marker='^', color=color)
ax2.set_xlabel('Backup', fontsize=ftsize, color=color)
ax2.tick_params(labelsize='large')
ax2.legend(['backup (AL-MDP)', 'backup (pMDP)'], fontsize=ftsize-1, loc=(0.55, 0.1))


plt.grid(True, alpha=0.5)

plt.savefig("{}/results/time_hist_big.png".format(os.getcwd()))
plt.show(block=False)


print("end")
