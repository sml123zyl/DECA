import numpy as np
import matplotlib.pyplot as plt
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import parcorr, gpdc, cmiknn
import pandas as pd
# def dyn(x1,x2,x3,x4):
#         x1 = -0.287*x2 + np.random.normal()
#         x2 = 0.4*x2 + 0.287*x1+ np.random.normal()
#         x3 = 0.9*x3 + np.random.normal()
#         x4 = 0.9 * x2 + np.random.normal()
#         return x1,x2,x3,x4
#
#
# x1_ini,x2_ini,x3_ini,x4_ini = np.random.rand(4,1)  #初态
# timestep = 200 #时间步长
# data = np.expand_dims(np.concatenate((x1_ini,x2_ini,x3_ini,x4_ini ), axis=0),axis=0) #shape = [1,4]
#
#
# for step in range(timestep): # run dynamics
#         x1_ini,x2_ini,x3_ini,x4_ini = dyn(x1_ini,x2_ini,x3_ini,x4_ini)
#         temp = np.expand_dims(np.concatenate((x1_ini,x2_ini,x3_ini,x4_ini), axis=0),axis=0)
#         data = np.concatenate((data, temp), axis=0)
# print(data.shape) #shape = [timestep+1,4]
data_name='house sales'
data_df = pd.read_csv('./{}.csv'.format(data_name))
data_df = data_df.iloc[:, 1:]
data = np.array(data_df)
# var_names = data_df.columns.tolist()
# var_names = ['x1', 'x2', 'x3', 'x4','x5', 'x6', 'x7', 'x8', 'x9']
num_columns = data_df.shape[1]
# 自动生成变量名称
var_names = [f'x{i+1}' for i in range(num_columns)]

dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
# print(dataframe)

tp.plot_timeseries(dataframe,'DRAW',label_fontsize=20,tick_label_size=10)
# correlations = PCMCI.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
# lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations,name='draw1',setup_args={'var_names':var_names,'x_base':5, 'y_base':.5})
ParCorr = parcorr.ParCorr(significance='analytic')
pcmci = PCMCI(
dataframe=dataframe,
cond_ind_test=ParCorr,
verbosity=1)

pcmci.verbosity = 1
tau_max=2
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')

# tp.plot_graph(graph = results['graph'],
#         val_matrix=results['val_matrix'],
#         var_names=var_names,
#         save_name='draw2',
#
# )
# tp.plot_graph(graph = results['graph'],
#         figsize=(9, 9),
#         val_matrix=results['val_matrix'],
#         var_names=var_names,
#         save_name='draw2',
#         link_colorbar_label='MCI',
#         node_colorbar_label='auto-MCI',
#         link_label_fontsize=8,
#         label_fontsize=5,
#         tick_label_size=14,
#         node_label_size=14,
#         edge_ticks=0.5,
#         node_ticks=0.2,
#         node_size=0.2
# )
# tp.plot_time_series_graph(
#         figsize=(9, 9),
#         val_matrix=results['val_matrix'],
#         graph = results['graph'],
#         var_names=var_names,
#         link_colorbar_label='MCI',
#         label_fontsize=20,
#         tick_label_size=14,
#         save_name='draw3'
# )
val_matrix = results['val_matrix']
p_matrix = results['p_matrix']
# 创建一个28x28的有向邻接矩阵
adjacency_matrix = np.random.randint(2, size=(num_columns, num_columns), dtype=int)

# 打印有向邻接矩阵
print(adjacency_matrix)

# adjacency_matrix = np.array([
#     [0, 1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0]
# ])

result_array = val_matrix[:, :, 0].T
result_array1 =p_matrix[:, :, 0].T
result_array =  np.array(result_array)
result_array1 =  np.array(result_array1)
print(result_array)
np.save('./{}.npy'.format(data_name), [result_array, result_array1,adjacency_matrix])
# Create a figure and axis
fig, ax = plt.subplots()

# Plot the matrix as a heatmap
cax = ax.matshow(result_array, cmap='viridis')

# Add text annotations to each cell
for i in range(result_array.shape[0]):
    for j in range(result_array.shape[1]):
        ax.text(j, i, f'{result_array[i, j]:.2f}', ha='center', va='center', color='w' if result_array[i, j] < 0.5 else 'black')

# Add colorbar
cbar = plt.colorbar(cax)

# Set axis labels
ax.set_xticks(np.arange(result_array.shape[1]))
ax.set_yticks(np.arange(result_array.shape[0]))
# ax.set_xticklabels(['Col1', 'Col2', 'Col3'])
# ax.set_yticklabels(['Row1', 'Row2', 'Row3'])
plt.savefig('./{}.png'.format(data_name))
# Show the plot
plt.show()