import cdt
from cdt import SETTINGS
SETTINGS.verbose=True
#SETTINGS.NJOBS=16
#SETTINGS.GPU=1
import networkx as nx
import matplotlib.pyplot as plt
plt.axis('off')

# Load data
data = pd.read_csv("lucas0_train.csv")
print(data.head())

# Finding the structure of the graph
glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(data)

# Pairwise setting
model = cdt.causality.pairwise.ANM()
output_graph = model.predict(data, skeleton)

# Visualize causality graph
options = {
        "node_color": "#A0CBE2",
        "width": 1,
        "node_size":400,
        "edge_cmap": plt.cm.Blues,
        "with_labels": True,
    }
nx.draw_networkx(output_graph,**options)
