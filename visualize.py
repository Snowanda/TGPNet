import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_tree(nodes, title="Tree"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for node in nodes.values():
        x, y, z = node['pos']
        for child_id in node['children']:
            child = nodes[child_id]
            x2, y2, z2 = child['pos']
            ax.plot([x, x2], [y, y2], [z, z2], color='blue')
    ax.set_title(title)
    plt.show()