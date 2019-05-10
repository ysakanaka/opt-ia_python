import optIA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import  sys
import seaborn as sns


class Plot:

    def plot(self, xy, z, s):
        df = pd.DataFrame(xy, columns=['X', 'Y'])
        df['Z'] = z
        #print(type(df))
        #print(df)
        # Make the plot
        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis,
                        linewidth=0.2)
        plt.title(s)
        plt.show()


        # to Add a color bar which maps values to colors.
        surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis,
                               linewidth=0.2)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        # Rotate it
        ax.view_init(30, 45)
        plt.show()


        # Other palette
        ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
        plt.show()
        return
