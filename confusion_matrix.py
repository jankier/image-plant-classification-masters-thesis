import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plot = sns.heatmap(
        cm, 
        annot = True, 
        square = True, 
        xticklabels = classes, 
        yticklabels = classes,
        fmt = 'd', 
        cmap = plt.cm.Blues,
        cbar = False,
    )
    
    plot.set_title(title + " - prediction results", fontsize=16)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right")
    plot.set_yticklabels(plot.get_yticklabels(), rotation=45, ha="right")
    plot.set_ylabel("True Labels", fontsize=12)
    plot.set_xlabel("Predicted Labels", fontsize=12)