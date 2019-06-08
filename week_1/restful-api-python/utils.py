from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(model, X_columns, y_true, x_size=12, y_size=12):
    """Returns a ROC plot
    """
    y_pred = model.predict_proba(X_columns)

    fpr, tpr, threshold = roc_curve(y_true, y_pred[:,1])
    area = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(x_size, y_size))
    model_name = str(type(model)).split('.')[-1].strip(">\'")
    plt.title(f'{model_name} ROC')
    ax.plot(fpr, tpr, 'k', label='AUC = %0.3f' % area)

    ax.legend(loc='lower right')
    ax.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()