#==========================================
# Plotting functions
#==========================================

from helper_functions import *

def plot_model_train_info (train_losses, val_losses, save_path=None):
    font_size = 25
    plt.figure(1)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.plot(train_losses)
    plt.plot(val_losses)
    #plt.title('model loss')
    plt.ylabel('Loss', fontsize=font_size)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.legend(['Train', 'Val'], fontsize=22)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    

    return True

def plot_y_yhat (y_train,yhat_train,y_test,yhat_test, save_path):
    plt.figure(3, figsize=(35, 5))
    plt.plot(yhat_train, label='prediction')
    plt.plot(y_train, label='ground truth')
    plt.plot(y_train-yhat_train, label='error (truth-pred)')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Train samples')
    plt.ylabel('UL delay (ms)')
    plt.xlabel('Samples')
    plt.ylim(-100, 100)
    plt.legend()
    if save_path:
        plt.savefig(save_path + '_train'+'.pdf')    
    plt.show()
  
    plt.figure(4, figsize=(35, 5))
    plt.plot(yhat_test, label='prediction')
    plt.plot(y_test, label='ground truth')
    plt.plot(y_test-yhat_test, label='error (truth-pred)')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Test samples')
    plt.ylabel('UL delay (ms)')
    plt.xlabel('Samples')
    plt.ylim(-100, 100)
    plt.legend()
    if save_path:
        plt.savefig(save_path + '_test'+'.pdf')
    plt.show()
    
    return True

def plot_heatmap(windowed_combined_data, plot_name):
    plt.rcParams["figure.autolayout"] = True
    corr_data = windowed_combined_data.corr(method='spearman')
    fig = plt.figure(figsize=(11,11))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    mask = mask[1:, :-1]
    corr = corr_data.iloc[1:,:-1].copy()
    hmap = sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True, mask=mask, 
                       vmax=1, annot_kws={"fontsize":15}, linecolor='black')
    fig = hmap.get_figure()
    plt.yticks(rotation=45)
    fig.autofmt_xdate(rotation=45)
    #plt.savefig(plot_name)
    fig.show()

def tabnet_explain(model, X, feat_filter, X_feats):
    # Global explainability : feat importance summing to 1
    print('Global explainability')
    importances = model.feature_importances_
    plot_feature_importance(importances, X_feats, feat_filter)
    
    # Local explainability and masks
    print('Local explainability')
    explain_matrix, masks = model.explain(X)
    print('Explainability Matrix')
    print(explain_matrix)
    fig, axs = plt.subplots(1, 3, figsize=(20,20))
    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
    return

def xgb_explain(model, feat_filter, X_feats):
    # The length of importances reflects the number of features used
    importances = model.feature_importances_
    plot_feature_importance(importances, X_feats, feat_filter)
    return

def plot_feature_importance(importances, X_feats, feat_filter):
    # increasing order in value and hence decreasing order in importance 
    # sort the importances and then fetch the index value of those importances 
    indices = np.argsort(importances)
    #This is in ascending order of 
    bar_vals = importances[np.flip(indices)[0:feat_filter]]
    bar_names = X_feats[np.flip(indices)[0:feat_filter]]
    #print(importances[np.flip(indices)[0:feat_filter]])
    #print(X_feats[np.flip(indices)[0:feat_filter]])

    #top_n_features = list( set(top_n_features).union(set(bar_names)))
    #print('Top n feature list: ', top_n_features)
    plt.figure(figsize=(7, 4))
    plt.barh(range(len(bar_vals)), np.flip(bar_vals), color='b', align='center')
    plt.yticks(range(len(bar_vals)), np.flip(bar_names))

    plt.title('Feature importance')
    plt.xlabel('Relative Importance')
    #plt.savefig('plots_for_paper/feat_imp'+'.pdf', bbox_inches='tight')
    plt.show() 
    return

def draw_confusion_matrix(true, pred, title):                
    cm = confusion_matrix(true, pred, normalize='true')
    # Create a confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=np.unique(true), yticklabels=np.unique(pred))
    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    # Show the plot
    plt.show()

# Function to set up each subplot for the Q-Q plots
def setup_axes(ax, x, y, title, color, bounds):
    ax.plot(x, y, color, marker='.', linestyle='none')
    #ax.set_title(title, fontsize=16)
    ax.set_xlabel('Ground truth', fontsize=20)
    ax.set_ylabel('Predictions', fontsize=20)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.plot(bounds, bounds, 'k-')  # Diagonal line
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    # Set ticks at fixed intervals (modify interval as needed)
    tick_locator = MaxNLocator(nbins=5)  # Adjust number of bins as needed
    ax.xaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_locator(tick_locator)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_xticklabels(ax.get_xticks().round(3), fontsize=20)
    ax.set_yticklabels(ax.get_yticks().round(3), fontsize=20)
    return

# Plot hist of output
def plot_hist_of_y(y_train, y_test, learning_task):
    # If classification it will just bin it
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
    fig.subplots_adjust(top=0.85, bottom=0.20)
    fig.supxlabel(learning_task)
    fig.suptitle('Histogram')
    ax1.hist(y_train, bins=50, color='r', edgecolor='k', label='train samples')
    #ax1.set_xlabel(learning_task)
    ax1.set_yticks([])
    ax1.set_title('Train')
    ax2.hist(y_test, bins=50, color='b', edgecolor='k', label='test_samples')
    
    ax2.set_yticks([])
    ax2.set_title('Test')
    plt.show()
