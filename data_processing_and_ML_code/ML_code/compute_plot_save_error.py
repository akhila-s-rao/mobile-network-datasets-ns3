from helper_functions import *

def compute_plot_save_error(y_train, yhat_train, y_test, yhat_test, learning_task, learning_task_type, rs, train_results, test_results, plot_save_path):
    
    if learning_task_type == 'reg':
        if np.isnan(yhat_train).any():
            print('WARNING: nans in the predicted train vals')
            print(yhat_train)
        train_results.loc[learning_task, 'MAE'][rs] = compute_error(y_train, yhat_train, 'mae')
        train_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_train, yhat_train, 'mape')
        train_results.loc[learning_task, 'R2'][rs] = compute_error(y_train, yhat_train, 'r2')
        # Compute and Print Test set errors
        if np.isnan(yhat_test).any():
            print('WARNING: NaNs in the predicted test vals')
            print(yhat_test)
        test_results.loc[learning_task, 'MAE'][rs] = compute_error(y_test, yhat_test, 'mae')
        test_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_test, yhat_test, 'mape')
        test_results.loc[learning_task, 'R2'][rs] = compute_error(y_test, yhat_test, 'r2')

        # Predicted versus ground-truth plots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        #fig.suptitle(learning_task, fontsize=16)
        y = np.concatenate((y_train, y_test))
        yhat = np.concatenate((yhat_train, yhat_test))
        bounds=[min( min(y),min(yhat) ), max( max(y),max(yhat) )]        
        # Configure each subplot
        setup_axes(axes[0], y_train, yhat_train, 'Train', COLOUR_HEX[0], bounds)  # Adjust color as needed
        setup_axes(axes[1], y_test, yhat_test, 'Test', COLOUR_HEX[1], bounds)  # Adjust color as needed
        #plt.tight_layout()
        if plot_save_path:
            plt.savefig(plot_save_path) 
        plt.show()
    else: # clas
        if np.isnan(yhat_train).any():
            print('WARNING: nans in the predicted train vals')
            print(yhat_train)                
        
        train_results.loc[learning_task, 'MAE'][rs] = compute_error(y_train, yhat_train, 'acc')
        train_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_train, yhat_train, 'f1score')
        train_results.loc[learning_task, 'R2'][rs] = compute_error(y_train, yhat_train_proba, 'roc_auc')
    
        # Compute and Print Test set errors
        if np.isnan(yhat_test).any():
            print('WARNING: NaNs in the predicted test vals')
            print(yhat_test)
        test_results.loc[learning_task, 'MAE'][rs] = compute_error(y_test, yhat_test, 'acc')
        test_results.loc[learning_task, 'MAPE'][rs] = compute_error(y_test, yhat_test, 'f1score')
        test_results.loc[learning_task, 'R2'][rs] = compute_error(y_test, yhat_test_proba, 'roc_auc')
        
        # Confusion matrix plots
        draw_confusion_matrix(y_train, yhat_train, 'Train: Confusion Matrix')
        draw_confusion_matrix(y_test, yhat_test, 'Test: Confusion Matrix')
   
    print('')
    #print(test_results)
    
    return train_results, test_results