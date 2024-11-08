# plot the cdf of the train error 
print(yhat_train.shape) # ,1
print(y_train.shape) # ,
yhat_train_a = np.squeeze(yhat_train)
yhat_test_a = np.squeeze(yhat_test)
ecdf_train = ECDF(yhat_train_a - y_train)
plt.step(ecdf_train.x, ecdf_train.y)
plt.axvline(x=0, color='red', linestyle='--')
plt.axhline(y=ecdf_train(0), color='red', linestyle='--')
plt.xlabel('Pred err (truth - pred)')
plt.title('Train samples')
#plt.hist((yhat_train - y_train), bins=200, edgecolor='k')
#plt.xlim(-20, 50)
plt.show()
print('Train: Probability mass of pred err (truth-pred) below 0 is: ',  ecdf_train(0))
print('Train: Probability mass of pred err (truth-pred) above 0 is: ',  1-ecdf_train(0))

# plot the cdf of the test error 
ecdf_test = ECDF(yhat_test_a - y_test)
plt.step(ecdf_test.x, ecdf_test.y)
plt.axvline(x=0, color='red', linestyle='--')
plt.axhline(y=ecdf_test(0), color='red', linestyle='--')
plt.xlabel('Pred err (truth - pred)')
plt.title('Test samples')
#plt.hist((yhat_test - y_test), bins=200, edgecolor='k')
#plt.xlim(-20, 50)
plt.show()
print('Test: Probability mass of pred err (truth-pred) below 0 is: ',  ecdf_test(0))
print('Test: Probability mass of pred err (truth-pred) above 0 is: ',  1-ecdf_test(0))

#===================================== plot sorted samples of prediction overlayed with ground truth  ==========================
# 
#if sort_test_samples:   
#    #train_baseline_vals = np.repeat(baseline_pred, len(y_train))
#    #test_baseline_vals = np.repeat(baseline_pred, len(y_test))
#
#    tmp1 = np.append(np.expand_dims(y_train, axis=1), np.expand_dims(yhat_train, axis=1), axis=1)
#    tmp1 = tmp1[tmp1[:, 0].argsort()]
#    y_train = tmp1[:,0]
#    yhat_train = tmp1[:,1]
#
#    tmp2 = np.append(np.expand_dims(y_test, axis=1), np.expand_dims(yhat_test, axis=1), axis=1)
#    tmp2 = tmp2[tmp2[:, 0].argsort()]
#    y_test = tmp2[:,0]
#    yhat_test = tmp2[:,1]

##=============================================== bin the delay values to observe err per bin ==================================
##
## bin index for each delay value, so that we can put the values in the right bin 
#bin_indices = np.digitize(y_train, bin_edges)
#
## I want to take all the delay values for each bin
#for bin_ind in np.unique(bin_indices):
#    # these are the delay values in bin bin_edges[bin_ind]
#    train_bin_uldelay_mean[bin_ind-1] = train_bin_uldelay_mean[bin_ind-1] + np.sum(y_train[bin_indices == bin_ind])
#    train_bin_count[bin_ind-1] = train_bin_count[bin_ind-1] + len(y_train[bin_indices == bin_ind])
#    
#    # I want the corresponding err values for these delay values  
#    train_bin_err_mean[bin_ind-1] = (train_bin_err_mean[bin_ind-1] + 
#                                     np.sum(np.abs(y_train[bin_indices == bin_ind] - yhat_train[bin_indices == bin_ind]) ))
#    train_bin_baseline_err_mean[bin_ind-1] = (train_bin_baseline_err_mean[bin_ind-1] + 
#                                             np.sum(np.abs(y_train[bin_indices == bin_ind] - train_baseline_vals[bin_indices == bin_ind]) ))
#    train_bin_perc_err_mean[bin_ind-1] = (train_bin_perc_err_mean[bin_ind-1] + 
#                                          np.sum(np.abs((y_train[bin_indices == bin_ind] - yhat_train[bin_indices == bin_ind]))/(y_train[bin_indices == bin_ind]) ))
#    train_bin_baseline_perc_err_mean[bin_ind-1] = (train_bin_baseline_perc_err_mean[bin_ind-1] + 
#                                                  np.sum(np.abs((y_train[bin_indices == bin_ind] - train_baseline_vals[bin_indices == bin_ind]))/(y_train[bin_indices == bin_ind])) )
#
## bin index for each delay value
#bin_indices = np.digitize(y_test, bin_edges)
#
## I want to take all the delay values for each bin 
#for bin_ind in np.unique(bin_indices):
#    # these are the delay values in bin bin_edges[bin_ind]
#    test_bin_uldelay_mean[bin_ind-1] = test_bin_uldelay_mean[bin_ind-1] + np.sum(y_test[bin_indices == bin_ind])
#    test_bin_count[bin_ind-1] = test_bin_count[bin_ind-1] + len(y_test[bin_indices == bin_ind])
#    
#    # I want the corresponding err values for these delay values
#    test_bin_err_mean[bin_ind-1] = (test_bin_err_mean[bin_ind-1] + 
#                                    np.sum(np.abs(y_test[bin_indices == bin_ind] - yhat_test[bin_indices == bin_ind])) )
#    test_bin_baseline_err_mean[bin_ind-1] = (test_bin_baseline_err_mean[bin_ind-1] + 
#                                             np.sum(np.abs(y_test[bin_indices == bin_ind] - test_baseline_vals[bin_indices == bin_ind])))
#    test_bin_perc_err_mean[bin_ind-1] = ( test_bin_perc_err_mean[bin_ind-1] + 
#                                         np.sum(np.abs((y_test[bin_indices == bin_ind] - yhat_test[bin_indices == bin_ind]))/(y_test[bin_indices == bin_ind])) )
#    test_bin_baseline_perc_err_mean[bin_ind-1] = (test_bin_baseline_perc_err_mean[bin_ind-1] + 
#                                                  np.sum(np.abs((y_test[bin_indices == bin_ind] - test_baseline_vals[bin_indices == bin_ind]))/(y_test[bin_indices == bin_ind])) ) 
#
## Plot
#plot_y_yhat(y_train, yhat_train, y_test, yhat_test, models_folder)
#
## Convert the regression output to class labels and do confusion matrix
#cf_matrix = confusion_matrix(value_to_class_label(y_test, delay_class_edges), 
#                             value_to_class_label(yhat_test, delay_class_edges), normalize='true')
#sns.set(rc={'figure.figsize':(8,7)}, font_scale = 1.5)
#sns.heatmap(cf_matrix, annot=True, 
#    fmt='.1%', cmap='Blues')
#
#
#=============================================== plot q-q prediction versus ground truth  ==================================


# Binning by Y vale to see if the prediction error in each bin is consistent or not 

print('-----------------------------------------------------------')
print('Top n feature list size: ', len(top_n_features))
print(top_n_features)
print('-----------------------------------------------------------')

# After going over all runs     
fig, ax1 = plt.subplots(figsize=(16, 4))
ax2 = ax1.twinx()
ax1.plot(train_bin_uldelay_mean/train_bin_count, train_bin_err_mean/train_bin_count, 'r.-', label='XGB pred err (truth-pred)')
#ax1.plot(train_bin_uldelay_mean/train_bin_count, train_bin_baseline_err_mean/train_bin_count, 'm.-', label='baseline pred err (truth-pred)')
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_ylabel('error')
#ax1.set_ylim(-5,25)
ax1.legend()
plt.xlabel('uplink delay (ms)')
plt.title('Train samples')
plt.grid()
ax2.set_ylabel('relative err')
ax2.plot(train_bin_uldelay_mean/train_bin_count, train_bin_perc_err_mean/train_bin_count, 'g*-', label='XGB relative err')
#ax2.plot(train_bin_uldelay_mean/train_bin_count, train_bin_baseline_perc_err_mean/train_bin_count, 'c*-', label='baseline relative err')
ax2.axhline(y=0, color='g', linestyle='--')
plt.xscale('log')
ax1.legend(loc=6)
ax2.legend(loc=1)
plt.show() 

plt.figure(figsize=(16, 2))
plt.plot(train_bin_uldelay_mean/train_bin_count, train_bin_count, 'b*-')
plt.xlabel('Train samples uplink delay bin (ms)')
plt.xscale('log')
plt.ylabel('bin count')
plt.grid()
plt.show()

fig, ax1 = plt.subplots(figsize=(16, 4))
ax2 = ax1.twinx()
ax1.plot(test_bin_uldelay_mean/test_bin_count, test_bin_err_mean/test_bin_count, 'r.-', label='XGB pred err (truth-pred)')
#ax1.plot(test_bin_uldelay_mean/test_bin_count, test_bin_baseline_err_mean/test_bin_count, 'm.-', label='baseline pred err (truth-pred)')
ax1.axhline(y=0, color='r', linestyle='--')
plt.xlabel('uplink delay (ms)')
ax1.set_ylabel('error')
#ax1.set_ylim(-500,250)
ax1.legend()
plt.title('Test samples')
plt.grid()
ax2.set_ylabel('relative err')
ax2.plot(test_bin_uldelay_mean/test_bin_count, test_bin_perc_err_mean/test_bin_count, 'g*-', label='relative err')
#ax2.plot(test_bin_uldelay_mean/test_bin_count, test_bin_baseline_perc_err_mean/test_bin_count, 'c*-', label='baseline relative err')
ax2.axhline(y=0, color='g', linestyle='--')
plt.xscale('log')
ax1.legend(loc=6)
ax2.legend(loc=1)
plt.show()

plt.figure(figsize=(16, 2))
plt.plot(test_bin_uldelay_mean/test_bin_count, test_bin_count, 'b*-')
plt.xlabel('Test samples uplink delay bins (ms)')
plt.ylabel('bin count')
plt.xscale('log')
plt.grid()
plt.show()

#train_results.loc[learning_task, 'model MAE'][rs] = compute_error(y_train, yhat_train, 'mae')
#train_results.loc[learning_task, 'baseline1 MAE'][rs] = compute_error(y_train, np.full(y_train.shape, np.mean(y_train)), 'mae')
#train_results.loc[learning_task, 'baseline2 MAE'][rs] = compute_error(y_train, np.full(y_train.shape, np.median(y_train)), 'mae')
#train_results.loc[learning_task, 'model MAPE'][rs] = compute_error(y_train, yhat_train, 'mape')
#train_results.loc[learning_task, 'baseline1 MAPE'][rs] = compute_error(y_train, np.full(y_train.shape, np.mean(y_train)), 'mape')
#train_results.loc[learning_task, 'baseline2 MAPE'][rs] = compute_error(y_train, np.full(y_train.shape, np.median(y_train)), 'mape')

# Compute and Print Test set errors
#test_results.loc[learning_task, 'model MAE'][rs] = compute_error(y_test, yhat_test, 'mae')
#test_results.loc[learning_task, 'baseline1 MAE'][rs] = compute_error(y_test, np.full(y_test.shape, np.mean(y_train)), 'mae')
#test_results.loc[learning_task, 'baseline2 MAE'][rs] = compute_error(y_test, np.full(y_test.shape, np.median(y_train)), 'mae')
#test_results.loc[learning_task, 'model MAPE'][rs] = compute_error(y_test, yhat_test, 'mape')
#test_results.loc[learning_task, 'baseline1 MAPE'][rs] = compute_error(y_test, np.full(y_test.shape, np.mean(y_train)), 'mape')
#test_results.loc[learning_task, 'baseline2 MAPE'][rs] = compute_error(y_test, np.full(y_test.shape, np.median(y_train)), 'mape')

#train_results.loc[learning_task, 'model MAE'][rs] = compute_error(y_train, yhat_train, 'acc')
#train_results.loc[learning_task, 'model MAPE'][rs] = compute_error(y_train, yhat_train, 'f1score')
# Compute and Print Test set errors
#test_results.loc[learning_task, 'model MAE'][rs] = compute_error(y_test, yhat_test, 'acc')
#test_results.loc[learning_task, 'model MAPE'][rs] = compute_error(y_test, yhat_test, 'f1score')

# Baseline 1 is the mean of the train set and Baseline 2 is the median of the train set 





#from torch.multiprocessing import Process
## Run this parallelly for all the models
#if __name__ == '__main__':
#    model_types = ['dae']#['dae', 'vime', 'scarf', 'subtab']
#    processes = []
#    print(torch.cuda.is_available())
#    print(torch.version.cuda) 
#    for model_type in model_types:
#        p = Process(target=second_phase_training, args=(model_type,))
#        p.start()
#        processes.append(p)

#    for p in processes:
#        p.join()


    
#from torch.multiprocessing import Pool
#if __name__ == '__main__':
#    pool = Pool(processes=4)
#    inputs = [(type,) for type in ['dae', 'vime']]
#    #inputs = [(type,) for type in ['dae']]
#    outputs = pool.starmap(second_phase_training, inputs)  
#    #outputs = pool.map(setup_run, range(0,runs))
#    print(outputs)
