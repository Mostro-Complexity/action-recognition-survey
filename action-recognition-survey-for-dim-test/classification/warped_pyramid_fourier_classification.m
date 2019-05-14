function [avg_total_accuracy] = warped_pyramid_fourier_classification(root_dir, subject_labels,...
    action_labels, tr_subjects, te_subjects, action_names) %#ok<INUSD>

    n_tr_te_splits = size(tr_subjects, 1);
    n_classes = length(unique(action_labels));   
   
    C_val = 1;
          
    loadname = 'linear_kernel';        

    dir = [root_dir, '/dtw_warped_pyramid_lf_fourier_kernels'];
    loadfile_tag = 'warped_pyramid_lf_fourier_kernels';    
       
    total_accuracy = zeros(n_tr_te_splits, 1);        
    cw_accuracy = zeros(n_tr_te_splits, n_classes);
    confusion_matrices = cell(n_tr_te_splits, 1);
        
    i = 1;
    
    tr_subject_ind = ismember(subject_labels, tr_subjects(i,:));
    te_subject_ind = ismember(subject_labels, te_subjects(i,:));        
    tr_labels = action_labels(tr_subject_ind);
    te_labels = action_labels(te_subject_ind);

    K_train_train = cell(n_classes, 1);
    K_test_train = cell(n_classes, 1);
    
    for class = 1:n_classes

        data = load ([dir, '/', loadfile_tag, '_split_',...
            num2str(i), '_class_', num2str(class)], loadname);

        K = data.(loadname);

        K_train_train{class} = K(tr_subject_ind, tr_subject_ind);
        K_test_train{class} = K(te_subject_ind, tr_subject_ind);
    end

    [total_accuracy(i), cw_accuracy(i, :), confusion_matrices{i}] =...
        kernel_svm_one_vs_all_modified(K_train_train,...
        K_test_train, tr_labels, te_labels, C_val);


    avg_total_accuracy = mean(total_accuracy); 
    
end
