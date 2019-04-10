function [] = naive_bayes_nearest_neighbor_classification(root_dir,...
    subject_labels, action_labels, tr_subjects, te_subjects, action_names )


    results_dir = [root_dir, '/nbnn_modeling_result'];
    mkdir(results_dir);

    n_tr_te_splits = size(tr_subjects, 1);
    n_classes = length(unique(action_labels));   
   
    % C_val = 1;
          
    loadname = 'features';        
  
    total_accuracy = zeros(n_tr_te_splits, 1);        
    cw_accuracy = zeros(n_tr_te_splits, n_classes);
    confusion_matrices = cell(n_tr_te_splits, 1);
        
    for i = 1:n_tr_te_splits         
        tr_subject_ind = ismember(subject_labels, tr_subjects(i,:));
        te_subject_ind = ismember(subject_labels, te_subjects(i,:));        
        tr_labels = action_labels(tr_subject_ind);
        te_labels = action_labels(te_subject_ind);
            
        data = load ([root_dir, '/', loadname], loadname);
        
        features = data.(loadname);
        
        features_train = features(tr_subject_ind);
        features_test = features(te_subject_ind);
    
        [total_accuracy(i), cw_accuracy(i, :), confusion_matrices{i}] =...
            naive_bayes_nearest_neighbor(features_train,...
            features_test, tr_labels, te_labels);

    end

    avg_total_accuracy = mean(total_accuracy);
    avg_cw_accuracy = mean(cw_accuracy);

    avg_confusion_matrix = zeros(size(confusion_matrices{1}));
    for i = 1:length(confusion_matrices)
        avg_confusion_matrix = avg_confusion_matrix + confusion_matrices{i};
    end
    avg_confusion_matrix = avg_confusion_matrix / length(confusion_matrices);

    save ([results_dir, '/classification_results.mat'],...
        'total_accuracy', 'cw_accuracy', 'avg_total_accuracy',...
        'avg_cw_accuracy', 'confusion_matrices', 'avg_confusion_matrix');
    
    % save confusion matrices as excel files.
    for i = 1:length(confusion_matrices)
        xlswrite([results_dir, '/confusion_matrices.xlsx'], confusion_matrices{i},...
            ['confusion_matrix', num2str(i)])
    end
    xlswrite([results_dir, '/confusion_matrices.xlsx'], action_names,...
            'text_labels')

    xlswrite([results_dir, '/avg_confusion_matrix.xlsx'], avg_confusion_matrix)
    xlswrite([results_dir, '/avg_confusion_matrix.xlsx'], action_names,...
            'text_labels')
        

end

