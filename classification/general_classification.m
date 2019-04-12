function [] = general_classification(root_dir, subject_labels,...
    action_labels, tr_subjects, te_subjects, action_names) %#ok<INUSD>

    n_tr_te_splits = size(tr_subjects, 1);
    n_classes = length(unique(action_labels));   
   
    C_val = 1;
          
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
        [dim_2, dim_3] = size(features{1});
        
        feature_to_classify = zeros(length(features), dim_2, dim_3);
        for j=length(features)
            feature_to_classify(j, :, :) = features{j};
        end
        
        dim_1 = size(feature_to_classify, 1);
        feature_to_classify = reshape(feature_to_classify, dim_1, dim_2 * dim_3);
        
        features_train = cell(n_classes, 1);
        features_test = cell(n_classes, 1);

        for class = 1:n_classes
            
            features_train{class} = feature_to_classify(tr_subject_ind, :);
            features_test{class} = feature_to_classify(te_subject_ind, :);
        
        end

        
        [total_accuracy(i), cw_accuracy(i, :), confusion_matrices{i}] =...
            kernel_svm_one_vs_all_modified(features_train,...
            features_test, tr_labels, te_labels, C_val);

    end

    avg_total_accuracy = mean(total_accuracy); %#ok<NASGU>
    avg_cw_accuracy = mean(cw_accuracy); %#ok<NASGU>

    avg_confusion_matrix = zeros(size(confusion_matrices{1}));
    for i = 1:length(confusion_matrices)
        avg_confusion_matrix = avg_confusion_matrix + confusion_matrices{i};
    end
    avg_confusion_matrix = avg_confusion_matrix / length(confusion_matrices); %#ok<NASGU>
    
    results_dir = [root_dir, '/general_modeling_result'];
    results_saving(results_dir,...
        total_accuracy,...
        cw_accuracy,...
        avg_total_accuracy,...
        avg_cw_accuracy,...
        confusion_matrices,...
        avg_confusion_matrix,...
        action_names);

end

