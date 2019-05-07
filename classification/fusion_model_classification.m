function [] = fusion_model_classification(root_dir, subject_labels,...
    action_labels, tr_subjects, te_subjects, action_names)
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
            
        load([root_dir, sprintf('/advanced_split_%d',i)], 'advanced_features')
        data = load ([root_dir, '/', loadname], loadname);
        
        features = data.(loadname);
        
        features_train = features(tr_subject_ind);
        features_test = features(te_subject_ind);
        
        advanced_features_tr = advanced_features(tr_subject_ind, :);
        advanced_featreus_te = advanced_features(te_subject_ind, :);
 
        % classifier
        [total_accuracy(i), cw_accuracy(i, :), confusion_matrices{i}] =...
            fusion_svm_model(features_train, features_test,...
            advanced_features_tr, advanced_featreus_te, tr_labels, te_labels);

    end

    avg_total_accuracy = mean(total_accuracy); 
    avg_cw_accuracy = mean(cw_accuracy); 

    avg_confusion_matrix = zeros(size(confusion_matrices{1}));
    for i = 1:length(confusion_matrices)
        avg_confusion_matrix = avg_confusion_matrix + confusion_matrices{i};
    end
    avg_confusion_matrix = avg_confusion_matrix / length(confusion_matrices); 
    
    results_dir = [root_dir, '/fusion_model_modeling_result'];
    results_saving(results_dir,...
        total_accuracy,...
        cw_accuracy,...
        avg_total_accuracy,...
        avg_cw_accuracy,...
        confusion_matrices,...
        avg_confusion_matrix,...
        action_names);
    
end

