function [total_accuracy, class_wise_accuracy, confusion_matrix]...
    = fusion_svm_model(features_train, features_test, advanced_features_tr,...
    advanced_features_te, tr_labels, te_labels)

    n_tr_samples = length(features_train);
    n_te_samples = length(features_test);
    unique_classes = unique(tr_labels);
    n_classes = length(unique_classes);
    [n_dim, n_frames] = size(features_train{1});
    
    features_train = cell2mat(features_train');
    features_train = reshape(features_train, n_dim * n_frames, n_tr_samples);
    
    features_test = cell2mat(features_test');
    features_test = reshape(features_test, n_dim * n_frames, n_te_samples);
         
    low_model = fitcecoc(features_train' ,tr_labels, 'FitPosterior', 1);
    high_model = fitcecoc(advanced_features_tr ,tr_labels, 'FitPosterior', 1, 'Learners',t);
    
    [~, ~, ~, low_poster] = predict(low_model, features_test');
    [~, ~, ~, high_poster] = predict(high_model, advanced_features_te);
    
    poster = low_poster + high_poster;
    [~, predicted_labels] = max(poster, [], 2);
    predicted_labels = predicted_labels';
        
    % evaluation
    class_wise_accuracy = zeros(n_classes, 1);    
    confusion_matrix = zeros(n_classes, n_classes);    
    for i = 1:n_classes     % arranged by order
        temp = find(te_labels == unique_classes(i));
        class_wise_accuracy(i) =...
            length(find(predicted_labels(temp) == unique_classes(i))) / length(temp);
        
         confusion_matrix(i, :) = hist(predicted_labels(temp), unique_classes)...
             / length(temp);
    end
    
    total_accuracy = length(find(te_labels == predicted_labels))...
        / n_te_samples;

end

