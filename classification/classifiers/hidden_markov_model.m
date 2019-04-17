function [total_accuracy, class_wise_accuracy, confusion_matrix]...
    = hidden_markov_model(features_train, features_test, tr_labels, te_labels, ~)

    % init
    unique_classes = unique(tr_labels);
    n_classes = length(unique_classes);
    
    n_tr_samples = length(features_train);
    n_te_samples = length(features_test);
    
    n_states = n_classes;
    n_clusters = 15;
    
    A_of_models = cell(n_classes, 1);
    B_of_models = cell(n_classes, 1);
        
    [features_train, features_test] = clustering(features_train,...
        features_test, n_clusters);

    
    % training
    for c=1:n_classes        
        features_in_c = features_train(tr_labels == unique_classes(c));
        n_features_in_c = length(features_in_c);
        trans_in_c = zeros(n_states, n_states, n_features_in_c);
        emis_in_c = zeros(n_states, n_clusters, n_features_in_c);
        
        for i=1:n_features_in_c
            trans = rand(n_states);
            trans = trans./sum(trans, 1);

            emis = rand(n_states, n_clusters); %500 clusters
            emis = emis./sum(emis, 1);

            [trans_in_c(:, :, i), emis_in_c(:, :, i)] = hmmtrain(...
                features_in_c{i}', trans, emis, 'Maxiterations', 30);
        end
        
        A_of_models{c} = sum(trans_in_c, 3)./n_features_in_c;
        B_of_models{c} = sum(emis_in_c, 3)./n_features_in_c;
    end
    
    predicted_labels = zeros(n_te_samples, 1);

    for i=1:n_te_samples
        obs = features_test{i}';
        prob = zeros(n_classes, 1);
        
        for c=1:n_classes
            [~, prob(c)] = hmmdecode(obs, A_of_models{c}, B_of_models{c});
        end
        prob(isnan(prob)) = -inf;
        
        [~, predicted_ind] = max(prob);
        predicted_labels(i) = unique_classes(predicted_ind);
        classes_prob = prob./sum(prob);
        
    end
    
    % evaluation
    class_wise_accuracy = zeros(n_classes, 1);    
    confusion_matrix = zeros(n_classes, n_classes);    
    for i = 1:n_classes     % arranged by order
        temp = find(te_labels == unique_classes(i));
        class_wise_accuracy(i) =...
            length(find(predicted_labels(temp) == unique_classes(i))) / length(temp);
        
         confusion_matrix(i, :) = hist(predicted_labels(temp),...
             unique_classes) / length(temp);
    end
    
    total_accuracy = length(find(te_labels == predicted_labels))...
        / n_te_samples;

end

    
function [features_train, features_test] = clustering(features_train,...
    features_test, n_clusters)
    % clustering
    n_train_samples = length(features_train);
    n_test_samples = length(features_test);
    n_frames = size(features_train{1}, 1);
    
    frames_train = cell2mat(features_train);
    frames_test = cell2mat(features_test);

    [v_words, centers] = kmeans(frames_train, n_clusters);
    features_train = mat2cell(v_words, ones(n_train_samples, 1) * n_frames, 1);

    v_words = kmeans(frames_test, n_clusters, 'Start', centers);
    features_test = mat2cell(v_words, ones(n_test_samples, 1) * n_frames, 1);

end
