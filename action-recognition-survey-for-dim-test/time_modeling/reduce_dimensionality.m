function [desired_features] = reduce_dimensionality(features, n_desired_dim, varargin)

    paramNames = {'Method','Labels'};
    defaults   = {'pca', []};
    [method, labels] = internal.stats.parseArgs(paramNames, defaults, varargin{:});

    if strcmp(method,'lda') && isempty(labels)
        error('Labels must be a cell mat with one column');
    end
    
    desired_features = cell(length(features), 1);
    
    if strcmp(method, 'pca')
        
        for i=1:length(features)
            feature_before_PCA = features{i}';
            [coeff] = pca(feature_before_PCA, 'Economy', false, 'Centered', false);
            feature_after_PCA = feature_before_PCA * coeff;

            if(size(feature_after_PCA, 2) <= n_desired_dim)
                feature_after_PCA = feature_after_PCA';
            else
                feature_after_PCA = feature_after_PCA(:, 1:n_desired_dim)';
            end

            desired_features{i} = feature_after_PCA;
        end      
        
    elseif strcmp(method, 'lda')
        
    unique_classes = unique(labels);
    n_classes = length(unique_classes);

    n_train_samples = length(features);
    [n_dim, n_frames] = size(features{1});
    
    frames = cell2mat(features')';
    
    desired_labels = zeros(n_train_samples * n_frames, 1);
    
    temp = 0;
    for c=1:n_classes
        n_frames_in_c = length(labels(unique_classes(c) == labels));
        desired_labels(temp + 1:temp + n_frames_in_c) = unique_classes(c);
        temp = temp + n_frames_in_c;
    end
    
    W = lda(frames, desired_labels, unique_classes, n_desired_dim);
    
    for i=1:size(frames, 1)
        frames(i, :) = frames(i, :) * W;
    end
    
    frames = real(frames);
    
    features = mat2cell(frames,...
        ones(n_train_samples, 1) * n_frames, [n_dim - n_desired_dim, n_desired_dim]);

    desired_features = features(:, 2);
        
    end
end

