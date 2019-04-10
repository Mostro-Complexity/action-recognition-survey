function [features_after_PCA] = reduce_dimensionality(features, desired_dim)
    features_after_PCA{length(features)} = [];
    
    for i=1:length(features)
        feature_before_PCA = features{i}';
        [coeff] = pca(feature_before_PCA, 'Economy', false, 'Centered', false);
        feature_after_PCA = feature_before_PCA * coeff;
        
        if(size(feature_after_PCA, 2) <= desired_dim)
            feature_after_PCA = feature_after_PCA';
        else
            feature_after_PCA = feature_after_PCA(:, 1:desired_dim)';
        end
        
        features_after_PCA{i} = feature_after_PCA;
    end      
end

