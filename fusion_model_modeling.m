function [] = fusion_model_modeling(dataset_idx, feature_idx,...
    feature_types, datasets)


    warning off  
    dbstop if error
    
    addpath(genpath('./time_modeling'))
    addpath(genpath('./data'))
    addpath(genpath('./classification'))


    if (feature_idx > 10)
        error('Feature index should be less than 11');
    end

    if (dataset_idx > 5)
        error('Dataset index should be less than 6');
    end
    
    
    if (strcmp(datasets{dataset_idx}, 'UTKinect'))       
        desired_frames = 74;  

    elseif (strcmp(datasets{dataset_idx}, 'Florence3D'))
        desired_frames = 35;
        
    elseif (strcmp(datasets{dataset_idx}, 'MSRAction3D'))
        desired_frames = 76;
        
    elseif (strcmp(datasets{dataset_idx}, 'G3D'))
        desired_frames = 100;
    
    elseif (strcmp(datasets{dataset_idx}, 'MSRPairs'))
        desired_frames = 111;
        
    else
        error('Unknown dataset')
    end
    
        directory = [datasets{dataset_idx}, '_experiments/', feature_types{feature_idx}];
    mkdir(directory)

    
    % Training and test subjects
    tr_info = load(['data/', datasets{dataset_idx}, '/tr_te_splits']);
    n_tr_te_splits = size(tr_info.tr_subjects, 1);

    %% Skeletal representation
    disp ('Generating skeletal representation')
    generate_features(directory, datasets{dataset_idx}, feature_types{feature_idx}, desired_frames);
    advanced_features_dir = ['data/', datasets{dataset_idx}, '/Advanced_Feat_Mat'];
    for i=1:n_tr_te_splits
        copyfile([advanced_features_dir, sprintf('/advanced_split_%d.mat',i)], directory)
    end
    clear('advanced_features_dir');
    %% Classification
    disp ('Classification')
    
    action_names = load(['data/', datasets{dataset_idx}, '/action_names'], 'action_names');
    labels = load([directory, '/labels'], 'action_labels', 'subject_labels');

    fusion_model_classification(directory, labels.subject_labels,...
        labels.action_labels, tr_info.tr_subjects, tr_info.te_subjects, action_names.action_names);

%     if (strcmp(datasets{dataset_idx}, 'MSRAction3D'))
%         hmm_classification_with_subsets(directory, labels.subject_labels,...
%             labels.action_labels, tr_info.tr_subjects, tr_info.te_subjects,...
%             tr_info.action_sets, action_names.action_names);
%     end    
    
    %% Finishing    
    delete([directory, '/labels.mat']);
    delete([directory, '/features.mat']);
    for i=1:n_tr_te_splits
        delete([directory, sprintf('/advanced_split_%d.mat',i)])
    end
end

