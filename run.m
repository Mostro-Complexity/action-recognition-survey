% feature_types = {'absolute_joint_positions', 'relative_joint_positions',...
%                  'joint_angles_quaternions', 'SE3_lie_algebra_absolute_pairs',...
%                  'SE3_lie_algebra_relative_pairs', 'relative_joint_IP_positions',...
%                  'relative_joint_IP_angle_positions', 'relative_IP_se3',...
%                  'eigenjoints', 'trajectory_3'};

% datasets = {'UTKinect', 'Florence3D', 'MSRAction3D', 'G3D', 'MSRPairs'};


%% super awesome model testing
% writing by others

% feature_types = {'absolute_joint_positions', 'relative_joint_positions', ...
%     'eigenjoints'};
% 
% datasets = {'MSRAction3D'};
% 
% tic
% for i = 1:length(datasets)
%     for j = 1:length(feature_types)
%         warped_pyramid_fourier_modeling(i, j, feature_types, datasets);
%     end
% end
% toc


%% my stupid model
% 
% feature_types = {'absolute_joint_positions', 'relative_joint_positions', ...
%     'eigenjoints'};
% 
% datasets = {'MSRAction3D'};
% 
% tic
% for i = 1:length(datasets)
%     for j = 1:length(feature_types)
%         general_modeling(i, j, feature_types, datasets);
%     end
% end
% toc

%% nbnn model
% 'histograms_of_3D_joint_locations ',
% feature_types = {'absolute_joint_positions', ...
%     'relative_joint_positions', 'eigenjoints'};
% 
% datasets = {'MSRAction3D'};
% 
% tic
% for i = 1:length(datasets)
%     for j = 1:length(feature_types)
%         naive_bayes_nearest_neighbor_modeling(i, j, feature_types, datasets);
%     end
% end
% toc

%% hmm model
feature_types = {'absolute_joint_positions', ...
    'relative_joint_positions', 'eigenjoints'};

datasets = {'MSRAction3D'};

tic
for i = 1:length(datasets)
    for j = 1:length(feature_types)
        hidden_markov_model_modeling(i, j, feature_types, datasets);
    end
end
toc