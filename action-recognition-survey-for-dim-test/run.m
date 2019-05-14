% feature_types = {'absolute_joint_positions', 'relative_joint_positions',...
%                  'joint_angles_quaternions', 'SE3_lie_algebra_absolute_pairs',...
%                  'SE3_lie_algebra_relative_pairs', 'relative_joint_IP_positions',...
%                  'relative_joint_IP_angle_positions', 'relative_IP_se3',...
%                  'eigenjoints', 'trajectory_3'};

% datasets = {'UTKinect', 'Florence3D', 'MSRAction3D', 'G3D', 'MSRPairs'};


%% init
addpath(genpath('./time_modeling'))
addpath(genpath('./classification'))
addpath(genpath('./data'))

desired_dim = 50:50:500;
avg_total_accuracy = zeros(length(desired_dim), 1);

feature_types = {'eigenjoints'};
datasets = {'MSRAction3D'};

[data, labels] = generate_features('MSRAction3D', 'eigenjoints', 76);


%% LDA test

tic
disp('LDA Test Start');

parfor d=1:length(desired_dim)
    fprintf('Test dimensionality:%d\n', desired_dim(d));

    avg_total_accuracy(d) = warped_pyramid_fourier_modeling(...
        data, labels, 1, 1, feature_types, datasets, 'lda', desired_dim(d));
end
toc
save('lda_analysis', avg_total_accuracy);

%% PCA test
tic
disp('PCA Test Start');
parfor d=1:length(desired_dim)
    fprintf('Test dimensionality:%d\n', desired_dim(d));

    avg_total_accuracy(d) = warped_pyramid_fourier_modeling(...
        data, labels, 1, 1, feature_types, datasets, 'pca', desired_dim(d));
end
toc
save('pca_analysis', avg_total_accuracy);
