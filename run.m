% feature_types = {'absolute_joint_positions', 'relative_joint_positions',...
%                  'joint_angles_quaternions', 'SE3_lie_algebra_absolute_pairs',...
%                  'SE3_lie_algebra_relative_pairs', 'relative_joint_IP_positions',...
%                  'relative_joint_IP_angle_positions', 'relative_IP_se3',...
%                  'eigenjoints', 'trajectory_3'};

% datasets = {'UTKinect', 'Florence3D', 'MSRAction3D', 'G3D', 'MSRPairs'};

feature_types = {'eigenjoints'};
datasets = {'MSRAction3D'};

tic
for i = 1:length(datasets)
    for j = 1:length(feature_types)
        skeletal_action_classification(i, j, feature_types, datasets);
    end
end
toc

%{
clear all;
clc;
%%
e = 2; g = 1;
[x,y] = meshgrid(0:20,0:15);  % This makes regular grid
u = e*x-g*y;                  % Linear velocity field
v = g*x-e*y;
[phi,psi] = flowfun(u,v);  % Here comes the potential and streamfun.
%
contour(phi,20,'--r','Displayname','\phi')   % Contours of potential
hold on
contour(psi,20,'-g','Displayname','\psi')    % Contours of streamfunction
quiver(x,y,u,v,'Displayname','velocity')         % Now superimpose the velocity field
legend show;
%}