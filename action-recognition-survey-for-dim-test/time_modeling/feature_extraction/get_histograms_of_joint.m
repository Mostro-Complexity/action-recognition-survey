function [bin_prob] = get_histograms_of_joint(joint_locations, body_model, n_desired_frames)
    
    [n_coord, n_joints, ~] = size(joint_locations);
    joint_locations = get_absolute_position_features(joint_locations, body_model, n_desired_frames);
    joint_locations = reshape(joint_locations, n_coord, n_joints - 1, n_desired_frames);
    
    X = joint_locations(1, :, :);
    Y = joint_locations(2, :, :);
    Z = joint_locations(3, :, :);
    
    [alpha, theta, ~] = cart2sph(X, Y, Z); %[-pi, pi]  [-pi/2, pi/2]
    alpha = alpha + pi; %[0, 2*pi]  
    theta = theta + pi/2; %[0, pi]
    
    alpha_degrees = 0:pi/6:2*pi;
    theta_degrees = [0, pi/12, pi/4, 5*pi/12, 7*pi/12, 3*pi/4, 11*pi/12, pi];

    [bin_prob] = prob_voting(alpha, theta, alpha_degrees, theta_degrees);
%     
%     for i=1:size(bin_prob, 2)
%         imagesc(reshape(bin_prob(:,i),7,12));
%         drawnow;
%         pause(0.5);
%     end
%     
end



function [bin_prob] = prob_voting(alpha, theta, alpha_degrees, theta_degrees)
    
    n_theta_degrees = length(theta_degrees);
    n_alpha_degrees = length(alpha_degrees);
    n_frames = size(alpha, 3);
    bin_prob = zeros((n_theta_degrees-1) * (n_alpha_degrees-1), n_frames); % ordered by column
    
    offsets = gen_paired_offset([-1; 0 ;1]);
    
    for j=1:n_alpha_degrees - 1
        for i=1:n_theta_degrees - 1
            for s=1:size(offsets, 1)
                % indices of joints in bin(i,j)
                idx_theta = (theta_degrees(i) < theta) & (theta <= theta_degrees(i + 1));
                idx_alpha = (alpha_degrees(i) < alpha) & (alpha <= alpha_degrees(i + 1));

                % boundary selection
                [theta_2, theta_1, alpha_2, alpha_1] = select_boundary(...
                    theta_degrees, alpha_degrees, i, j, offsets, s);
                
                % probability selection(independent of boundary selection)
                p_theta = normcdf(theta_2, theta, 1) - normcdf(theta_1, theta, 1);
                p_alpha = normcdf(alpha_2, alpha, 1) - normcdf(alpha_1, alpha, 1);
                p_ = p_theta.*p_alpha;
                p_(~(idx_theta&idx_alpha)) = 0;  % erase prob value not in bin(i,j)
                
                % probability filling in the corresponding bin
                % sum by time order
                bin_prob = fill_prob(bin_prob, sum(p_), i, j, offsets, s,...
                    n_theta_degrees, n_alpha_degrees);
        
            end
        end
    end
end


function [offsets] = gen_paired_offset(elems)
    n_elems = length(elems);
    offsets = zeros(n_elems^2, 2);
    
    for i=1:n_elems
        offsets(n_elems*(i-1) + 1:n_elems*i, 2) = elems;
        offsets(n_elems*(i-1) + 1:n_elems*i, 1) = elems(i);
    end
end


function [theta_2, theta_1, alpha_2, alpha_1] = ...
    select_boundary(theta_degrees, alpha_degrees, prob_row, prob_col, offsets, offset_idx)
    n_theta_degrees = length(theta_degrees);
    n_alpha_degrees = length(alpha_degrees);

    row = prob_row + offsets(offset_idx, 1);
    col = prob_col + offsets(offset_idx, 2);
    
    if (col <= 0)
        alpha_2 = alpha_degrees(1);
        alpha_1 = alpha_degrees(1) - pi/12;   
    elseif (col >= n_alpha_degrees)
        alpha_2 = alpha_degrees(n_alpha_degrees) + pi/12;
        alpha_1 = alpha_degrees(n_alpha_degrees);
    else
        alpha_2 = alpha_degrees(col + 1);
        alpha_1 = alpha_degrees(col);
    end
    
    if (row <= 0)
        theta_2 = theta_degrees(1);
        theta_1 = theta_degrees(1) - pi/6;
    elseif (row >= n_theta_degrees)
        theta_2 = theta_degrees(n_theta_degrees) + pi/6;
        theta_1 = theta_degrees(n_theta_degrees);
    else
        theta_2 = theta_degrees(row + 1);
        theta_1 = theta_degrees(row);
    end
end

function [bin_prob] = fill_prob(bin_prob, sum_by_time, i, j, offsets, offset_idx,...
    n_theta_degrees, n_alpha_degrees)
    
    row = i + offsets(offset_idx, 1);
    col = j + offsets(offset_idx, 2);

    position = 0;
    if (col <= 0)
        position = position + (n_theta_degrees-1)*(n_alpha_degrees-2);
    elseif (col >= n_alpha_degrees) 
        position = 0; 
    else
        position = position + (n_theta_degrees-1)*(col-1);
    end
    
    if (row <= 0)
        position = position + n_theta_degrees - 1;
    elseif (row >= n_theta_degrees)
        position = position + 1;
    else
        position = position + row;
    end
    
    bin_prob(position, :) = bin_prob(position, :)' + squeeze(sum_by_time); 

end