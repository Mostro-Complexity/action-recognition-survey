function integrate_depth_videos

    tic;
    dbstop if error
    warning off 
    dir = ['data/MSRAction3D/'];

    load(['data/MSRAction3D', '/body_model'])

    n_actions = 20;
    n_subjects = 10;
    n_instances = 3;

    % depth_videos = cell(n_actions, n_subjects, n_instances);
    depth_videos_validity = zeros(n_actions, n_subjects, n_instances);

    frame_length = 240;
    frame_width = 320;
    desired_length = 32;
    desired_width = 32;
    desired_frames = 38;
    
    for a = 1:n_actions
        for s = 1:n_subjects
            for e = 1:n_instances  
                filename = [dir, 'Depth/', sprintf('a%02i_s%02i_e%02i_sdepth.bin',a,s,e)]; 

                if(exist(filename,'file'))
                    videos_dir = [dir, '/Depth_Mat'];
                    mkdir(videos_dir);

                    depth_videos_validity(a, s, e) = 1;

                    video = load_depth_map(filename);
                    n_frames = length(video);
                    
                    % normalization            
                    video_array = cell2mat(video); 
                    video_array = reshape(video_array, frame_length * frame_width, n_frames);   
                    video_array = zscore(video_array);
                    % video_array = mapminmax(video_array', -1, 1)'; 
                    % video_array = video_array./max(abs(video_array));
                    video_array = reshape(video_array, frame_length, frame_width, n_frames);  
                 
                    % bounding box cutting and resize frames
                    video_array = cut_bounding_box(video_array, desired_length, desired_width);

                    % video resizing and interpolation
                    video_array = imresize3(video_array,...
                        [desired_length, desired_width, desired_frames]);
                    
                    % resize time length of depth sequence

                    filename = [videos_dir, '/', sprintf('a%02i_s%02i_e%02i_sdepth',a,s,e)];             
                    save(filename, 'video_array');

                end
            end
        end
    end

    toc;
    
end
    

function [desired_video_array] = cut_bounding_box(video_array, desired_length, desired_width)
    n_frames = size(video_array, 3);
    desired_video_array = zeros(desired_length, desired_width, n_frames);
    for i=1:n_frames
        frame = video_array(:, :, i);
        
        width_range = find(any(frame, 1));
        length_range = find(any(frame, 2));
        
        w_lower = width_range(1);
        w_upper = width_range(end);
        l_lower = length_range(1);
        l_upper = length_range(end);
        
        desired_frame = frame(l_lower:l_upper, w_lower:w_upper);
        desired_video_array(:, :, i) = imresize(desired_frame,...
            [desired_length, desired_width]);
    end
end
