n_class = 20;
n_frames = 76;
n_dim = 57;
n__tr_samples = 557;
skeletal_data = cell(n__tr_samples, 1);
labels = unidrnd(n_class, n__tr_samples, 1);

for c=1:n__tr_samples
	skeletal_data{c} = 5 * rand(n_frames, n_dim) - 2.5; 
end

n_te_samples = 270;
test_skeletal_data = cell(n_te_samples, 1);

for c=1:n_te_samples
    test_skeletal_data{c} = 5 * rand(n_frames, n_dim) - 2.5;
end

for video_ind=1:n_te_samples
    video = test_skeletal_data{video_ind};
    
    each_frame_dist = zeros(n_class, 1);
    for c=1:n_class

        videos_in_c = skeletal_data(labels == c);
        frames_in_c = cell2mat(videos_in_c);
        nn_ind = knnsearch(frames_in_c, video);

        % frames:norm(di-NNc_di)
        each_frame_norms = sum(abs(video - frames_in_c(nn_ind, :)), 1);

        each_frame_dist(c) = sum(each_frame_norms.^2);

    end
    
    [~, ind] = min(each_frame_dist);
end
