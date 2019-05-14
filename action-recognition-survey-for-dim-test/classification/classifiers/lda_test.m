%% 二维测试
samples = rand(50, 2) * 5;
labels = ones(50, 1);
y = samples(:, 2);
x = samples(:, 1);
labels(2 * x - 0.2 < y) = 1;
labels(1.3 * x - 0.2 < y <= 2 * x - 0.2) = 2;
labels(y <= 1.3 * x - 0.2 ) = 3;

figure;

% subplot(2, 2, 1);
% gscatter(samples(:,1), samples(:,2), labels);
% hold on
% grid on


W=lda(samples, labels,unique(labels),1);
Y=zeros(size(samples,1), 2);
for i=1:size(samples,1)
    Y(i,1:2)=samples(i,1:2)*W;
end

subplot(2, 2, 1);
gscatter(Y(:,1), Y(:,2), labels, 'rbg');
hold on
grid on

subplot(2, 2, 2);
gscatter(zeros(length(labels), 1), Y(:, 2), labels, 'rbg', '*');
% 只要他们能被一条线分开，就可以使用LDA算法
% 抽取的特征在高维度

%% 多kernel测试
unique_labels = unique(labels);

hold on
grid on

Y = samples;
for c=1:length(unique_labels)
    kernal_labels = labels == unique_labels(c);

    W = lda(Y, kernal_labels, unique(kernal_labels),1);

    for i=1:size(Y, 1)
        Y(i, :) = Y(i, :) * W;
    end
    
    subplot(2, 2, 4);
    gscatter(zeros(length(labels), 1), Y(:, 2), labels, 'rbg', '*');  
    
    subplot(2, 2, 3);
    gscatter(Y(:, 1), Y(:, 2), labels, 'rbg');  
    
    drawnow
    pause
end

%% 三维测试
samples = rand(50, 3) * 5;
labels = ones(50, 1);
z = samples(:, 3);
y = samples(:, 2);
x = samples(:, 1);
labels(1.8 * y + 0.5< z) = 1;
labels(0.5 * y + 0.5< z <= 1.8 * y - 0.5) = 2;
labels(z <= 0.5 * y - 0.5) = 3;
colors_and_labels = {'r*', 'b*', 'g*'};
unique_labels = unique(labels);

figure;
subplot(1, 2, 1);
for c=1:length(unique_labels)
    grpidx = labels == unique_labels(c);
    scatter3(samples(grpidx,1), samples(grpidx,2), samples(grpidx,3),...
        colors_and_labels{c});
    hold on
end
grid on

W = lda(samples, labels, unique(labels), 2);
Y = zeros(size(samples,1), 3);
for i=1:size(samples,1)
    Y(i, :) = samples(i, :) * W;
end

subplot(1, 2, 2);
gscatter(Y(:, 2), Y(:, 3), labels);

