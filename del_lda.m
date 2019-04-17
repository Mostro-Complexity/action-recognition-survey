function [k] = lda(X, classes)
    
    unique_class = unique(classes);
    n_classes = length(unique_class);
    [n_X, n_dim] = size(X);
    
    avg_in_all = mean(X);
    
    
    Sw = zeros(n_dim, n_dim, n_classes);
    Sb = zeros(n_dim, n_dim, n_classes);
    
    for c=1:n_classes
        x_in_c = X(unique_class(c) == classes, :);
        n_x_in_c = size(x_in_c, 1);
        avg_in_c = mean(x_in_c);
        
        Sw(:, :, c) = (x_in_c - avg_in_c)'*(x_in_c - avg_in_c) * n_x_in_c;
        Sb(:, :, c) = (avg_in_c-avg_in_all)'*(avg_in_c-avg_in_all) * n_x_in_c;
    end
    
    Sw = sum(Sw, 3) / n_X;
    Sb = sum(Sb, 3) / n_X;

    [V, L] = eig(Sw\Sb);
    [a, b] = max(max(L));
    
    W = V(:,b);%最大特征值所对应的特征向量
    k = W(2)/W(1);
    b = 0;
    
    desired_X = cell(n_classes, 1);
    
    %计算第c类样本在直线上的投影点
    for c=1:n_classes
        x_in_c = X(unique_class(c) == classes, :);
        n_x_in_c = size(x_in_c, 1);

        xi = zeros(n_x_in_c, 1);
        for i=1:n_x_in_c
            y0 = x_in_c(i,2);
            x0 = x_in_c(i,1);
            xi(i) = (k *(y0 - b) + x0)/(k^2 + 1);
        end
        yi = k * xi + b;
        desired_X{c} = [xi, yi];
    end
    
end