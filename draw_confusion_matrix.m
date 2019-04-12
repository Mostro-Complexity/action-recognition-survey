function draw_confusion_matrix(mat, tick, n_classes)
    n_classes = length(tick);
    imagesc(1:n_classes, 1:n_classes,mat);            %# in color

    colormap(flipud(gray));  %# for gray; black for large value.

    textStrings = num2str(mat(:),'%0.2f');  
    textStrings = strtrim(cellstr(textStrings)); 

    [x,y] = meshgrid(1:n_classes); 

    hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
    midValue = mean(get(gca,'CLim')); 
    textColors = repmat(mat(:) > midValue,1,3); 

    set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
    set(gca,'xticklabel',tick,'XAxisLocation','top');
    set(gca, 'XTick', 1:n_classes, 'YTick', 1:n_classes);
    set(gca,'yticklabel',tick);
    xtickangle(50)

%     rotateXLabels(gca, 315 );% rotate the x tick

end 

 

 
