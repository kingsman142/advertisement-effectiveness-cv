%% Density plot with binned averages line of predicted values versus actual
%% values

function [] = PlotScatterPredictedVsActual(actual, predicted)
    
    x = [actual, predicted];
    x = sortrows(x);
    
    % count number of times each (actual, predicted) pair occurs
    %{
    last = [-1 -1];
    y = zeros(length(x),1);
    j = -1;
    for i=1:size(x,1)
        if (x(i,1) == last(1) && x(i,2) == last(2) && j ~= -1)
            y(j:i) = y(j) + 1;
        else
            y(i) = x(i);
            j = i;
        end
    end
    %}
    
    h = scatter(actual, predicted, 5, 'b');%, 10, 1-y, 'filled'); % 20*y, 1-y, 'filled');
    colormap('gray');
    %[x i] = sort(actual,'descend');
    %[x i] = sort(predicted,'descend');
    %hold on;
    %num_top = 400;
    %scatter(actual(i(1:num_top)), predicted(i(1:num_top)), '.', 'r');
    
    h = lsline;
    set(h,'Color','b');
    
    xlabel('actual hit rate');
    ylabel('predicted hit rate');
    set(gca,'XLim',[0.2,1]);
    set(gca,'YLim',[0.2,1]);
    axis('square');
    
    %c = get(h,'Children');
    %set(c,'FaceAlpha',0.5);
    
end