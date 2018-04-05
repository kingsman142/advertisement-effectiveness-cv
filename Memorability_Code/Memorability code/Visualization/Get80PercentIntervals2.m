%% Return vector of 80% confidence interval on ys

function [intervals] = Get80PercentIntervals(ys)

    [N M] = size(ys);
    
    intervals = zeros(M,2);
    
    N_10 = ceil(0.1*N);
    N_90 = floor(0.9*N);
    
    for i=1:M
        
        u = mean(ys(:,i));
        
        [sorted jj] = sort(ys(:,i),'ascend');
        
        intervals(i,1) = sorted(N_10) - u;
        intervals(i,2) = sorted(N_90) - u;
    end
    
end