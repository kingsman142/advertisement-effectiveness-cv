%% Make dist uniform by removing

function [indices2] = MakeDistUniformByRemoving(labels, indices)

    N = size(labels,1);
    bin_size = 0.08;
    
    %% set desired_number_per_bin to be equal to the greatest count_in_bin
    min_desired_per_bin = 10;
    desired_number_per_bin = -1;
    for i=0.20:bin_size:1-bin_size
        count_in_bin = sum(labels>i & labels<=i+bin_size);
        if ((desired_number_per_bin == -1 || count_in_bin < desired_number_per_bin) && count_in_bin >= min_desired_per_bin)
            desired_number_per_bin = count_in_bin;
        end
    end
    desired_number_per_bin
    
    indices2 = [];
    
    %% resample each bin so that they become equally sized
    for i=0.20:bin_size:1-bin_size
        
        curr_bin_indices = indices(labels>i & labels<=i+bin_size);
        
        if (length(curr_bin_indices) >= desired_number_per_bin)
        
            p = randperm(length(curr_bin_indices));
            x = sortrows([p' curr_bin_indices]);
            curr_bin_indices = x(:,2);
            
            indices2 = [indices2; curr_bin_indices(1:desired_number_per_bin)];
            
            length(curr_bin_indices(1:desired_number_per_bin))
        end
    end

end