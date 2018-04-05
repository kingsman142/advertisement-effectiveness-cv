%% MakeDistUniform

function [indices2] = MakeDistUniform(labels, indices)

    N = size(labels,1);
    bin_size = 0.1;
    
    %% set desired_number_per_bin to be equal to the greatest count_in_bin
    desired_number_per_bin = -1;
    for i=0:bin_size:1-bin_size
        count_in_bin = sum(labels>=i & labels<=i+bin_size);
        if (desired_number_per_bin == -1 || count_in_bin > desired_number_per_bin)
            desired_number_per_bin = count_in_bin;
        end
    end
    
    %image_features2 = image_features;
    %labels2 = labels;
    indices2 = indices;
    
    %% resample each bin so that they become equally sized
    for i=0:bin_size:1-bin_size
       
        %curr_bin_image_features = image_features(labels>=i & labels<=i+bin_size,:);
        %curr_bin_labels = labels(labels>=i & labels<=i+bin_size);
        curr_bin_indices = indices(labels>=i & labels<=i+bin_size);
        
        count_in_bin = size(curr_bin_indices,1);
        
        if (count_in_bin > 0)
            
            rand_addition_index = randi(count_in_bin, desired_number_per_bin-count_in_bin, 1);
            %image_features2 = [image_features2; curr_bin_image_features(rand_addition_index,:)];
            %labels2 = [labels2; curr_bin_labels(rand_addition_index)];
            indices2 = [indices2; curr_bin_indices(rand_addition_index)];
            
        end
        
    end

end