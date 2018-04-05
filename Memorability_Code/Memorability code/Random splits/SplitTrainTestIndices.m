%% SplitTrainTestIndices

function [train_indices test_indices] = SplitTrainTestIndices(max_index)

    % rand sort
    p = randperm(max_index);
    
    % split half
    half = ceil(max_index/2);
    train_indices = p(1:half);
    test_indices = p((half+1):end);
end