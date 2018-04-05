%% Visualize one set


%% Visualize predicted sets

function [] = VisualizeOnePredictedSet(predicted, test_label, test_set_img, which_set, k)

    N = size(test_label,1);
    
    [p i] = sort(predicted,'descend');
    
    if (strcmp(which_set,'top'))
        range = 1:k;
        MontageSubplots(test_set_img(:,:,:,i(range)));
        curr_actual = test_label(i(range));
        title(sprintf('Top memorability\nactual mean: %g, min: %g, max: %g', mean(curr_actual), min(curr_actual), max(curr_actual)));
    elseif (strcmp(which_set,'middle'))
        range = (floor(N/2-k/2)+1):(floor(N/2+k/2));
        MontageSubplots(test_set_img(:,:,:,i(range)));
        curr_actual = test_label(i(range));
        title(sprintf('Middle memorability\nactual mean: %g, min: %g, max: %g', mean(curr_actual), min(curr_actual), max(curr_actual)));
    elseif (strcmp(which_set,'bottom'))
        range = (N-k+1):N;
        MontageSubplots(test_set_img(:,:,:,i(range)));
        curr_actual = test_label(i(range));
        title(sprintf('Bottom memorability\nactual mean: %g, min: %g, max: %g', mean(curr_actual), min(curr_actual), max(curr_actual)));
    end
end
