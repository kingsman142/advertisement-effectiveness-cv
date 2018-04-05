%% Visualize one error set

function [] = VisualizeOneErrorSet(predicted, test_label, test_set_img, which_set, k)

    N = size(test_label,1);
    
    %errs = predicted-test_label; % abs error
    
    % rank error
    [foo i1] = sort(predicted,'descend');
    [foo i2] = sort(i1,'ascend');
    [foo j1] = sort(test_label,'descend');
    [foo j2] = sort(j1,'ascend');
    errs = j2-i2;
    
    [p i] = sort(errs,'descend');
    
    if (strcmp(which_set,'too high'))
        range = 1:k;
        MontageSubplots(test_set_img(:,:,:,i(range)));
        curr_errs = errs(i(range));
        title(sprintf('Predict high, actual low\nmean error: %g, min: %g, max: %g', mean(curr_errs), min(curr_errs), max(curr_errs)));
    elseif (strcmp(which_set,'too low'))
        range = (N-k+1):N;
        MontageSubplots(test_set_img(:,:,:,i(range)));
        curr_errs = errs(i(range));
        title(sprintf('Predict low, actual high\nmean error: %g, min: %g, max: %g', mean(curr_errs), min(curr_errs), max(curr_errs)));
    end
end