%% Visualize error sets

function [] = VisualizeErrorSets(predicted, test_label, test_set_img)

    N = size(test_label,1);

    k = 25;
    
    predictions = predicted;

    
    %% visualize worst predictions
    errs = predictions-test_label;
    [p i] = sort(errs,'descend');

    subplot(131)
    curr_errs = errs(i(1:k));
    VisualizeImages(i,test_set_img,1:k,sprintf('Predict high, actual low\nmean error: %g, min: %g, max: %g', ...
        mean(curr_errs), min(curr_errs), max(curr_errs)));

    subplot(132)
    curr_errs = errs(i((floor(N/2-k/2)+1):(floor(N/2+k/2))));
    VisualizeImages(i,test_set_img,(floor(N/2-k/2)+1):(floor(N/2+k/2)),sprintf('Predict == actual\nmean error: %g, min: %g, max: %g', ...
        mean(curr_errs), min(curr_errs), max(curr_errs)));

    subplot(133)
    curr_errs = errs(i((N-k+1):N));
    VisualizeImages(i,test_set_img,(N-k+1):N,sprintf('Predict low, actual high\nmean error: %g, min: %g, max: %g', ...
        mean(curr_errs), min(curr_errs), max(curr_errs)));
    
end