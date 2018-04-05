%% Visualize empirical sets

function [] = VisualizeEmpiricalSets(actual, img, k)

    N = size(actual,1);


    %% visualize actual
    [p i] = sort(actual,'descend');

    subplot(131)
    curr_actual = actual(i(1:k));
    VisualizeImages(i,img,1:k,sprintf('High actual memorability\nmean: %g, min: %g, max: %g', ...
        mean(curr_actual), min(curr_actual), max(curr_actual)));

    subplot(132)
    curr_actual = actual(i((floor(N/2-k/2)+1):(floor(N/2+k/2))));
    VisualizeImages(i,img,(floor(N/2-k/2)+1):(floor(N/2+k/2)),sprintf('Middle actual memorability\nmean: %g, min: %g, max: %g', ...
        mean(curr_actual), min(curr_actual), max(curr_actual)));

    subplot(133)
    curr_actual = actual(i((N-k+1):N));
    VisualizeImages(i,img,(N-k+1):N,sprintf('Low actual memorability\nmean: %g, min: %g, max: %g', ...
        mean(curr_actual), min(curr_actual), max(curr_actual)));

end

