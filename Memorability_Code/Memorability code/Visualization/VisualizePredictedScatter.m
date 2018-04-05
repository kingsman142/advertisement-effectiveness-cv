%% Visualize predicted vs actual scatter images

function [] = VisualizePredictedScatter(predicted, test_label, test_indices, img, n_desired)

    N = size(test_label,1);
    
    predictions = predicted;

    
    %% scatter images by prediction versus actual hr

    % choose a random subset of the images to display
    p = randperm(size(test_indices,1));
    x = sortrows([p' test_indices test_label predictions]);
    test_indices_sampled = x(:,2);
    test_label_sampled = x(:,3);
    predictions_sampled = x(:,4);
    test_set_img_sampled = img(:,:,:,test_indices_sampled(1:n_desired));

    % spaces visualization of prediction versus actual hr
    showSpaceImages(test_set_img_sampled, test_label_sampled(1:n_desired), predictions_sampled(1:n_desired), 1.0);

end
