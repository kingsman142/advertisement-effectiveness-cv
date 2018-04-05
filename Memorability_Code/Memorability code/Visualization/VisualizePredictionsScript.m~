%% Visualize predictions script
% run SVR_stat_features_2.m (e.g.) before running this

addpath('/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/Visualization');

N = size(test_label,1);

test_set_img = img(:,:,:,test_indices);

k = 25;


%% visualize actual
[p i] = sort(test_label,'descend');

subplot(231)
curr_actual = test_label(i(1:k));
VisualizeImages(i,test_set_img,1:k,sprintf('High actual memorability\nmean: %g, min: %g, max: %g', ...
    mean(curr_actual), min(curr_actual), max(curr_actual)));

subplot(232)
curr_actual = test_label(i((floor(N/2-k/2)+1):(floor(N/2+k/2))));
VisualizeImages(i,test_set_img,(floor(N/2-k/2)+1):(floor(N/2+k/2)),sprintf('Middle actual memorability\nmean: %g, min: %g, max: %g', ...
    mean(curr_actual), min(curr_actual), max(curr_actual)));

subplot(233)
curr_actual = test_label(i((N-k+1):N));
VisualizeImages(i,test_set_img,(N-k+1):N,sprintf('Low actual memorability\nmean: %g, min: %g, max: %g', ...
    mean(curr_actual), min(curr_actual), max(curr_actual)));



%% visualize predictions
[p i] = sort(predictions,'descend');

subplot(234)
curr_actual = test_label(i(1:k));
VisualizeImages(i,test_set_img,1:k,sprintf('High predicted memorability\nmean: %g, min: %g, max: %g', ...
    mean(curr_actual), min(curr_actual), max(curr_actual)));

subplot(235)
curr_actual = test_label(i((floor(N/2-k/2)+1):(floor(N/2+k/2))));
VisualizeImages(i,test_set_img,(floor(N/2-k/2)+1):(floor(N/2+k/2)),sprintf('Middle predicted memorability\nmean: %g, min: %g, max: %g', ...
    mean(curr_actual), min(curr_actual), max(curr_actual)));

subplot(236)
curr_actual = test_label(i((N-k+1):N));
VisualizeImages(i,test_set_img,(N-k+1):N,sprintf('Low predicted memorability\nmean: %g, min: %g, max: %g', ...
    mean(curr_actual), min(curr_actual), max(curr_actual)));



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



%% scatter images by prediction versus actual hr

% choose a random subset of the images to display
n_desired = 1000;
p = randperm(size(test_indices,1));
x = sortrows([p' test_indices test_label predictions]);
test_indices_sampled = x(:,2);
test_label_sampled = x(:,3);
predictions_sampled = x(:,4);
test_set_img_sampled = img(:,:,:,test_indices_sampled(1:n_desired));

% spaces visualization of prediction versus actual hr
showSpaceImages(test_set_img_sampled, test_label_sampled(1:n_desired), predictions_sampled(1:n_desired), 1.0);

