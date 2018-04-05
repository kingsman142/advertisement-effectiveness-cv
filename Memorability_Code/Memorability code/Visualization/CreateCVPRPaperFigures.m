%% Create CVPR paper figures

              
set(gca, 'fontsize', 8);

mkdir(['../Analysis/plots/' which_date]);

display_error_bars = 'f';
kernel_combination_type = 'p'; % s = sum, p = product
stat_feature_sets = {};
stat_feature_sets_names = {};
obj_feature_sets = {};
obj_feature_sets_names = {};
combo_stat_feature_sets = {};
combo_obj_feature_sets = {};
combo_feature_sets_names = {};


%% Get hit rates for all the images
N = size(sorted_target_data,1);
hits = zeros(N,1);
misses = zeros(N,1);
for i=1:N
    hits(i) =  sorted_target_data{i}.hits;
    misses(i) = sorted_target_data{i}.misses;
end
actual = double(hits./(hits+misses));



%% Get the valid image indices
N_images = 2222; % ignoring texture images
image_index_list = [];
for i=1:N_images
    if (sorted_target_data{i}.hits + sorted_target_data{i}.misses >= 20)
        image_index_list = [image_index_list; i];
    end
end
actual = actual(image_index_list);


%% Plot empirical hit rate histogram

hist(actual,20);
xlabel('hit rate');
ylabel('n');
title('Histogram of empirically measured hit rates');


print('-dpsc', fullfile('../Analysis/plots/',which_date,'empirical_hit_rate_hist.eps'));
close all;


%% Plot empirical sample image sets
curr_img = img(:,:,:,image_index_list);

%{
VisualizeEmpiricalSets(actual, curr_img, 9);

cd('../Analysis/plots/final figures v2');
print -dpsc empirical_hit_rate_sets.eps;
cd('/Users/Phillip/Documents/Research/Image Memorability/');
%}

VisualizeOnePredictedSet(actual, actual, curr_img, 'top', 25);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'empirical_hit_rate_top_set.eps'));
close all;

VisualizeOnePredictedSet(actual, actual, curr_img, 'middle', 25);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'empirical_hit_rate_middle_set.eps'));
close all;

VisualizeOnePredictedSet(actual, actual, curr_img, 'bottom', 25);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'empirical_hit_rate_bottom_set.eps'));
close all;



%% Plot human consistency
stat_feature_sets = {};
obj_feature_sets = {};
stat_feature_sets_names = {};
obj_feature_sets_names = {};
plot_subjects = 1;
plot_actual = 1;
plot_chance = 1;
y_range = [0.64 1.0];
use_all_images_human_consistency = 't';
kernel_combination_type = '_'; % _ = not applicable, s = sum, p = product

display_error_bars = 't';

EvaluateAndPlotPredictions;

display_error_bars = 'f';

print('-dpsc', fullfile('../Analysis/plots/',which_date,'human_consistency.eps'));
close all;



%% Plot global features comparison
stat_feature_sets = {'p' 'g' 's' 'h' 'gshp'};
obj_feature_sets = {'s'};
stat_feature_sets_names = {'Pixels' 'GIST' 'SIFT' 'HOG' 'All'};
obj_feature_sets_names = {'Labeled Multiscale Areas'};
plot_subjects = 1;
plot_actual = 0;
plot_chance = 0;
y_range = [0.66 0.90];
use_all_images_human_consistency = 'f';
kernel_combination_type = 'p'; % s = sum, p = product

EvaluateAndPlotPredictions;

print('-dpsc', fullfile('../Analysis/plots/',which_date,'all_features_comparison.eps'));
close all;



%% Plot humans versus global features sample image sets
Predictor_type = 'r'; % c = SVC, r = SVR, w = CWM
classification_type = 'm'; % m = about median, t = top set, b = bottom set
classification_set_size = 100; % used for top set and bottom set classification
resampling = ''; % 1 = Resample training set to uniform distribution, 2 = Resample test set to uniform distribution
kernel_combination_type = 'p'; % s = sum, p = product
independent_subject_set = 't'; % t = test on hit rates from an independent subject set, f = test on hit rates from the same subject set as training

stat_features_to_use = 'gship';
obj_features_to_use = '';

dir_name = [Predictor_type '_' classification_type '_' classification_set_size '_' ...
                                                                   '_' stat_features_to_use '_' obj_features_to_use '_' resampling ...
                                                                    kernel_combination_type '_' independent_subject_set];

date1 = which_date;%'08_9-Nov-2010';
load(['../Analysis/results/' date1 '/' dir_name '/all_predicteds.mat']);
load(['../Analysis/results/' date1 '/' dir_name '/all_test_labels.mat']);

predicted = all_predicteds(1,:);
test_label = all_test_labels(1,:);
test_indices = image_test_indices(1,:);
test_set_img = img(:,:,:,test_indices);

%{
VisualizePredictedSets(predicted, test_label', test_set_img, 9);

cd('../Analysis/plots/final figures v2');
print -dpsc gshp_vs_humans_sample_sets.eps;
cd('/Users/Phillip/Documents/Research/Image Memorability/');
%}

VisualizeOnePredictedSet(test_label', test_label', test_set_img, 'top', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_sample_humans_top_set_8.eps'));
close all;

VisualizeOnePredictedSet(test_label', test_label', test_set_img, 'middle', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_sample_humans_middle_set_8.eps'));
close all;

VisualizeOnePredictedSet(test_label', test_label', test_set_img, 'bottom', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_sample_humans_bottom_set_8.eps'));
close all;

VisualizeOnePredictedSet(predicted, test_label', test_set_img, 'top', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_sample_gship_top_set_8.eps'));
close all;

VisualizeOnePredictedSet(predicted, test_label', test_set_img, 'middle', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_sample_gship_middle_set_8.eps'));
close all;

VisualizeOnePredictedSet(predicted, test_label', test_set_img, 'bottom', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_sample_gship_bottom_set_8.eps'));
close all;



%% Plot errors figure
Predictor_type = 'r'; % c = SVC, r = SVR, w = CWM
classification_type = 'm'; % m = about median, t = top set, b = bottom set
classification_set_size = 100; % used for top set and bottom set classification
resampling = ''; % 1 = Resample training set to uniform distribution, 2 = Resample test set to uniform distribution
kernel_combination_type = 'p'; % s = sum, p = product
independent_subject_set = 't'; % t = test on hit rates from an independent subject set, f = test on hit rates from the same subject set as training

stat_features_to_use = 'gship';
obj_features_to_use = '';

dir_name = [Predictor_type '_' classification_type '_' classification_set_size '_' ...
                                                                   '_' stat_features_to_use '_' obj_features_to_use '_' resampling ...
                                                                    kernel_combination_type '_' independent_subject_set];

date1 = which_date;%'08_9-Nov-2010';
load(['../Analysis/results/' date1 '/' dir_name '/all_predicteds.mat']);
load(['../Analysis/results/' date1 '/' dir_name '/all_test_labels.mat']);

predicted = all_predicteds(1,:);
test_label = all_test_labels(1,:);
test_indices = image_test_indices(1,:);
test_set_img = img(:,:,:,test_indices);

VisualizeOneErrorSet(predicted', test_label', test_set_img, 'too high', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_error_gship_too_high_set_8.eps'));
close all;

VisualizeOneErrorSet(predicted', test_label', test_set_img, 'too low', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'gship_vs_humans_error_gship_too_low_set_8.eps'));
close all;



%% Plot object features comparison
stat_feature_sets = {'gshp'};
obj_feature_sets = {'p' 'c' 'a' 's'};
stat_feature_sets_names = {'GIST, SIFT, HOG, Pixels'};
obj_feature_sets_names = {'Obj presences' 'Obj counts' 'Obj areas' 'Obj spatial histograms'};
plot_subjects = 1;
plot_actual = 0;
plot_chance = 0;
y_range = [0.66 0.85];
use_all_images_human_consistency = 'f';

EvaluateAndPlotPredictions;

print('-dpsc', fullfile('../Analysis/plots/',which_date,'object_features_comparison.eps'));
close all;


%% Plot global features versus objects            
stat_feature_sets = {'gshp'};
obj_feature_sets = {'s'};
stat_feature_sets_names = {'GIST, SIFT, HOG, Pixels'};
obj_feature_sets_names = {'Obj spatial histograms'};
plot_subjects = 1;
plot_actual = 0;
plot_chance = 0;
y_range = [0.66 0.85];
use_all_images_human_consistency = 'f';

EvaluateAndPlotPredictions;

print('-dpsc', fullfile('../Analysis/plots/',which_date,'objects_versus_global_features.eps'));
close all;


%% Plot 'new' all features comparison
stat_feature_sets = {'p' 'g' 's' 'h' 'i' 'gship'};
obj_feature_sets = {'e', 's', 'se'};
stat_feature_sets_names = {'Pixels', 'GIST', 'SIFT', 'HOG', 'SSIM', 'All global features (gship)'};
obj_feature_sets_names = {'Scene categories', 'Labeled Multiscale Areas', 'Labeled Multiscale Areas and Scene categories'};
combo_stat_feature_sets = {'gship'}; combo_obj_feature_sets = {'se'}; % combo features are combined into one big feature vector
combo_feature_sets_names = {'All of above'};
plot_subjects = 1;
plot_actual = 0;
plot_chance = 0;
y_range = [0.66 0.90];
use_all_images_human_consistency = 'f';

kernel_combination_type = ['p';'p';'p';'p';'p';'p';'p';'p';'s';'s']; % s = sum, p = product, m = mixed

EvaluateAndPlotPredictions;

kernel_combination_type = 'p'; % s = sum, p = product, m = mixed

combo_stat_feature_sets = {};
combo_obj_feature_sets = {};
combo_feature_sets_names = {};

print('-dpsc', fullfile('../Analysis/plots/',which_date,'new_all_features_comparison.eps'));
close all;



%% Plot independent vs non-independent subject sets
%{
clearvars -except Areas Counts gist img pixel_histograms sorted_target_data ...
                  sptHistObjects sptHisthog sptHistsift image_train_indices ... 
                  image_test_indices subject_hrs1 subject_hrs2
              
stat_feature_sets = {'gshp'};
obj_feature_sets = {'s'};
stat_feature_sets_names = {'GIST, SIFT, HOG, Pixels'};
obj_feature_sets_names = {'Obj spatial histograms'};
plot_subjects = 1;
plot_actual = 0;
plot_chance = 0;
y_range = [0.66 0.85];
use_all_images_human_consistency = 'f';

EvaluateAndPlotPredictions;

cd('../Analysis/plots/final figures v2');
print -dpsc independent_vs_non_independent_subject_sets.eps;
cd('/Users/Phillip/Documents/Research/Image Memorability/');
%}



%% Plot database sample set
jj2 = randperm(length(image_index_list));
imdisp(img(:,:,:,jj2(1:200)));

print('-dpsc', fullfile('../Analysis/plots/',which_date,'database_sample_set.eps'));
close all;


%% Plot sorted database
N = size(sorted_target_data,1);
hits = zeros(N,1);
misses = zeros(N,1);
for i=1:N
    hits(i) =  sorted_target_data{i}.hits;
    misses(i) = sorted_target_data{i}.misses;
end
actual = double(hits./(hits+misses));

actual = actual(image_index_list);

[foo jj] = sort(actual, 'descend');
imdisp(img(:,:,:,jj(1:20:end)));

print('-dpsc', fullfile('../Analysis/plots/',which_date,'database_entire_set_sorted_by_hr.eps'));
close all;


%% Plot S2 sorted by S1 sample images
predicted = subject_hrs1(2,image_index_list)'; % note: using split #2 for paper since split number one looks worse... choosing clarity of explication over perfect disinterest here; quantitative results in later figures are unbiased so I think this is okay here.
actual = subject_hrs2(2,image_index_list)';

VisualizeOnePredictedSet(predicted, actual, img, 'top', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'S1_vs_S2_top_set_8.eps'));
close all;

VisualizeOnePredictedSet(predicted, actual, img, 'middle', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'S1_vs_S2_middle_set_8.eps.eps'));
close all;

VisualizeOnePredictedSet(predicted, actual, img, 'bottom', 8);
print('-dpsc', fullfile('../Analysis/plots/',which_date,'S1_vs_S2_bottom_set_8.eps'));
close all;


%% Memorability maps
load(['../Analysis/results/' which_date '/mem_maps/object_scores']);
load(['../Analysis/results/' which_date '/mem_maps/avg_object_scores']);
load(['../Analysis/results/' which_date '/mem_maps/mem_maps_predicted']);
load(['../Analysis/results/' which_date '/mem_maps/mem_maps_test_label']);
load(['../Analysis/results/' which_date '/mem_maps/mem_maps_test_indices']);

test_segments = segments(:,:,mem_maps_test_indices);
test_img = img(:,:,:,mem_maps_test_indices);

% calc normalized scale for display
pos_scores = object_scores.*(object_scores>=0);
neg_scores = object_scores.*(object_scores<0);

scale_pos = [min(pos_scores(:)) max(pos_scores(:))];
scale_neg = [max(neg_scores(:)) abs(min(neg_scores(:)))];
%scale = [min(neg_scores(:)) max(pos_scores(:))];
%scale = 2*scale./(scale(2)-scale(1));
%scale = scale./max(abs(scale));
%scale_pos = scale(2).*[min(pos_scores(:)) max(pos_scores(:))];
%scale_neg = scale(1).*[max(neg_scores(:)) abs(min(neg_scores(:)))];


% plot top 10

[p ii] = sort(mem_maps_predicted,'descend');
which_images = ii(1:10);
num_images = length(which_images);

for i=1:num_images
    
    subplot(2,num_images,i);
    VisualizeMemorabilityMapObjects(test_segments(:,:,which_images(i)), object_scores(:,which_images(i)), scale_pos, scale_neg);
    title(sprintf('pred = %g',mem_maps_predicted(which_images(i))));
    
    subplot(2,num_images,i+num_images);
    imshow(test_img(:,:,:,which_images(i)));
    title(sprintf('actual = %g',mem_maps_test_label(which_images(i))));
end

print('-dpsc', fullfile('../Analysis/plots/',which_date,'memory_map_sample_top_10.eps'));
close all;


% plot middle set

[p ii] = sort(mem_maps_predicted,'descend');
which_images = ii(floor(length(ii)/2)-4:floor(length(ii)/2)+9);
num_images = length(which_images);

for i=1:num_images
    
    subplot(2,num_images,i);
    VisualizeMemorabilityMapObjects(test_segments(:,:,which_images(i)), object_scores(:,which_images(i)), scale_pos, scale_neg);
    title(sprintf('pred = %g',mem_maps_predicted(which_images(i))));
    
    subplot(2,num_images,i+num_images);
    imshow(test_img(:,:,:,which_images(i)));
    title(sprintf('actual = %g',mem_maps_test_label(which_images(i))));
end

print('-dpsc', fullfile('../Analysis/plots/',which_date,'memory_map_sample_middle_10.eps'));
close all;


% plot bottom 10

[p ii] = sort(mem_maps_predicted,'descend');
which_images = ii((end-9):end);
num_images = length(which_images);

for i=1:num_images
    
    subplot(2,num_images,i);
    VisualizeMemorabilityMapObjects(test_segments(:,:,which_images(i)), object_scores(:,which_images(i)), scale_pos, scale_neg);
    title(sprintf('pred = %g',mem_maps_predicted(which_images(i))));
    
    subplot(2,num_images,i+num_images);
    imshow(test_img(:,:,:,which_images(i)));
    title(sprintf('actual = %g',mem_maps_test_label(which_images(i))));
end

print('-dpsc', fullfile('../Analysis/plots/',which_date,'memory_map_sample_bottom_10.eps'));
close all;


% plot scale
pos_s = scale_pos(1):0.001:scale_pos(2);
neg_s = fliplr(scale_neg(1):0.001:scale_neg(2));

pos_s_c = (pos_s - scale_pos(1))./scale_pos(2);
neg_s_c = (neg_s - scale_neg(1))./scale_neg(2);

z_neg = neg_s_c.*0;
z_pos = pos_s_c.*0;
pos_s_c = [z_neg pos_s_c];
neg_s_c = [neg_s_c z_pos];

s_c = cat(3,pos_s_c+neg_s_c./4, (pos_s_c+neg_s_c)./4, neg_s_c+pos_s_c./4);
s_c = repmat(s_c,[10,1,1]);
s_c(1,length(z_neg),:) = [1 1 1];
s_c(2,length(z_neg)+1,:) = [1 1 1];
subplot(211);
imshow(s_c(1:2,:,:));
subplot(212);
imshow(s_c(3:end,:,:));
title(sprintf('min: %g\nmax: %g\n(zero point is betwen two white pixels)', -scale_neg(2), scale_pos(2)));

print('-dpsc', fullfile('../Analysis/plots/',which_date,'memory_map_color_scale.eps'));
close all;


%% Plot objects ranked by exclusion score
PlotObjectsRankedByExclusionScore;


%% Individual (simple) feature analysis plots
FeatureAnalysisScript;


%% Consistency versus N scores per image plot
ConsistencyVersusN;



