%% Reload splits
load('../Data/Random splits/image_train_indices.mat');   % train images
load('../Data/Random splits/image_test_indices.mat');    % test images
load('../Data/Random splits/subject_hrs1');              % train subjects
load('../Data/Random splits/subject_hrs2');              % test subjects
load('../Data/Random splits/T1_train_indices.mat');      % grid search train images (random halves of train images)
load('../Data/Random splits/T2_train_indices.mat');      % grid search test images
load('../Data/Random splits/T1_train_subject_hrs.mat');  % grid search train subjects (random halves of train subjects)
load('../Data/Random splits/T2_train_subject_hrs.mat');  % grid search test subjects


%% Human consistency analysis (evaluate on all images)
Predictor_type = 'r'; % c = SVC, r = SVR, w = CWM
classification_type = 'm'; % m = about median, t = top set, b = bottom set
classification_set_size = 100; % used for top set and bottom set classification
resampling = ''; % 1 = Resample training set to uniform distribution, 2 = Resample test set to uniform distribution
kernel_combination_type = 'p'; % s = sum, p = product
independent_subject_set = 't'; % t = test on hit rates from an independent subject set, f = test on hit rates from the same subject set as training
num_trials = 25;


stat_feature_sets = {};  % g = GIST, s = SIFT, h = HOG, p = Pixels
obj_feature_sets = {};  % c = Counts, p = Presence, a = Areas, s = Spatial histograms
stat_feature_sets_names = {};
obj_feature_sets_names = {};
n_stat_feature_sets = length(stat_feature_sets);
n_obj_feature_sets = length(obj_feature_sets);
use_all_images_human_consistency = 't';

CreatePredictionsTable;


%% Regressions analysis (evaluate just on test set images)
Predictor_type = 'r'; % c = SVC, r = SVR, w = CWM
classification_type = 'm'; % m = about median, t = top set, b = bottom set
classification_set_size = 100; % used for top set and bottom set classification
resampling = ''; % 1 = Resample training set to uniform distribution, 2 = Resample test set to uniform distribution
independent_subject_set = 't'; % t = test on hit rates from an independent subject set, f = test on hit rates from the same subject set as training
num_trials = 25;

stat_feature_sets = {'p' 'g' 's' 'h' 'i' 'gsh' 'gshp' 'gship'};
obj_feature_sets = {'p' 'c' 'a' 's' 'n' 'm' 'b' 'e' 'se'};
stat_feature_sets_names = {'Pixels' 'GIST' 'SIFT' 'HOG' 'SSIM' 'GIST*SIFT*HOG' 'GIST*SIFT*HOG*Pixels' 'GIST*SIFT*HOG*SSIM*Pixels'};
obj_feature_sets_names = {'Obj presences' 'Obj counts' 'Obj areas' 'Obj spatial histograms' 'Marginalized obj counts' 'Marginalized obj areas' 'Marginalized obj spatial histograms' 'Scene categories' 'Obj spatial histograms and Scene categories'};
combo_stat_feature_sets = {'gship'}; combo_obj_feature_sets = {'se'}; % combo features are combined into one big feature vector
combo_feature_sets_names = {'GIST+SIFT+HOG+SSIM+Pixels and Obj spatial histograms and Scene categories'};

kernel_combination_type = ['p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'p';'s';'s']; % s = sum, p = product, m = mixed

use_all_images_human_consistency = 'f';

CreatePredictionsTable;

