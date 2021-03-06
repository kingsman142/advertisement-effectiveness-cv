%% Feature frequency versus score script

addpath('/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/Prediction/libsvm-mat-3.0-1');

cd '/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/Prediction/version_2010_aug_8/'
load datamem     % load annotations, stats, and precomputed descriptors (this file is generated by the script memory.m)
load('/Users/Phillip/Documents/Research/Image Memorability/Results/expt results/Matlab structured data/sorted_target_data.mat');
load('/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/Feature analysis/object_exclusion_scores.mat');
%load('/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/Feature analysis/object_expected_memorabilities.mat');
load('/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/test_indices.mat');
load('/Users/Phillip/Documents/Research/Image Memorability/MATLAB code/Analysis/train_indices.mat');


%% build feature vector for each image
features = Counts'>0;
which_images = test_indices;
features = features(which_images,:);
which_features = ~isnan(object_exclusion_scores(2:end));
features = features(:,which_features);
scores = object_exclusion_scores(2:end);
scores = scores(which_features);

PlotFeatureFrequencyVersusScore(features,scores);

