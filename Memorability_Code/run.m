%% CVPR paper script

%%
clear all;

%% set random seed (set this just once at start of experimental run)
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));

%%
addpath(genpath(pwd));

%% load
load('../Data/Experiment data/sorted_target_data.mat'); % experiment data for each image
load('../Data/Experiment data/subj_data.mat'); % experiment data for each subject

% load images, annotations, and precomputed descriptors
if (exist('../Data/Image data/target_features.mat','file'))
    load('../Data/Image data/target_features.mat');
    load('../Data/Image data/target_images.mat');
else
    ComputeImgFeats;
end

% load random splits
if (exist('../Data/Random splits/image_train_indices.mat','file'))
    load('../Data/Random splits/image_train_indices.mat');   % train images
    load('../Data/Random splits/image_test_indices.mat');    % test images
    load('../Data/Random splits/subject_hrs1');              % train subjects
    load('../Data/Random splits/subject_hrs2');              % test subjects
    load('../Data/Random splits/T1_train_indices.mat');      % grid search train images (random halves of train images)
    load('../Data/Random splits/T2_train_indices.mat');      % grid search test images
    load('../Data/Random splits/T1_train_subject_hrs.mat');  % grid search train subjects (random halves of train subjects)
    load('../Data/Random splits/T2_train_subject_hrs.mat');  % grid search test subjects
else
    GenerateRandomSplits3;
end


%% Basic stats
CalculateBasicStatsForCVPR;


%% Run SVRs
RunAllCVPRRegressions;


%% Calc object exclusion scores
CreateMemorabilityMapObjectsScript3;


%% Create figures and tables
which_date = date; % create figures for results created on which date?
CreateCVPRPaperFigures;
CreatePredictionsTables; % creates 'precision-recall' data tables


