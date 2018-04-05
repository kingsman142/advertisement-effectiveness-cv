%% Visualize objects-based memorability map script


%% setup rand seed
%RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));

Predictor_type = 'r'; % c = SVC, r = SVR, w = CWM
classification_type = 'm'; % m = about median, t = top set, b = bottom set
classification_set_size = 100; % used for top set and bottom set classification
resampling = ''; % 1 = Resample training set to uniform distribution, 2 = Resample test set to uniform distribution
kernel_combination_type = 'p'; % s = sum, p = product
independent_subject_set = 't'; % t = test on hit rates from an independent subject set, f = test on hit rates from the same subject set as training
num_trials = 1; % number of test/training splits
num_grid_search_trials = 4;

do_save = 'f';

stat_features_to_use = ''; % g = GIST, s = SIFT, h = HOG, p = Pixels
obj_features_to_use = 's'; % c = Counts, p = Presence, a = Areas, s = Spatial histograms, n = marginal counts, m = marginal areas, b = marginal spatial histograms

Prediction_5; % generates param, model, test_features, predicted, test_label, train_indices, test_indices

if (length(feature_types) > 1)
    error('memorability maps can handle just 1 feature type!');
end
test_features = test_features.(feature_types{1}); % reformat feature matrices 
train_features = train_features.(feature_types{1});
        

%% calc object scores

%which_images = [800 900 1000 1100];
%[p i] = sort(predicted,'descend');
%which_images = i;%(1:10);%1:150:end);
%which_images = 1:size(test_data,1);

num_images = length(predicted);

object_scores = nan(size(Areas,1)+1,num_images);

n_obj_classes = length(test_features)/5; % test features is 2-level sptHist of object class data
for i=1:num_images
    
    curr_test_features = test_features(i,:);
    
    which_objects = find(curr_test_features(1:n_obj_classes)~=0);
    
    for j=1:length(which_objects)
        object_scores(which_objects(j)+1,i) = CalcObjectExclusionFactorSptHistObjs(train_features, curr_test_features, predicted(i), param, model, int16(which_objects(j)));
    end
    
end


%%
min_area = 4000;
object_presences = (full(Areas(:,test_indices)))>min_area;
object_presences = [zeros(1,size(object_presences,2)); object_presences]; % first position is for 0/'none' label (refers to unlabeled segments of image)


%% calc rank object scores
rank_object_scores = nan(size(Areas,1)+1,num_images);

[foo i1] = sort(predicted,'descend');
[foo i2] = sort(i1,'ascend');

for i=1:num_images
    
    which_objects = find(~isnan(object_scores(:,i)));
    
    for j=1:length(which_objects)
        
        a = which_objects(j);
        score_for_object_a_in_image_i = object_scores(a,i);
        
        predicted2 = predicted;
        predicted2(i) = predicted2(i) + score_for_object_a_in_image_i; % set of predictions when object a is omitted from image i
        
        [foo j1] = sort(predicted2,'descend');
        [foo j2] = sort(j1,'ascend');
        
        new_rank_for_image_i = j2(i);
        old_rank_for_image_i = i2(i);
        
        rank_change_for_image_i = old_rank_for_image_i-new_rank_for_image_i;
        
        rank_object_scores(a,i) = rank_change_for_image_i;
    end
end


%% calc average object scores aggregated over all images containing each
%% object

avg_object_scores = nan(size(object_scores,1),1); % nans are set to zero for sorting below (zero since nan means we have no data so should go with prior guess of 0 effect)
avg_rank_object_scores = nan(size(rank_object_scores,1),1);
for i=1:length(avg_object_scores)
    curr_object_scores = object_scores(i,:);
    curr_rank_object_scores = rank_object_scores(i,:);
    %which_scores = find(~isnan(curr_object_scores));
    which_scores = find(object_presences(i,:));
    if (~isempty(which_scores))
        avg_object_scores(i) = mean(curr_object_scores(which_scores));
        avg_rank_object_scores(i) = mean(curr_rank_object_scores(which_scores));
    end
end

avg_object_scores(isnan(avg_object_scores)) = 0; % remove nans
avg_rank_object_scores(isnan(avg_rank_object_scores)) = 0;
[s i] = sort([avg_object_scores(2:end)],'descend');
sorted_objectnames = cell(size(i));
for j=1:length(i)
    sorted_objectnames{j} = objectnames{i(j)};
end
sorted_objectnames(1:20)
sorted_objectnames((end-20):end)


%% save scores
date1 = date;

mkdir(['../Analysis/results/' date1]);
mkdir(['../Analysis/results/' date1 '/mem_maps']);

save(['../Analysis/results/' date1 '/mem_maps/object_scores'], 'object_scores');
save(['../Analysis/results/' date1 '/mem_maps/rank_object_scores'], 'rank_object_scores');
save(['../Analysis/results/' date1 '/mem_maps/avg_object_scores'], 'avg_object_scores');
save(['../Analysis/results/' date1 '/mem_maps/avg_rank_object_scores'], 'avg_rank_object_scores');
mem_maps_predicted = predicted;
save(['../Analysis/results/' date1 '/mem_maps/mem_maps_predicted'], 'mem_maps_predicted');
mem_maps_test_label = test_label;
save(['../Analysis/results/' date1 '/mem_maps/mem_maps_test_label'], 'mem_maps_test_label');
mem_maps_train_indices = train_indices;
save(['../Analysis/results/' date1 '/mem_maps/mem_maps_train_indices'], 'mem_maps_train_indices');
mem_maps_test_indices = test_indices;
save(['../Analysis/results/' date1 '/mem_maps/mem_maps_test_indices'], 'mem_maps_test_indices');

