%% Generate many random splits

N_splits = 25;
N_images = 2222; % here we are ignoring the texture images above index 2222


%% build subject half splits

target_detection_vectors = GetTargetDetectionVectors(subj_data, 1.0);
target_detection_vectors = target_detection_vectors(:,1:N_images); % ignore texture images

subject_hits1 = zeros(N_splits, N_images);
subject_hits2 = zeros(N_splits, N_images);
subject_misses1 = zeros(N_splits, N_images);
subject_misses2 = zeros(N_splits, N_images);
T1_train_subject_hits = zeros(N_splits, N_images);
T1_train_subject_misses = zeros(N_splits, N_images);
T2_train_subject_hits = zeros(N_splits, N_images);
T2_train_subject_misses = zeros(N_splits, N_images);
for i=1:N_splits
    [x y] = SplitHalfTargetDetectionVectors(target_detection_vectors); % split half
    
    subject_hits1(i,:) = sum(x==1,1);
    subject_misses1(i,:) = sum(x==-1,1);
    subject_hits2(i,:) = sum(y==1,1);
    subject_misses2(i,:) = sum(y==-1,1);
    
    [x y] = SplitHalfTargetDetectionVectors(x); % half again (gives 2 quarters)
    
    T1_train_subject_hits(i,:) = sum(x==1,1);
    T1_train_subject_misses(i,:) = sum(x==-1,1);
    T2_train_subject_hits(i,:) = sum(y==1,1);
    T2_train_subject_misses(i,:) = sum(y==-1,1);
end

subject_hrs1 = subject_hits1./(subject_hits1+subject_misses1);
subject_hrs2 = subject_hits2./(subject_hits2+subject_misses2);
T1_train_subject_hrs = T1_train_subject_hits./(T1_train_subject_hits+T1_train_subject_misses);
T2_train_subject_hrs = T2_train_subject_hits./(T2_train_subject_hits+T2_train_subject_misses);


save('../Data/Random splits/subject_hrs1','subject_hrs1');
save('../Data/Random splits/subject_hrs2','subject_hrs2');
save('../Data/Random splits/T1_train_subject_hrs','T1_train_subject_hrs');
save('../Data/Random splits/T2_train_subject_hrs','T2_train_subject_hrs');


%% image splits

% get just the images that have >= 20 subjects (of the N_images we are
% considering)

image_index_list = [];
for i=1:N_images
    if (sorted_target_data{i}.hits + sorted_target_data{i}.misses >= 20)
        image_index_list = [image_index_list i];
    end
end

N_images = length(image_index_list);


% half splits

[x, y] = SplitTrainTestIndices(N_images);
image_train_indices = zeros(N_splits, size(x,2));
image_test_indices = zeros(N_splits, size(y,2));
for i=1:N_splits
    [x, y] = SplitTrainTestIndices(N_images); 
    image_train_indices(i,:) = image_index_list(x);
    image_test_indices(i,:) = image_index_list(y);
end

save('../Data/Random splits/image_train_indices','image_train_indices');
save('../Data/Random splits/image_test_indices','image_test_indices');


% training set splits

N_train_images = size(image_train_indices,2);

[x y] = SplitTrainTestIndices(N_train_images);
T1_train_indices = zeros(N_splits, N_splits, size(x,2));
T2_train_indices = zeros(N_splits, N_splits, size(y,2));
for i=1:N_splits
    for j=1:N_splits
        [x, y] = SplitTrainTestIndices(N_train_images); 
        T1_train_indices(i,j,:) = image_train_indices(i,x);
        T2_train_indices(i,j,:) = image_train_indices(i,y);
    end
end

save('../Data/Random splits/T1_train_indices','T1_train_indices');
save('../Data/Random splits/T2_train_indices','T2_train_indices');




