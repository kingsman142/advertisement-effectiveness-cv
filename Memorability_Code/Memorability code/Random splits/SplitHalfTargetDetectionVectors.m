%% Split target detection vectors into training and test hrs

function [train_target_detection_vectors test_target_detection_vectors] = SplitHalfTargetDetectionVectors(target_detection_vectors)
    
    n_subjects = size(target_detection_vectors,1); % each row corresponds to target detection vector for one subject
    
    [train_indices test_indices] = SplitTrainTestIndices(n_subjects);
    
    train_target_detection_vectors = target_detection_vectors(train_indices,:);
    test_target_detection_vectors = target_detection_vectors(test_indices,:); 
end