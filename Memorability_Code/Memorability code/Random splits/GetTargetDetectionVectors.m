%% Get target detection vectors for all subjects

function [target_detection_vectors] = GetTargetDetectionVectors(subj_data, subj_proportion)

    target_detection_vectors = [];
    j = 1;
    N = floor(length(subj_data)*subj_proportion);
    p1 = randperm(N); % randomize order of subj_data so we look at a random subj_proportion subset of subj_data
    for i=1:N
        if (~isempty(subj_data{p1(i)}))
            if (sum(abs(subj_data{p1(i)}.target_detection_vector)) ~= 0) % only considering subjects who actually contributed target detection data
                target_detection_vectors(j,:) = subj_data{p1(i)}.target_detection_vector;
                j = j+1;
            end
        end
    end
    
end