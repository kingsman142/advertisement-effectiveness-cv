%% SplitHalfSubjectHitsMisses

function [subject_hits1, subject_misses1, subject_hits2, subject_misses2] = SplitHalfSubjectHitsMissesProportionSubjects(subj_data, subj_proportion)

    target_detection_vectors = [];
    j = 1;
    p1 = randperm(length(subj_data)); % randomize order of subj_data so we look at a random subj_proportion subset of subj_data
    N = floor(length(subj_data)*subj_proportion);
    for i=1:N
        if (~isempty(subj_data{p1(i)}))
            if (sum(abs(subj_data{p1(i)}.target_detection_vector)) ~= 0)
                target_detection_vectors(j,:) = subj_data{p1(i)}.target_detection_vector;
                j = j+1;
            end
        end
    end
    
    n_subjects = size(target_detection_vectors,1);
    p = randperm(n_subjects);
    half_subjects = ceil(n_subjects/2);
    v1 = target_detection_vectors(p(1:half_subjects),:);
    v2 = target_detection_vectors(p((half_subjects+1):end),:);
    
    subject_hits1 = sum(v1==1,1);
    subject_misses1 = sum(v1==-1,1);
    subject_hits2 = sum(v2==1,1);
    subject_misses2 = sum(v2==-1,1);
    
end