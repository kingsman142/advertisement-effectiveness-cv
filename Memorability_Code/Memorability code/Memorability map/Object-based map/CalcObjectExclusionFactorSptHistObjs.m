%% Calc object exclusion factor

function [exclusion_factor] = CalcObjectExclusionFactorSptHistObjs(train_features, test_features, prediction1, kernel_param, model, i_out)
    
    n_obj_classes = length(test_features)/5; % test features is 2-level sptHist of object class data
    
    % remove feature at index i_out
    replace_with = 0;
    test_features(i_out) = replace_with;
    test_features(i_out+n_obj_classes) = replace_with;
    test_features(i_out+(2*n_obj_classes)) = replace_with;
    test_features(i_out+(3*n_obj_classes)) = replace_with;
    test_features(i_out+(4*n_obj_classes)) = replace_with;
    
    % make new prediction without feature at index i_out
    K_test = kernel(test_features, train_features, kernel_param);
    K_test = double([(1:size(test_features,1))', K_test]);
    
    [prediction2, accuracy, dec_value] = svmpredict(1, K_test, model);
    
    exclusion_factor = prediction1-prediction2;
    
end