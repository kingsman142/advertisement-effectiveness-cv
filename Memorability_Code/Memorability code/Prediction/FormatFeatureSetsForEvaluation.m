%% FormatFeatureSetsForEvaluation

feature_sets = [];
feature_sets_names = [];
clearvars new_feature_set;
for i=1:length(stat_feature_sets)
    new_feature_set.stat_features_to_use = stat_feature_sets{i};
    new_feature_set.obj_features_to_use = '';
    feature_sets = [feature_sets new_feature_set];
    feature_sets_names = [feature_sets_names stat_feature_sets_names(i)];
end
for i=1:length(obj_feature_sets)
    new_feature_set.stat_features_to_use = '';
    new_feature_set.obj_features_to_use = obj_feature_sets{i};
    feature_sets = [feature_sets new_feature_set];
    feature_sets_names = [feature_sets_names obj_feature_sets_names(i)];
end
for i=1:length(combo_stat_feature_sets)
    new_feature_set.stat_features_to_use = combo_stat_feature_sets{i};
    new_feature_set.obj_features_to_use = combo_obj_feature_sets{i};
    feature_sets = [feature_sets new_feature_set];
    feature_sets_names = [feature_sets_names combo_feature_sets_names(i)];
end

n_feature_sets = length(feature_sets);

if (length(kernel_combination_type) == 1)
    kernel_combination_type = repmat(kernel_combination_type, [n_feature_sets,1]);
end