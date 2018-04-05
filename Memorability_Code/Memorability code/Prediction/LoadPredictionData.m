%% LoadPredictionData

if (k<=n_feature_sets)
    stat_features_to_use = feature_sets(k).stat_features_to_use;
    obj_features_to_use = feature_sets(k).obj_features_to_use;
    
elseif (k==n_feature_sets+1) % compare with subjects predicting subjects
    %all_predicteds = subject_hrs1(1:num_trials,:)';
    %all_test_labels = subject_hrs2(1:num_trials,:)';

    for trial_i=1:num_trials
        train_labels_all = subject_hrs1(trial_i,:)';
        test_labels_all = subject_hrs2(trial_i,:)';

        if (use_all_images_human_consistency == 't')
            train_label = train_labels_all(1:2222);
            test_label = test_labels_all(1:2222);
        else
            train_indices = image_test_indices(trial_i,:); % not a typo, train is set to test set here
            test_indices = image_test_indices(trial_i,:);

            train_label = train_labels_all(train_indices);
            test_label = test_labels_all(test_indices);
        end

        all_predicteds(trial_i,:) = train_label;
        all_test_labels(trial_i,:) = test_label;
    end

    %all_predicteds = all_predicteds(image_test_indices(1:num_trials,:));
    %all_test_labels = all_test_labels(image_test_indices(1:num_trials,:));
elseif (k==n_feature_sets+2) % compare with actual predicting actual

    for trial_i=1:num_trials
        test_labels_all = subject_hrs2(trial_i,:)';

        if (use_all_images_human_consistency == 't')
            test_label = test_labels_all(1:2222);
        else 
            test_indices = image_test_indices(trial_i,:);
            test_label = test_labels_all(test_indices);
        end

        all_predicteds(trial_i,:) = test_label;
        all_test_labels(trial_i,:) = test_label;
    end
elseif (k==n_feature_sets+3) % compare with chance predictions

     for trial_i=1:num_trials
        test_labels_all = subject_hrs2(trial_i,:)';

        if (use_all_images_human_consistency == 't')
            test_label = test_labels_all(1:2222);
        else 
            test_indices = image_test_indices(trial_i,:);
            test_label = test_labels_all(test_indices);
        end

        all_predicteds(trial_i,:) = rand(size(test_label));
        all_test_labels(trial_i,:) = test_label;
    end
end


if (k<=n_feature_sets)
    %% load predicteds and actuals

    dir_name = [Predictor_type '_' classification_type '_' classification_set_size '_' ...
                                                               '_' stat_features_to_use '_' obj_features_to_use '_' resampling ...
                                                                kernel_combination_type(k) '_' independent_subject_set];

    date1 = which_date;
    load(['../Analysis/results/' date1 '/' dir_name '/all_predicteds.mat']);
    load(['../Analysis/results/' date1 '/' dir_name '/all_test_labels.mat']);
end