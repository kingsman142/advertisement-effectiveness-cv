%% Get split-half correlations averaged over many splits

function [avg_r, avg_rho, avg_mse, r, rho, mse] = GetConsistencyStats(subject_hrs1, subject_hrs2, use_all_images_human_consistency, image_test_indices)

    num_trials = 25;

    for trial_i=1:num_trials
        train_labels_all = subject_hrs1{trial_i}';
        test_labels_all = subject_hrs2{trial_i}';

        if (use_all_images_human_consistency == 't')
            train_label = train_labels_all(1:2222);
            test_label = test_labels_all(1:2222);
        else
            
            train_indices = find(image_test_indices);%(trial_i,:); % not a typo, train is set to test set here
            test_indices = find(image_test_indices);%(trial_i,:);

            train_label = train_labels_all(train_indices);
            test_label = test_labels_all(test_indices);
        end
        
        % remove nans (occur in cases of not enough data for certain images)
        mask = ~isnan(train_label) & ~isnan(test_label);
        train_label = train_label(mask);
        test_label = test_label(mask);

        all_predicteds{trial_i} = train_label;
        all_test_labels{trial_i} = test_label;
    end

    
    %% calc stats
    
    N = length(all_predicteds);

    r = zeros(N,1);
    rho = zeros(N,1);
    mse = zeros(N,1);

    for i=1:N
        test_label = all_test_labels{i};
        predicted = all_predicteds{i};
        r(i) = corr(test_label, predicted, 'type', 'Pearson');
        rho(i) = corr(test_label, predicted, 'type', 'Spearman');
        mse(i) = mean((predicted-test_label).^2);
    end
    
    avg_r = mean(r);
    avg_rho = mean(rho);
    avg_mse = mean(mse);
    
end
  


