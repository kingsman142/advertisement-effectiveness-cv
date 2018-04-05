%% Plot table of prediction results


%%
clearvars feature_sets feature_sets_names;

all_predicteds = [];
all_test_labels = [];

FormatFeatureSetsForEvaluation;

names = [feature_sets_names {'Humans'} {'Actual'} {'Chance'}];

avg_rank_corr = [];

mean_set_top_10 = [];
mean_set_top_20 = [];
mean_set_top_100 = [];
mean_set_top_500 = [];
mean_set_101_500 = [];
mean_set_101_989 = [];
mean_set_bottom_500 = [];
mean_set_bottom_100 = [];
mean_set_bottom_20 = [];
mean_set_bottom_10 = [];

for k=1:(n_feature_sets+3)
    

    LoadPredictionData;
    
    
    %% quantify results

    N = size(all_predicteds,1);

    set_top_10 = zeros(N,1);
    set_top_20 = zeros(N,1);
    set_top_100 = zeros(N,1);
    set_top_500 = zeros(N,1);
    set_101_500 = zeros(N,1);
    set_101_989 = zeros(N,1);
    set_bottom_500 = zeros(N,1);
    set_bottom_100 = zeros(N,1);
    set_bottom_20 = zeros(N,1);
    set_bottom_10 = zeros(N,1);
    
    rank_corr = zeros(N,1);
    
    for i=1:N
        test_label = all_test_labels(i,:)';
        predicted = all_predicteds(i,:)';
        
        rank_corr(i) = corr(test_label, predicted, 'type', 'Spearman');
        
        [foo,jj] = sort(predicted,'descend');
        
        set_top_10(i) = 100*mean(test_label(jj(1:10)));
        set_top_20(i) = 100*mean(test_label(jj(1:20)));
        set_top_100(i) = 100*mean(test_label(jj(1:100)));
        set_top_500(i) = 100*mean(test_label(jj(1:500)));
        set_101_500(i) = 100*mean(test_label(jj(101:500)));
        set_101_989(i) = 100*mean(test_label(jj(101:989)));
        set_bottom_500(i) = 100*mean(test_label(jj(end-499:end)));
        set_bottom_100(i) = 100*mean(test_label(jj(end-99:end)));
        set_bottom_20(i) = 100*mean(test_label(jj(end-19:end)));
        set_bottom_10(i) = 100*mean(test_label(jj(end-9:end)));
    end
    
    avg_rank_corr(k) = mean(rank_corr);
    
    mean_set_top_10(k) = mean(set_top_10);
    mean_set_top_20(k) = mean(set_top_20);
    mean_set_top_100(k) = mean(set_top_100);
    mean_set_top_500(k) = mean(set_top_500);
    mean_set_101_500(k) = mean(set_101_500);
    mean_set_101_989(k) = mean(set_101_989);
    mean_set_bottom_500(k) = mean(set_bottom_500);
    mean_set_bottom_100(k) = mean(set_bottom_100);
    mean_set_bottom_20(k) = mean(set_bottom_20);
    mean_set_bottom_10(k) = mean(set_bottom_10);

end

results_table = cat(1, names, num2cell(mean_set_top_10), num2cell(mean_set_top_20), ...
                        num2cell(mean_set_top_100), num2cell(mean_set_top_500), ...
                        num2cell(mean_set_101_500), num2cell(mean_set_101_989), num2cell(mean_set_bottom_500), ...
                        num2cell(mean_set_bottom_100), num2cell(mean_set_bottom_20), ...
                        num2cell(mean_set_bottom_10), num2cell(avg_rank_corr));
                    
cell2csv(fullfile('../Analysis/results/',which_date,strcat('/results_table_',use_all_images_human_consistency,'.csv')), results_table);
                    