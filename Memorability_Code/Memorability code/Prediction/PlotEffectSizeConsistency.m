%% Plot effect size consistency for VSS poster


%%
clearvars feature_sets feature_sets_names;


%% options:
Predictor_type = 'r'; % c = SVC, r = SVR, w = CWM
classification_type = 'm'; % m = about median, t = top set, b = bottom set
classification_set_size = 100; % used for top set and bottom set classification
resampling = ''; % 1 = Resample training set to uniform distribution, 2 = Resample test set to uniform distribution
%kernel_combination_type = 'p'; % s = sum, p = product
independent_subject_set = 't'; % t = test on hit rates from an independent subject set, f = test on hit rates from the same subject set as training
num_trials = 25;


%{
stat_feature_sets = {};%{'p' 'g' 's' 'h' 'gshp'};  % g = GIST, s = SIFT, h = HOG, p = Pixels
obj_feature_sets = {'n' 'm' 'b'};%{'p' 'c' 'a' 's'};  % c = Counts, p = Presence, a = Areas, s = Spatial histograms
stat_feature_sets_names = {};%{'Pixels' 'GIST' 'SIFT' 'HOG' 'GIST, SIFT, HOG, Pixels'};
obj_feature_sets_names = {'Marginalized obj counts' 'Marginalized obj areas' 'Marginalized obj spatial histograms'};%{'Obj presences' 'Obj counts' 'Obj areas' 'Obj spatial histograms'};
plot_subjects = 1;
plot_actual = 1;
plot_chance = 1;
use_all_images_human_consistency = 'f';

y_range = [0.64 1.0];
%}

all_predicteds = [];
all_test_labels = [];
h = [];

FormatFeatureSetsForEvaluation;

%n_stat_feature_sets = length(stat_feature_sets);
%n_obj_feature_sets = length(obj_feature_sets);
%hues = ((1:n_stat_feature_sets)./n_stat_feature_sets)/3 + 0.66;
%hues = [hues ((1:n_obj_feature_sets)./n_obj_feature_sets)/3 + 0.33];
hues = ((1:n_feature_sets)./n_feature_sets);
hues = [hues 0.25 0.5 0.75];

figure;
hold on;

for k=1:(n_feature_sets+plot_subjects+plot_actual+plot_chance)
    
    LoadPredictionData;
    
    
    %% quantify results

    N = size(all_predicteds,1);

    r = zeros(N,1);
    rank_corr = zeros(N,1);
    mse = zeros(N,1);

    for i=1:N
        test_label = all_test_labels(i,:)';
        predicted = all_predicteds(i,:)';
        r(i) = corr(test_label, predicted, 'type', 'Pearson');
        rank_corr(i) = corr(test_label, predicted, 'type', 'Spearman');
        mse(i) = mean((predicted-test_label).^2);
    end
    
    avg_r = mean(r);
    avg_rank_corr = mean(rank_corr);
    avg_mse = mean(mse);
    
    if (k<=n_feature_sets)
        feature_sets_names{k} = [feature_sets_names{k} sprintf('\nr = %4.2f, rank corr = %4.2f, mse = %4.2f', avg_r, avg_rank_corr, avg_mse)];
    elseif (k==n_feature_sets+1)
        subj_preds_name = sprintf('Subjects\nr = %4.2f, rank corr = %4.2f, mse = %4.2f', avg_r, avg_rank_corr, avg_mse);
    elseif (k==n_feature_sets+2)
        actual_pred_actual_name = sprintf('Actual\nr = %4.2f, rank corr = %4.2f, mse = %4.2f', avg_r, avg_rank_corr, avg_mse);
    else
        chance_pred_name = sprintf('Chance\nr = %4.2f, rank corr = %4.2f, mse = %4.2f', avg_r, avg_rank_corr, avg_mse);
    end
        
    
    
    %% plot results

    all_plots = [];
    for i=1:N
        test_label = all_test_labels(i,:)';
        predicted = all_predicteds(i,:)';
        [foo,jj] = sort(predicted,'descend');
        all_plots(i,:) = conv(test_label(jj),ones(25,1)./25,'valid');
        %all_plots(i,:) = cumsum(test_label(jj))./(1:length(test_label))';
        %plot(all_plots(i,:), 'b');
    end
    
    avg_plot = mean(all_plots,1);
    
    %e = std(all_plots,1)/sqrt(size(all_plots,1));
    
    e = Get80PercentIntervals2(all_plots);
    e_display_indices = 100:300:length(avg_plot);
    if (display_error_bars == 't')
        errorbar(e_display_indices, avg_plot(e_display_indices), abs(e(e_display_indices,1)), e(e_display_indices,2), '.', 'Color', hsv2rgb([hues(k),0.5,0.8]));
    end
    
    h(k) = plot(avg_plot, 'Color', hsv2rgb([hues(k),1,0.8]));
    set(gca,'XLim',[12, size(all_plots,2)-12]);
    set(gca,'YLim',y_range);
    
    

    %%
    %PlotScatterPredictedVsActual(all_test_labels(1,:), all_predicteds(1,:));


end


if (k==n_feature_sets+3)
    legend(h, [feature_sets_names {subj_preds_name} {actual_pred_actual_name} {chance_pred_name}]);
elseif (k==n_feature_sets+1)
    legend(h, [feature_sets_names {subj_preds_name}]);
else
    legend(h, feature_sets_names);
end 
xlabel('N');
ylabel('Average actual hit rate on images 1 through N');
if (use_all_images_human_consistency == 't')
    title(sprintf('Cumulative average over actual hit rates sorted by predicted hit rates\nAveraged over %d random image set half split trials', N));
else
    title(sprintf('Cumulative average over actual hit rates sorted by predicted hit rates\nAveraged over %d random 4-way-split trials', N));
end


