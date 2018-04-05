%% Plot feature frequency versus score
% feature_matrix is NxD where N is number of images (or other things
% described by features) and D is dimension of features
% feature_scores is Dx1
% feature_matrix must be binary expressing the presence or absence of that
% feature in an image
% currently modified speicifically for plotting object presence frequency
% versus score

function [] = PlotFeatureFrequencyVersusScore(feature_matrix, feature_scores)

    scatter(log(sum(feature_matrix,1)), feature_scores);
    lsline;
    XLabel('Log object (presence) frequency');
    YLabel('Object (presence) score');
    
end