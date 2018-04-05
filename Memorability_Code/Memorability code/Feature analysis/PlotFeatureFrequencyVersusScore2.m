%% Plot feature frequency versus score

function [] = PlotFrequencyVersusScore2(frequencies, scores)

    scatter(log(frequencies), scores);
    lsline;
    xlabel('Log object frequency');
    ylabel('Object score');
    
end