%% Gets the hsv mean color of an image

function [mean_hsv] = GetMeanColorHSV(im)

    %addpath('/Users/Phillip/Documents/MATLAB/toolboxes/CircStat2010d');
    
    rs = im(:,:,1)./255;
    gs = im(:,:,2)./255;
    bs = im(:,:,3)./255;
    mean_r = mean(rs(:));
    mean_g = mean(gs(:));
    mean_b = mean(bs(:));
    
    mean_hsv = rgb2hsv([mean_r, mean_g, mean_b]);

end