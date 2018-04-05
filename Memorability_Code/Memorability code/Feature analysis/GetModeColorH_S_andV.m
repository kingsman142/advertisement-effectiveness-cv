%% Gets the hsv mean color of an image

function [mode_hsv] = GetModeColorH_S_andV(im)

    %addpath('/Users/Phillip/Documents/MATLAB/toolboxes/CircStat2010d');
    
    rs = double(im(:,:,1))./255;
    gs = double(im(:,:,2))./255;
    bs = double(im(:,:,3))./255;
    
    rs = rs(:);
    gs = gs(:);
    bs = bs(:);
    
    [hs, ss, vs] = rgb2hsv([rs,gs,bs]);
    
    step_size = 0.05;
    edges = 0:step_size:1;
    [n_hs, bin_hs] = histc(hs(:),edges);
    [n_ss, bin_ss] = histc(ss(:),edges);
    [n_vs, bin_vs] = histc(vs(:),edges);
    
    mode_h = edges(mode(bin_hs))+step_size/2;
    mode_s = edges(mode(bin_ss))+step_size/2;
    mode_v = edges(mode(bin_vs))+step_size/2;
    
    mode_hsv = [mode_h, mode_s, mode_v];

end