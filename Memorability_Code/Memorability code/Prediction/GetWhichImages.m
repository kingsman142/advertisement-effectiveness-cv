%% returns indices of images in specified bins (here the specified bins
%% are, for the time being, simply hard-coded)

function [which_images] = GetWhichImages(hits, misses)

    N = length(hits);
    
    %% find bin boundaries
    finest_grain = 8;
    
    % randomly remove enough images such that all bins have the same size
    data = [hits misses (hits./(hits+misses)) randperm(N)'];
    %{
    new_N = floor(N/finest_grain)*finest_grain;
    data = sortrows(data,4); % randomize order
    data = data(1:new_N,:); % truncate
    data = data(:,1:3);
    hits = data(:,1);
    misses = data(:,2);
    N = new_N;
    %}
    
    [sorted_data i] = sortrows(data,3);
    bin_size = N/finest_grain;
    bin_bounds = zeros(finest_grain,2);
    bin_bounds(:,1) = sorted_data(1:bin_size:N,3);
    bin_bounds(:,2) = sorted_data(bin_size:bin_size:N,3);
    
    % select images in specified bins
    hrs = hits./(hits+misses);
    which_images = [find(hrs<bin_bounds(1,2))' find(hrs>bin_bounds(end,1))'];

end