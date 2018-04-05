%% Create pixel (color) histograms


%% get pixel histograms
bins = 0:0.05:1;
N = size(sorted_target_data,1);
pixel_histograms = zeros(N,length(bins),3);
for i=1:N
    im = double(imread(fullfile(HOMEMEMORYIMAGES,sorted_target_data{i}.filepath)));
    rs = im(:,:,1)./255;
    gs = im(:,:,2)./255;
    bs = im(:,:,3)./255;
    pixel_histograms(i,:,1) = hist(rs(:),bins);
    pixel_histograms(i,:,2) = hist(gs(:),bins);
    pixel_histograms(i,:,3) = hist(bs(:),bins);
    i
end