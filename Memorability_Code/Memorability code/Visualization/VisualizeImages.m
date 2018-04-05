%% Visualize images
% im_indices is a vector of indices into ims in the order to display
% ims is a XxYx3xN matrix of N images each sized XxYx3
% range is the range of im_indices to display
% t is the text to display as the title

function [] = VisualizeImages(im_indices,ims,range,t)
    montage(ims(:,:,:,im_indices(range)));
    title(t)
end