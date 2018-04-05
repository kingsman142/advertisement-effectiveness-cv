% Demo

% load data
load datamem     % load annotations, stats, and precomputed descriptors (this file is generated by the script memory.m)

% DATA STRUCTURE
%    - Dmemory: labelme structure with the original annotations.
%         You can use this to access some basic information about each
%         image:
%         Dmemory(1).annotation.filename = image filename
%         Dmemory(1).annotation.folder = image folder
%         Dmemory(1).annotation.url = address of the image inside the
%                                     annotation tool online
%
%    - img: array[256 256 3 2400]  contains all the images
%
%    - segments: array [256 256 2400] segments for all images. Each pixel
%        contains an index to an object label. To know the label of pixel (i,j)
%        in image n, you can do: objectnames{segments(i,j,n)}
%
%    - objectnames: list of all the object classes
%
%    - Counts: array[Nobjects, 2400] Counts(i,n)=says how many instances of object
%        i are present in image n. The name of object i is objectnames{i}
%
%    - Areas: array[Nobjects,2400] Areas(i,n)=number of pixels occupied by
%        object i in image n
%
%    - gist:  array[2400, :] with gist vectors
%    - GISTparam: gist parameters
%
%    - VWsift: array[:,:,2400] sift visual words
%    - sptHistsift: histogram of sift visual words
%    - VWparamsift 
%
%    - VWhog: hog visual words
%    - sptHisthog: histogram of hog visual words
%    - VWparamhog

Nimages = size(img,4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo visualization
ndx = 1;   % choose an image (a number between 1 and 2400)

figure
subplot(231)
imshow(img(:,:,:,ndx))
title(Dmemory(ndx).annotation.filename, 'Interpreter', 'none')

subplot(232)
imshow(uint8(mod(segments(:,:,ndx), 255)))
colormap([0 0 0; hsv(255)])
title('segmentation')

subplot(233)
LMplot(Dmemory(ndx).annotation, img(:,:,:,ndx))
title('original annotation')

subplot(234)
showGist(gist(ndx,:), GISTparam)
title('gist')

subplot(235)
imagesc(VWsift(:,:,ndx))
axis('equal'); axis('tight'); axis('on')
title('sift')

subplot(236)
imagesc(VWhog(:,:,ndx))
axis('equal'); axis('tight'); axis('on')
title('hog')

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo queries using image features

ndx = 1  % choose an image (a number between 1 and 2400)

% 1) query with gist
gistQuery = gist(ndx,:);
distGist = sum((gist - repmat(gistQuery, [Nimages 1])).^2,2); % euclidean distance
[foo, j] = sort(distGist);

figure
montage(img(:,:,:,j(1:25)))
title('top-left=query image. Images sorted by GIST')


% 2) query with SIFT
siftQuery = sptHistsift(ndx,:);
distSift = 1-sum(min(sptHistsift, repmat(siftQuery, [Nimages 1])),2); % histogram intersection
[foo, j] = sort(distSift);

figure
montage(img(:,:,:,j(1:25)))
title('top-left=query image. Images sorted by SIFT')


% 3) query with HOG
hogQuery = sptHisthog(ndx,:);
distHog = 1-sum(min(sptHisthog, repmat(hogQuery, [Nimages 1])),2); % histogram intersection
[foo, j] = sort(distHog);

figure
montage(img(:,:,:,j(1:25)))
title('top-left=query image. Images sorted by HOG')

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Query with object labels
%
%  This query uses histograms over object labels to compute distance
%  between images. The problem of computing a distance between images using
%  ground truth annotations is still an open problem. If you explore
%  several queries you will see that sometimes the returned images are not
%  the ones that you would like to obtain. The right measure should weight the different
%  objects and also take into account their importance within the image. 

ndx=1; % choose an image (a number between 1 and 2400)

objQuery = sptHistObjects(ndx,:);
distObj = 1-sum(min(sptHistObjects, repmat(objQuery, [Nimages 1])),2); % histogram intersection
[foo, j] = sort(distObj);

figure
montage(img(:,:,:,j(1:25)))
title('top-left=query image. Images sorted using the groundtruth object labels.')






