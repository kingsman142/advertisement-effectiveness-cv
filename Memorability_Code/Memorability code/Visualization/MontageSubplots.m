%% MontageSubplots

function [] = MontageSubplots(ims)


    N = size(ims,4);
    
    ncols = floor(sqrt(N));
    nrows = ceil(N/ncols);
    
    for i=1:N
       
        subplot(ncols,nrows,i);
        imshow(ims(:,:,:,i));
        
    end

end