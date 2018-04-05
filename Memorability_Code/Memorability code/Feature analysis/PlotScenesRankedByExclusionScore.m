%% Plot objects ranked by exclusion score

%% get hit rates
N = size(sorted_target_data,1);
hits = zeros(N,1);
misses = zeros(N,1);
for i=1:N
    hits(i) =  sorted_target_data{i}.hits;
    misses(i) = sorted_target_data{i}.misses;
end
hrs = double(hits./(hits+misses));

%{
N_images = 2222; % ignoring texture images
image_index_list = [];
for i=1:N_images
    if (sorted_target_data{i}.hits + sorted_target_data{i}.misses >= 20)
        image_index_list = [image_index_list; i];
    end
end

hrs = hrs(image_index_list);
%}


%% calc scene scores (just mean hit rate per scene category)
MemorabilityPerCategoryScript;


%% plot
N = size(hrss_sorted_by_mean,1);
for i=1:N
    
    scene_name = hrss_sorted_by_mean{i,1};
    which_images = find(GetCategoricalMask2(sorted_image_scene_categories, scene_name));
    
    [foo jj] = sort(hrs(which_images),'descend');
    
    im_ind1 = which_images(jj(1));
    im_ind2 = which_images(jj(end));
    
    curr_scene_score = hrss_sorted_by_mean{i,3};
    
    subplot(2,N,(i-1)*2+1);
    imshow(img(:,:,:,im_ind1));
    subplot(2,N,(i-1)*2+2);
    imshow(img(:,:,:,im_ind2));
    
    title(sprintf('%s\nscene score = %2.2f', scene_name, curr_scene_score));
    
end


%%
print('-dpsc', fullfile('./Analysis/plots/',which_date,'scenes_sorted_by_mean_hit_rate.eps'));
close all;

