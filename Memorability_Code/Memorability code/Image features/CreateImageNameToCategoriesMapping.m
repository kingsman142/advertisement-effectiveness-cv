%% Create mapping from image name to category labels

load('./Data/Image data/SUN_urls.mat');


%% create image names mapping

image_scene_categories = {};
n = 1;

for i=1:length(SUN)
    i
    scene_name = SUN(i).category;
    scene_name(scene_name=='\') = '/';
    for j=1:length(SUN(i).images)
        
        x = SUN(i).images(j);
        [foo image_name bar] = fileparts(x{1});
        
        image_name = [image_name '_crop.jpg'];
        
        image_scene_categories(n,:) = {image_name, scene_name};
        n = n+1;
    end
    
end



%% create mapping for sorted target images
clear sorted_target_data_struct_array;
for i=1:length(sorted_target_data)
    sorted_target_data_struct_array(i) = sorted_target_data{i}; 
end

sorted_image_scene_categories = {};

for i=1:length(sorted_target_data_struct_array)

    i
    
    [foo x bar] = fileparts(sorted_target_data_struct_array(i).filepath);
    x = [x bar];
    
    a = image_scene_categories(strcmp(image_scene_categories(:,1), x), 2);
    if (~isempty(a))
        sorted_image_scene_categories{i} = a{1};
    else
        sorted_image_scene_categories{i} = 'none';
    end
    
end


%% create scene category feature vectors
x = unique(sorted_image_scene_categories);
N = length(sorted_target_data_struct_array);
M = length(x);
sceneCatFeatures = zeros(N,M);
for i=1:N
    sceneCatFeatures(i,strcmp(x,sorted_image_scene_categories(i))) = 1;
end

