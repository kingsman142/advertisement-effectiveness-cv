%% Feature analysis script

%% get the valid images
clear sorted_target_data_struct_array;
for i=1:length(sorted_target_data)
    sorted_target_data_struct_array(i) = sorted_target_data{i}; 
end

N_images = 2222; % ignoring texture images
image_index_list = [];
for i=1:N_images
    if (sorted_target_data_struct_array(i).hits + sorted_target_data_struct_array(i).misses >= 20)
        image_index_list = [image_index_list; i];
    end
end

im_results = sorted_target_data_struct_array(image_index_list);


%% get hit rates
hrs = [im_results.hits]./([im_results.hits]+[im_results.misses]);


%% get the color statistics
mode_colors_hsv = zeros(length(image_index_list),3);
mean_colors_hsv = zeros(length(image_index_list),3);
for i=1:length(image_index_list)
    mode_colors_hsv(i,:) = GetModeColorH_S_andV(img(:,:,:,image_index_list(i)));
    mean_colors_hsv(i,:) = GetMeanColorHSV(img(:,:,:,image_index_list(i)));
    i
end


%% plot color stats versus hit rate
subplot(131);
[x jj] = sort(mean_colors_hsv(:,1));
y = hrs(jj)';
scatter(x, y, 40, hsv2rgb(mean_colors_hsv(jj,:)),'filled');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',1,'Color',[0,0,0]);
xlabel('mean hue');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');

subplot(132);
[x jj] = sort(mean_colors_hsv(:,2));
y = hrs(jj)';
scatter(x, y, 40, hsv2rgb(mean_colors_hsv(jj,:)),'filled');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',1,'Color',[0,0,0]);
xlabel('mean saturation');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');

subplot(133);
[x jj] = sort(mean_colors_hsv(:,3));
y = hrs(jj)';
scatter(x, y, 40, hsv2rgb(mean_colors_hsv(jj,:)),'filled');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',1,'Color',[0,0,0]);
xlabel('mean value');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');


%% print color analysis
print('-dpsc', fullfile('../Analysis/plots/',which_date,'color_vs_hit_rate.eps'));
close all;



%% 'contrast' stats versus hit rate
intensity_mean = zeros(length(image_index_list),1);
intensity_var = zeros(length(image_index_list),1);
intensity_skew = zeros(length(image_index_list),1);
intensity_kurt = zeros(length(image_index_list),1);
for i=1:length(image_index_list)
    intensity_image = mean(img(:,:,:,image_index_list(i)), 3);
    intensity_mean(i) = mean(intensity_image(:));
    intensity_var(i) = var(intensity_image(:));
    intensity_skew(i) = skewness(intensity_image(:));
    intensity_kurt(i) = kurtosis(intensity_image(:));
    i
end


%% plot 'contrast' stats versus hit rate
subplot(131);
[x jj] = sort(intensity_mean);
y = hrs(jj)';
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('intensity mean');
ylabel('memorability score');
set(gca,'xlim',[1,255]);
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');

subplot(132);
[x jj] = sort(intensity_var);
y = hrs(jj)';
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('intensity variance');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');

subplot(133);
[x jj] = sort(intensity_skew);
y = hrs(jj)';
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('intensity skewness');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');

%{
subplot(133);
[x jj] = sort(intensity_kurt);
y = hrs(jj)';
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('intensity kurtosis');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');
%}


%% print 'contrast' analysis
print('-dpsc', fullfile('../Analysis/plots/',which_date,'contrast_vs_hit_rate_v2.eps'));
close all;



%% object stats (non-semantic)

% counts versus hit rates
subplot(131);
[x jj] = sort(sum(Counts(:,image_index_list),1)');
y = hrs(jj)';
x = log(x);
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('log number of objects');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');


% mean areas versus hit rates
subplot(132);

%obj_mean_areas = Areas./Counts;
%obj_mean_areas(isnan(obj_mean_areas))=0;

for i=1:length(image_index_list)
    areas_curr = Areas(:,image_index_list(i));
    x(i) = mean(areas_curr(areas_curr~=0));
end

[x jj] = sort(x);
y = hrs(jj)';
x = log(x);
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('log mean area');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');


% max areas versus hit rates
subplot(133);

for i=1:length(image_index_list)
    areas_curr = Areas(:,image_index_list(i));
    x(i) = max(areas_curr(areas_curr~=0));
end

[x jj] = sort(x);
y = hrs(jj)';
x = log(x);
scatter(x, y, 1, 'filled','black');
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]);
%plot(x,f,'--','LineWidth',2,'Color',[0,0,0]);
xlabel('log max area');
ylabel('memorability score');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));
axis('square');


%% print object stats analysis
print('-dpsc', fullfile('../Analysis/plots/',which_date,'object_stats_vs_hit_rate.eps'));
close all;




%% content frequency analysis

% obj presence frequencies
obj_presences = Counts(:,1:2222)'>0;
obj_expected_hrs = zeros(size(obj_presences,2),1);
for i=1:size(obj_expected_hrs,1)
    obj_expected_hrs(i) = mean(hrs(obj_presences(:,i)));
end

subplot(131);
y = obj_expected_hrs;
x = log(sum(obj_presences,1))';
mask = ~isnan(y);
y = y(mask);
x = x(mask);
scatter(x, y, 5, 'filled','blue');
set(gca,'xlim',[0,max(x)]);
set(gca,'ylim',[0,1]);
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]); axis('square');
xlabel('log number of images with object in dataset');
ylabel('mean memorability score for images with object');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));


% object counts
obj_counts = Counts(:,1:2222);
obj_expected_hrs = zeros(size(obj_presences,2),1);
for i=1:size(obj_expected_hrs,1)
    obj_expected_hrs(i) = mean(hrs(obj_presences(:,i)));
end

subplot(132);
y = obj_expected_hrs;
x = log(sum(obj_counts,1))';
mask = ~isnan(y);
y = y(mask);
x = x(mask);
scatter(x, y, 5, 'filled','blue');
set(gca,'xlim',[0,max(x)]);
set(gca,'ylim',[0,1]);
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]); axis('square');
xlabel('log object type count in dataset');
ylabel('mean memorability score for images with object');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));


% scene category frequencies
scene_presences = logical(sceneCatFeatures(1:2222,:));
scene_expected_hrs = zeros(size(scene_presences,2),1);
for i=1:size(scene_expected_hrs,1)
    scene_expected_hrs(i) = mean(hrs(scene_presences(:,i)));
end

subplot(133);
y = scene_expected_hrs;
x = log(sum(scene_presences,1))';
mask = ~isnan(y);
y = y(mask);
x = x(mask);
scatter(x, y, 5, 'filled','blue');
set(gca,'xlim',[0,max(x)]);
set(gca,'ylim',[0,1]);
p = polyfit(x,y,1);
f = polyval(p,x);
hold on;
plot(x,f,'-','LineWidth',2,'Color',[1,0,0]); axis('square');
xlabel('log scene type count in dataset');
ylabel('mean memorability score for images of scene type');
title(sprintf('rank corr = %2.2f', corr(x, y, 'type', 'Spearman')));


%% print content frequency analysis
print('-dpsc', fullfile('../Analysis/plots/',which_date,'content_frequency_analysis.eps'));
close all;

