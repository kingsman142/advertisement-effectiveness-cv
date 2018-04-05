%% Calculate basic data stats for CVPR paper


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



%% subject stats
num_repeats_seen = zeros(length(subj_data),1);
num_fillers_seen = zeros(length(subj_data),1);
for i=1:length(subj_data)
    if (isfield(subj_data{i},'target_detection_vector'))
        num_repeats_seen(i) = full(sum(abs(subj_data{i}.target_detection_vector)));
    end
    if (isfield(subj_data{i},'filler_detection_vector'))
        num_fillers_seen(i) = full(sum(abs(subj_data{i}.filler_detection_vector)));
    end
end
num_game_images_seen = num_repeats_seen + num_fillers_seen;

x = sort(num_repeats_seen,'descend');
y = cumsum(x)./sum(x);

disp('how many subjects cover >= 90% of target repeats: ');
disp(min(find(y>=0.90))) % how many subjects cover >= 90% of target repeats

disp('how many subjects saw any target repeats: ');
disp(sum(x~=0));


total_subj_entered_game = 0;
for i=1:length(subj_data)
    if (isfield(subj_data{i},'record_blocks'))
        total_subj_entered_game = total_subj_entered_game + (sum(~[subj_data{i}.record_blocks.was_demo])>0);
    end
end

disp('how many subjects entered the game (made it past the demo): ');
disp(total_subj_entered_game);

disp('avg number of levels played by subjects who entered the game (made it past demo) (level defined as 120 images, even though in actual game levels could have been shorter, e.g. if subject quit and resarted it would call it the next level): ');





%% num subjects per image stats
num_subjects_distr = [im_results.hits]+[im_results.misses];
disp('mean num scores per image: ');
disp(mean(num_subjects_distr));
disp('std num scores per image: ');
disp(std(num_subjects_distr))



%% hit rate stats
hrs = [im_results.hits]./([im_results.hits]+[im_results.misses]);
disp('mean hit rate: ');
disp(mean(hrs));
disp('std hit rate: ');
disp(std(hrs))



%% false alarm rate stats
fars = [im_results.false_alarms]./([im_results.false_alarms]+[im_results.correct_rejections]);
disp('mean false alarm rate: ');
disp(mean(fars));
disp('std false alarm rate: ');
disp(std(fars))





