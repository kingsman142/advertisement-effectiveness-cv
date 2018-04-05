%% Prediction with options for all memorability prediction tasks



clearvars image_features K;


%% build feature vector for each image
if (~isempty(strfind(stat_features_to_use,'g'))) % gist
    image_features.gist_features = single(gist);
end
if (~isempty(strfind(stat_features_to_use,'s'))) % sift
    image_features.sift_features = single(sptHistsift);
end
if (~isempty(strfind(stat_features_to_use,'h'))) % hog
    image_features.hog_features = single(sptHisthog);
end
if (~isempty(strfind(stat_features_to_use,'i'))) % ssim
    image_features.ssim_features = single(sptHistssim);
end
if (~isempty(strfind(stat_features_to_use,'p'))) % pixels
    image_features.pixel_features = single(reshape(pixel_histograms,size(pixel_histograms,1),size(pixel_histograms,2)*size(pixel_histograms,3)));
    image_features.pixel_features = image_features.pixel_features / sum(image_features.pixel_features(1,:));
end
if (~isempty(strfind(stat_features_to_use,'c'))) % hue
    image_features.hue_feature = single(mean_hues);
end

if (~isempty(strfind(obj_features_to_use,'c'))) % counts
    image_features.object_count_features = single(full(Counts'));
    image_features.object_count_features = image_features.object_count_features./repmat(sum(image_features.object_count_features,2),[1,size(image_features.object_count_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'p'))) % presence
    image_features.object_presence_features = single(full((Counts>0)'));
    image_features.object_presence_features = image_features.object_presence_features./repmat(sum(image_features.object_presence_features,2),[1,size(image_features.object_presence_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'e'))) % scene categories
    image_features.scene_category_features = single(full(sceneCatFeatures));
    image_features.scene_category_features = image_features.scene_category_features./repmat(sum(image_features.scene_category_features,2),[1,size(image_features.scene_category_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'a'))) % areas
    image_features.object_area_features = single(full(Areas'));
    image_features.object_area_features = image_features.object_area_features./repmat(sum(image_features.object_area_features,2),[1,size(image_features.object_area_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'s'))) % spatial histograms
    image_features.object_spt_hist_features = single(sptHistObjects);
end
if (~isempty(strfind(obj_features_to_use,'n'))) % marginal counts
    x = single(full(Counts'));
    max_count = max(x(:));
    num_steps = 20;
    step_size = max_count/num_steps;
    
    % marginalize
    m_x = [];
    for i=1:size(x,1)
        x_c = x(i,:);
        %x_c = x_c(x_c~=0);
        m_x(i,:) = hist(x_c,0:step_size:max_count);
    end
    
    image_features.object_marginal_count_features = m_x;
    image_features.object_marginal_count_features = image_features.object_marginal_count_features./repmat(sum(image_features.object_marginal_count_features,2),[1,size(image_features.object_marginal_count_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'m'))) % marginal areas
    x = single(full(Areas'));
    max_area = max(x(:));
    num_steps = 600;
    step_size = max_area/num_steps;
    
    % marginalize
    m_x = [];
    for i=1:size(x,1)
        x_a = x(i,:);
        %x_a = x_a(x_a~=0);
        m_x(i,:) = hist(x_a,0:step_size:max_area);
    end
    
    image_features.object_marginal_area_features = m_x;
    image_features.object_marginal_area_features = image_features.object_marginal_area_features./repmat(sum(image_features.object_marginal_area_features,2),[1,size(image_features.object_marginal_area_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'b'))) % marginal spatial histograms
    num_objects = size(Areas,1);
    
    x0 = single(full(sptHistObjects));
    
    num_spt_hists = size(sptHistObjects,2)/num_objects;
    m_xs = [];
    for j=0:(num_spt_hists-1)
        x = x0(:,(num_objects*j+1):(num_objects*(j+1)));
        max_area = max(x(:));
        num_steps = 20;
        step_size = max_area/num_steps;

        % marginalize
        m_x = [];
        for i=1:size(x,1)
            x_a = x(i,:);
            %x_a = x_a(x_a~=0);
            m_x(i,:) = hist(x_a,0:step_size:max_area);
        end
        
        m_xs = [m_xs m_x];
    end
    
    image_features.object_marginal_spt_hist_features = m_xs;
    image_features.object_marginal_spt_hist_features = image_features.object_marginal_spt_hist_features./repmat(sum(image_features.object_marginal_spt_hist_features,2),[1,size(image_features.object_marginal_spt_hist_features,2)]);
end
if (~isempty(strfind(obj_features_to_use,'t'))) % attributes
    image_features.attribute_features = single(atts);
end


feature_types = fieldnames(image_features);


%% compute kernels

if (isfield(image_features,'gist_features')) % gist

    param.type = 'rbf';
    param.sig = 0.5;
    K.gist = kernel(image_features.gist_features, image_features.gist_features, param);

end
if (isfield(image_features,'sift_features')) % sift

    param.type = 'histintersection';
    K.sift = kernel(image_features.sift_features, image_features.sift_features, param);

end
if (isfield(image_features,'hog_features')) % hog

    param.type = 'histintersection';
    K.hog = kernel(image_features.hog_features, image_features.hog_features, param);

end
if (isfield(image_features,'ssim_features')) % ssim

    param.type = 'histintersection';
    K.ssim = kernel(image_features.ssim_features, image_features.ssim_features, param);

end
if (isfield(image_features,'pixel_features')) % pixel histograms

    param.type = 'histintersection';
    K.pixels = kernel(image_features.pixel_features, image_features.pixel_features, param);

end
if (isfield(image_features,'hue_feature')) % hue

    %param.type = 'linear';
    param.type = 'rbf';
    param.sig = 0.5;
    K.hue = kernel(image_features.hue_feature, image_features.hue_feature, param);

end
if (isfield(image_features,'object_count_features')) % object counts

    param.type = 'histintersection';
    K.object_counts = kernel(image_features.object_count_features, image_features.object_count_features, param);

end
if (isfield(image_features,'scene_category_features')) % scene category

    param.type = 'histintersection';
    K.scene_category_features = kernel(image_features.scene_category_features, image_features.scene_category_features, param);

end
if (isfield(image_features,'object_presence_features')) % object presences

    param.type = 'histintersection';
    K.object_presences = kernel(image_features.object_presence_features, image_features.object_presence_features, param);

end
if (isfield(image_features,'object_area_features')) % object areas

    param.type = 'histintersection';
    K.object_areas = kernel(image_features.object_area_features, image_features.object_area_features, param);

end
if (isfield(image_features,'object_spt_hist_features')) % object spatial histograms

    param.type = 'histintersection';
    K.object_spt_hists = kernel(image_features.object_spt_hist_features, image_features.object_spt_hist_features, param);

end
if (isfield(image_features,'object_marginal_count_features')) % marginal counts

    param.type = 'histintersection';
    K.object_marginal_counts = kernel(image_features.object_marginal_count_features, image_features.object_marginal_count_features, param);

end
if (isfield(image_features,'object_marginal_area_features')) % marginal areas

    param.type = 'histintersection';
    K.object_marginal_areas = kernel(image_features.object_marginal_area_features, image_features.object_marginal_area_features, param);

end
if (isfield(image_features,'object_marginal_spt_hist_features')) % marginal spatial histograms

    param.type = 'histintersection';
    K.object_marginal_spt_hists = kernel(image_features.object_marginal_spt_hist_features, image_features.object_marginal_spt_hist_features, param);

end
if (isfield(image_features,'attribute_features')) % attributes

    param.type = 'rbf';
    param.sig = 2;
    K.attributes = kernel(image_features.attribute_features, image_features.attribute_features, param);

end


%% kernel combination

kernel_names = fieldnames(K);

if (kernel_combination_type == 's')

    K_all = zeros(size(K.(kernel_names{1})));
    for i=1:length(kernel_names)
        K_all = K_all + K.(kernel_names{i});
    end

elseif (kernel_combination_type == 'p')

    K_all = ones(size(K.(kernel_names{1})));
    for i=1:length(kernel_names)
        K_all = K_all .* K.(kernel_names{i});
    end

elseif (kernel_combination_type == 'm') % mixed combination
    
    if (k_combination_types(1) == 'p')
        K_all = ones(size(K.(kernel_names{1})));
    elseif (k_combination_types(1) == 's')
        K_all = zeros(size(K.(kernel_names{1})));
    else
        error('unknown kernel combination type');
    end
    K_all = ones(size(K.(kernel_names{1})));
    for i=1:length(kernel_names)
        if (k_combination_types(i) == 'p')
            K_all = K_all .* K.(kernel_names{i});
        elseif (k_combination_types(i) == 's')
            K_all = K_all + K.(kernel_names{i});
        else
            error('unknown kernel combination type');
        end
    end
    
end


all_predicteds = [];
all_test_labels = [];
for trial_i=1:num_trials
    
    %% load label vector
    %{
    N = length(sorted_target_data);
    hits = zeros(N,1);
    misses = zeros(N,1);
    for i=1:N
        hits(i) =  sorted_target_data{i}.hits;
        misses(i) = sorted_target_data{i}.misses;
    end
    labels = double(hits./(hits+misses));
    test_labels_all = labels;
    %}
    train_labels_all = subject_hrs1(trial_i,:)';
    if (independent_subject_set == 'f')
        test_labels_all = train_labels_all;
    else
        test_labels_all = subject_hrs2(trial_i,:)';
    end
    
    T1_train_labels_all = T1_train_subject_hrs(trial_i,:)';
    T2_train_labels_all = T2_train_subject_hrs(trial_i,:)';

    
    %% which images to use? (just loading previous which_images and test/training split for now)
    %{
    %which_images = GetWhichImages(hits,misses);
    which_images = 1:2222; % exclude textures
    %which_images = [1:2222, 2301:2400]; % exclude just photo textures (include SP textures)
    %which_images = [1:2300]; % exclude SP textures
    
    for i=1:length(feature_types)
        image_features.(feature_types{i}) = image_features.(feature_types{i})(which_images,:);
    end
    labels = labels(which_images);
    %}


    %% split into training and test sets
    %[train_indices, test_indices] = SplitTrainTestIndices(size(labels,1));
    train_indices = image_train_indices(trial_i,:);
    test_indices = image_test_indices(trial_i,:);

    train_label = train_labels_all(train_indices);
    test_label = test_labels_all(test_indices);

    % Resample samples to make distribution of hit rates uniform
    if (~isempty(strfind(resampling,'1'))) % make train set uniform
        [train_indices] = MakeDistUniformByRemoving(train_label, train_indices);
    end
    if (~isempty(strfind(resampling,'2'))) % make test set uniform
        [test_indices] = MakeDistUniformByRemoving(test_label, test_indices);
    end

    for i=1:length(feature_types)
        train_features.(feature_types{i}) = image_features.(feature_types{i})(train_indices,:);
        test_features.(feature_types{i}) = image_features.(feature_types{i})(test_indices,:);
    end

    train_label = train_labels_all(train_indices);
    test_label = test_labels_all(test_indices);
    %hist(train_label,0.05:.1:1)
    %hist(test_label,0.05:.1:1)


    %% assign labels for top 100 classification
    if (Predictor_type == 'c')
        if (classification_type == 'm') % classification about median hit rate

            median_hr = median(actual);
            labels = double(actual>=median_hr).*2-1;

        elseif (classification_type == 't') % classification into top set

            %how_many_bottom = 500;

            [x i] = sort(train_label,'descend');

            num_top = classification_set_size;
            top = zeros(size(train_label));
            top(i(1:num_top)) = 1;
            top(i((num_top+1):end)) = -1;
            %top(i((end-(how_many_bottom-1)):end)) = -1;

            train_label = top;

            %which_images = [i(1:num_top); i((end-(how_many_bottom-1)):end)];
            %train_label = train_label(which_images);
            %train_data = train_data(which_images,:);


            [x i] = sort(test_label,'descend');

            num_top = num_top;
            top = zeros(size(test_label));
            top(i(1:num_top)) = 1;
            top(i((num_top+1):end)) = -1;
            %top(i((end-(how_many_bottom-1)):end)) = -1;

            test_label = top;

            %which_images = [i(1:num_top); i((end-(how_many_bottom-1):end)];
            %test_label = test_label(which_images);
            %test_data = test_data(which_images,:);

        elseif (classification_type == 'b') % classification into bottom set

            %how_many_bottom = 500;

            [x i] = sort(train_label,'ascend');

            num_bottom = classification_set_size;
            bottom = zeros(size(train_label));
            bottom(i(1:num_bottom)) = 1;
            botom(i((num_bottom+1):end)) = -1;

            train_label = bottom;


            [x i] = sort(test_label,'ascend');

            num_bottom = classification_set_size;
            bottom = zeros(size(train_label));
            bottom(i(1:num_bottom)) = 1;
            botom(i((num_bottom+1):end)) = -1;

            test_label = bottom;

        end
    end


    %% select current train/test split of kernel
    K_train_curr = K_all(train_indices, train_indices);
    K_test_curr = K_all(test_indices, train_indices);
    
    K_train_curr = double([(1:size(K_train_curr,1))', K_train_curr]); % include sample serial number as first column
    K_test_curr = double([(1:size(K_test_curr,1))', K_test_curr]);

    
    %% select training set split half kernels (for grid search on
    %% hyperparameters)    
    half = ceil(size(train_indices,2)/2);
    %T1_indices = train_indices(1:half); % this can always just split in half since train_indices is already randomized per trial
    %T2_indices = train_indices((half+1):end);
    
    for i=1:num_grid_search_trials
        %[x y] = SplitTrainTestIndices(size(train_indices,2));
        %T1_indices = train_indices(x);
        %T2_indices = train_indices(y);
        T1_indices = squeeze(T1_train_indices(trial_i,i,:));
        T2_indices = squeeze(T2_train_indices(trial_i,i,:));
        
        T1_train_label{i} = T1_train_labels_all(T1_indices);
        if (independent_subject_set == 'f')
            T2_train_label{i} = T1_train_labels_all(T2_indices);
        else
            T2_train_label{i} = T2_train_labels_all(T2_indices);
        end
        
        tmp1 = K_all(T1_indices, T1_indices);
        tmp2 = K_all(T2_indices, T1_indices);

        T1_K_train{i} = double([(1:size(tmp1,1))', tmp1]); % include sample serial number as first column
        T2_K_train{i} = double([(1:size(tmp2,1))', tmp2]);
    end
        
    
    %% do prediction
    
    if (Predictor_type == 'w')
        
        error('CWM not currently supported');
        
        %{
        % do CWM

        % Parameters regression 
        paramHor.NgF=4;
        paramHor.npca=32; 
        paramHor.SigmaX=.1; 
        paramHor.iterF=100; 

        % learning
        paramHor.A = pca((train_features.gist_features), paramHor.npca);
        [fv,paramHor.py,paramHor.mgy,paramHor.Cgy,paramHor.Cy,paramHor.by] = ...
            CWM(double(train_label)', double((train_features.gist_features)*paramHor.A)', paramHor.NgF, paramHor.iterF, 1, paramHor.SigmaX);

        % test
        test_pcs = ((test_features.gist_features)*paramHor.A);
        predicted = zeros(size(test_label,1),1);
        for i=1:size(test_label,1)
            predicted(i) = maxCWM(test_pcs(i,:)', paramHor.py, paramHor.mgy, paramHor.Cgy, paramHor.Cy, paramHor.by);
        end
        %}
    else
        
        % grid search to optimize hyperparameters on training set
        best_sqr_corr = -1;
        c_best = '0.01';
        p_best = '0.01';
        for i=1:7
            for j=1:7
                c = 10^(i-5);
                c_param = num2str(c);
                
                p = 10^(j-5);
                p_param = num2str(p);
                
                sqr_corrs = [];
                for grid_search_trial=1:num_grid_search_trials
                    if (Predictor_type == 'c')
                        model = svmtrain(double(T1_train_label{grid_search_trial}), T1_K_train{grid_search_trial}, ['-t 4 -s 0 -c ' c_param]);
                        [predicted_classes_curr, accuracy_curr, predicted_curr] = svmpredict(double(T2_train_label{grid_search_trial}), T2_K_train{grid_search_trial}, model);
                        %if (accuracy_curr(3) ~= inf && accuracy_curr(3) > best_sqr_corr) % inf (or nan) if prediction vector is constant
                            %predicted = predicted_curr;
                            %predicted_classes = predicted_classes_curr;
                            %best_sqr_corr = accuracy_curr(3);
                            %c_best = c_param
                            %p_best = p_param
                        %end
                        if (accuracy_curr(3) ~= inf && accuracy_curr(3) ~= nan)
                            sqr_corrs = [sqr_corrs accuracy_curr(3)];
                        end
                    elseif (Predictor_type == 'r')
                        model = svmtrain(double(T1_train_label{grid_search_trial}), T1_K_train{grid_search_trial}, ['-t 4 -s 3 -c ' c_param ' -p ' p_param]);
                        [predicted_curr, accuracy_curr, dec_values_curr] = svmpredict(double(T2_train_label{grid_search_trial}), T2_K_train{grid_search_trial}, model);
                        %if (accuracy_curr(3) ~= inf && accuracy_curr(3) > best_sqr_corr) % inf (or nan) if prediction vector is constant
                            %predicted = predicted_curr;
                            %best_sqr_corr = accuracy_curr(3);
                            %c_best = c_param
                            %p_best = p_param
                        %end
                        if (accuracy_curr(3) ~= inf && accuracy_curr(3) ~= nan)
                            sqr_corrs = [sqr_corrs accuracy_curr(3)];
                        end
                    end
                end
                
                if (~isempty(sqr_corrs))
                    curr_sqr_corr_mean = mean(sqr_corrs);
                    if (curr_sqr_corr_mean > best_sqr_corr)
                        best_sqr_corr = curr_sqr_corr_mean;
                        c_best = c_param
                        p_best = p_param
                    end
                end
            end
        end
        
        
        % using best hyperparameters found on training set split, make
        % actual predictions
        if (Predictor_type == 'c')
            % do SVM classification
            model = svmtrain(double(train_label), double(K_train_curr), ['-t 4 -s 0 -c ' c_best]);
            [predicted_classes, accuracy, predicted] = svmpredict(double(test_label), double(K_test_curr), model);
        elseif (Predictor_type == 'r')
            % do SVM regression
            model = svmtrain(double(train_label), double(K_train_curr), ['-t 4 -s 3 -c ' c_best ' -p ' p_best]);
            [predicted, accuracy, dec_values] = svmpredict(double(test_label), double(K_test_curr), model);
        end
    end
    
    all_predicteds(trial_i,:) = predicted;
    all_test_labels(trial_i,:) = test_label;

end


%% save results
if (do_save == 't')
    dir_name = [Predictor_type '_' classification_type '_' classification_set_size '_' ...
                                                               '_' stat_features_to_use '_' obj_features_to_use '_' resampling kernel_combination_type ...
                                                               '_' independent_subject_set];

    date1 = date;

    mkdir(['../Analysis/results/' date1]);
    mkdir(['../Analysis/results/' date1 '/' dir_name]);

    save(['../Analysis/results/' date1 '/' dir_name '/all_predicteds'], 'all_predicteds');
    save(['../Analysis/results/' date1 '/' dir_name '/all_test_labels'], 'all_test_labels');
end

