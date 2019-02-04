'''def normalize_data_binary(names, labels, stats):
    topics_0_1 = {0: [], 1: []}
    names_two = {0: [], 1: []}
    for x in names:
        rating = int(labels[x])
        if x in stats and not rating == 3:
            if rating < 3:
                topics_0_1[0].append(stats[x])
                names_two[0].append(x)
            else:
                topics_0_1[1].append(stats[x])
                names_two[1].append(x)
    min_class = min(len(topics_0_1[0]), len(topics_0_1[1]))
    idx_0 = np.random.choice(np.arange(len(topics_0_1[0])), min_class, replace=False)
    idx_1 = np.random.choice(np.arange(len(topics_0_1[1])), min_class, replace=False)
    topics_0_1_0 = []
    topics_0_1_1 = []
    names_two_num = [[], []]
    for item in idx_0:
        topics_0_1_0.append(topics_0_1[0][item])
        names_two_num[0].append(names_two[0][item])
    for item in idx_1:
        topics_0_1_0.append(topics_0_1[1][item])
        names_two_num[1].append(names_two[1][item])
    new_list = topics_0_1_0 + topics_0_1_1
    new_list_output = [0]*min_class + [1]*min_class
    new_names_list = names_two_num[0] + names_two_num[1]
    indices = [i for i in range(min_class*2)]
    np.random.shuffle(indices)
    output_items = [new_list[indices[i]] for i in range(min_class*2)]
    output_labels = [new_list_output[indices[i]] for i in range(min_class*2)]
    output_names = [new_names_list[item] for item in indices]
    return output_items, output_labels, output_names'''

def normalize_data_binary(names, labels, stats):
    topics_0_1 = {1: [], 2: []}
    names_two = {1: [], 2: []}
    for x in names:
        rating = int(labels[x])
        if x in stats and not rating == 3:
            if rating < 3:
                topics_0_1[1].append(stats[x])
                names_two[1].append(x)
            else:
                topics_0_1[2].append(stats[x])
                names_two[2].append(x)
    min_class = min(len(topics_0_1[1]), len(topics_0_1[2]))
    idx_0 = np.random.choice(np.arange(len(topics_0_1[1])), min_class, replace=False)
    idx_1 = np.random.choice(np.arange(len(topics_0_1[2])), min_class, replace=False)
    topics_0_1_0 = []
    topics_0_1_1 = []
    names_two_num = [[], []]
    for item in idx_0:
        topics_0_1_0.append(topics_0_1[1][item])
        names_two_num[0].append(names_two[1][item])
    for item in idx_1:
        topics_0_1_0.append(topics_0_1[2][item])
        names_two_num[1].append(names_two[2][item])
    new_list = topics_0_1_0 + topics_0_1_1
    new_list_output = [1]*min_class + [2]*min_class
    new_names_list = names_two_num[0] + names_two_num[1]
    indices = [i for i in range(min_class*2)]
    np.random.shuffle(indices)
    output_items = [new_list[indices[i]] for i in range(min_class*2)]
    output_labels = [new_list_output[indices[i]] for i in range(min_class*2)]
    output_names = [new_names_list[item] for item in indices]
    return output_items, output_labels, output_names

def normalize_data_five(names, labels, stats):
    five_effectivenss_bins = {1: [], 2: [], 3: [], 4: [], 5: []} # store the data (e.g. optical flow values)
    names_five = {1: [], 2: [], 3: [], 4: [], 5: []} # store the IDs
    for x in names:
        rating = int(labels[x])
        if x in stats:
            five_effectivenss_bins[rating].append(stats[x])
            names_five[rating].append(x)
    min_class = min(len(five_effectivenss_bins[1]), len(five_effectivenss_bins[2]), len(five_effectivenss_bins[3]), len(five_effectivenss_bins[4]), len(five_effectivenss_bins[5]))
    idx_1 = np.random.choice(np.arange(len(five_effectivenss_bins[1])), min_class, replace=False)
    idx_2 = np.random.choice(np.arange(len(five_effectivenss_bins[2])), min_class, replace=False)
    idx_3 = np.random.choice(np.arange(len(five_effectivenss_bins[3])), min_class, replace=False)
    idx_4 = np.random.choice(np.arange(len(five_effectivenss_bins[4])), min_class, replace=False)
    idx_5 = np.random.choice(np.arange(len(five_effectivenss_bins[5])), min_class, replace=False)
    idx = [idx_1, idx_2, idx_3, idx_4, idx_5]
    five_effectivenss_bins_num = [[], [], [], [], []]
    names_five_num = [[], [], [], [], []]
    for i in range(5):
        for item in idx[i]:
            five_effectivenss_bins_num[i].append(five_effectivenss_bins[i+1][item])
            names_five_num[i].append(names_five[i+1][item])
    new_list = five_effectivenss_bins_num[0] + five_effectivenss_bins_num[1] + five_effectivenss_bins_num[2] + five_effectivenss_bins_num[3] + five_effectivenss_bins_num[4]
    new_names_list = names_five_num[0] + names_five_num[1] + names_five_num[2] + names_five_num[3] + names_five_num[4]
    new_list_output = [1]*min_class + [2]*min_class + [3]*min_class + [4]*min_class + [5]*min_class # store the ratings
    indices = [i for i in range(min_class*5)]
    np.random.shuffle(indices)
    output_items = [new_list[item] for item in indices] # values
    output_labels = [new_list_output[item] for item in indices] # effectiveness ratings
    output_names = [new_names_list[item] for item in indices] # IDs
    return output_items, output_labels, output_names

def normalize_data_four(names, labels, stats):
    topics_four = {1: [], 2: [], 4: [], 5: []}
    names_four = {1: [], 2: [], 3: [], 4: [], 5: []} # store the IDs
    for x in names:
        rating = int(labels[x])
        if x in stats and not rating == 3:
            topics_four[rating].append(stats[x])
            names_four[rating].append(x)
    min_class = min(len(topics_four[1]), len(topics_four[2]), len(topics_four[4]), len(topics_four[5]))
    idx_1 = np.random.choice(np.arange(len(topics_four[1])), min_class, replace=False)
    idx_2 = np.random.choice(np.arange(len(topics_four[2])), min_class, replace=False)
    idx_4 = np.random.choice(np.arange(len(topics_four[4])), min_class, replace=False)
    idx_5 = np.random.choice(np.arange(len(topics_four[5])), min_class, replace=False)
    idx = [idx_1, idx_2, idx_4, idx_5]
    topics_four_num = [[], [], [], []]
    names_four_num = [[], [], [], [], []]
    for i in range(4):
        for item in idx[i]:
            rating_val = i+1 if i < 2 else i+2
            topics_four_num[i].append(topics_four[rating_val][item])
            names_four_num[i].append(names_four[rating_val][item])
    new_list = topics_four_num[0] + topics_four_num[1] + topics_four_num[2] + topics_four_num[3]
    new_list_output = [1]*min_class + [2]*min_class + [4]*min_class + [5]*min_class
    new_names_list = names_four_num[0] + names_four_num[1] + names_four_num[2] + names_four_num[3]
    indices = [i for i in range(min_class*4)]
    np.random.shuffle(indices)
    output_items = [new_list[item] for item in indices]
    output_labels = [new_list_output[item] for item in indices]
    output_names = [new_names_list[item] for item in indices] # IDs
    return output_items, output_labels, output_names

def normalize(ids, labels, counts, ratings_range):
    output_video_ids = []
    output_video_labels = []
    lowest_index = -1
    for item in ratings_range:
        if lowest_index == -1 or counts[item] < counts[lowest_index]:
            lowest_index = item
    lowest_count = counts[lowest_index]
    for item in ratings_range:
        high = counts[item]
        indices_of_item = [i for i in range(len(labels)) if labels[i] == item]
        item_ids = [ids[i] for i in indices_of_item]
        item_labels = [labels[i] for i in indices_of_item]
        indices = random.sample(range(0, high), lowest_count)
        out_ids = [item_ids[i] for i in indices]
        out_labels = [item_labels[i] for i in indices]
        output_video_ids += out_ids
        output_video_labels += out_labels
    return output_video_ids
