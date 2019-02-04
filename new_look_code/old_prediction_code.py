correct = 0
total = 0
predictions_in = []
predictions_out = []
predictions_train_size = math.floor(len(test_ids) * .5)
predictions_test_size = len(test_ids) - predictions_train_size
#print("Predictions test size: %d" % predictions_test_size)
predictions_train, predictions_test = train_test_split(test_ids, train_size = predictions_train_size, test_size = predictions_test_size)
predictions_test_labels = [test_out[test_ids.index(x)] for x in predictions_test]
predictions_test_counts = {key : 0 for key in range(1, 6)}
for id in predictions_test:
    predictions_test_counts[test_out[test_ids.index(id)]] += 1
#print("predictions test counts: %s" % predictions_test_counts)
predictions_test = normalize(predictions_test, predictions_test_labels, predictions_test_counts, [1, 2, 3, 4, 5])
predictions_test_counts = {key : 0 for key in range(1, 6)}
for id in predictions_test:
    predictions_test_counts[test_out[test_ids.index(id)]] += 1
#print("predictions test counts: %s" % predictions_test_counts)
predictions_sub_train = [[], [], [], [], [], [], [], [], [], []]
#predictions_sub_train = []
predictions_sub_out = [[], [], [], [], [], [], [], [], [], []]

for sample in test_ids:
    sample_index = test_ids.index(sample)
    topics_svm_class = topics_pred[sample_index]
    sents_svm_class = sents_pred[sample_index]
    topics_dt_class = topics_DT_pred[sample_index]
    sents_dt_class = sents_DT_pred[sample_index]
    mem_svm_class = mem_SVM_pred[sample_index]
    opflow_svm_class = opflow_SVM_pred[sample_index]
    avg_hue_svm_class = avg_hue_SVM_pred[sample_index]
    med_hue_svm_class = med_hue_SVM_pred[sample_index]
    cropped_30_class = cropped_30_SVM_pred[sample_index]
    cropped_60_class = cropped_60_SVM_pred[sample_index]
    exciting_class = exciting_pred[sample_index]
    total_svm_class = total_SVM_pred[sample_index]
    text_length_class = text_length_pred[sample_index]
    meaningful_words_class = meaningfulness_pred[sample_index]
    avg_word_len_class = avg_word_len_pred[sample_index]
    sent_anal_class = sent_anal_pred[sample_index]
    duration_class = duration_pred[sample_index]
    word_count_class = word_count_pred[sample_index]
    audio_class = audio_pred[sample_index]
    objects_class = objects_pred[sample_index]
    true_label = test_out[sample_index]
    predicted_class = -1

    topic = list(test_topics[sample_index]).index(1)
    sent = list(test_sents[sample_index]).index(1)
    if sents_svm_class == true_label:
        sents_svm_correct[sent] += 1
    if topics_svm_class == true_label:
        topics_svm_correct[topic] += 1
    if opflow_svm_class == true_label:
        opflow_topics_correct[topic] += 1
        opflow_sents_correct[sent] += 1
    if cropped_30_class == true_label:
        cropped_topics_correct[topic] += 1
        cropped_sents_correct[sent] += 1
    if mem_svm_class == true_label:
        mem_topics_correct[topic] += 1
        mem_sents_correct[sent] += 1
    if med_hue_svm_class == true_label:
        med_hue_topics_correct[topic] += 1
        med_hue_sents_correct[sent] += 1
    if topics_dt_class == true_label:
        topics_dt_correct[topic] += 1
    if sents_dt_class == true_label:
        sents_dt_correct[sent] += 1
    if cropped_60_class == true_label:
        cropped_60_topics_correct[topic] += 1
        cropped_60_sents_correct[sent] += 1
    if text_length_class == true_label:
        text_len_topics_correct[topic] += 1
        text_len_sents_correct[sent] += 1
    if meaningful_words_class == true_label:
        meaningfulness_topics_correct[topic] += 1
        meaningfulness_sents_correct[sent] += 1
    if avg_word_len_class == true_label:
        avg_word_len_topics_correct[topic] += 1
        avg_word_len_sents_correct[sent] += 1
    if sent_anal_class == true_label:
        sent_anal_topics_correct[topic] += 1
        sent_anal_sents_correct[sent] += 1
    if duration_class == true_label:
        duration_topics_correct[topic] += 1
        duration_sents_correct[sent] += 1
    if word_count_class == true_label:
        word_count_topics_correct[topic] += 1
        word_count_sents_correct[sent] += 1
    if audio_class == true_label:
        audio_topics_correct[topic] += 1
        audio_sents_correct[sent] += 1
    if objects_class == true_label:
        objects_topics_correct[topic] += 1
        objects_sents_correct[sent] += 1
    if places_class == true_label:
        places_topics_correct[topic] += 1
        places_sents_correct[sent] += 1
    if expressions_class == true_label:
        expressions_topics_correct[topic] += 1
        expressions_sents_correct[sent] += 1
    if emotions_class == true_label:
        emotions_topics_correct[topic] += 1
        emotions_sents_correct[sent] += 1
    if climax_class == true_label:
        climax_topics_correct[topic] += 1
        climax_sents_correct[sent] += 1
    topics_totals[topic] += 1
    sents_totals[sent] += 1

    if sample in predictions_train:
        predictions_sub = [sents_svm_class, topics_svm_class, opflow_svm_class, cropped_30_class, mem_svm_class, med_hue_svm_class, topics_dt_class, sents_dt_class, cropped_60_class, avg_hue_svm_class]
        #predictions = [predictions_sub_clfs[0].predict(predictions_sub[0])[0], predictions_sub_clfs[1].predict(predictions_sub[1])[0], predictions_sub_clfs[2].predict(predictions_sub[2])[0], predictions_sub_clfs[3].predict(predictions_sub[3])[0], predictions_sub_clfs[4].predict(predictions_sub[4])[0], predictions_sub_clfs[5].predict(predictions_sub[5])[0], predictions_sub_clfs[6].predict(predictions_sub[6])[0], predictions_sub_clfs[7].predict(predictions[7])[0], predictions_sub_clfs[8].predict(predictions_sub[8])[0], predictions_sub_clfs[9].predict(predictions_sub[9])[0]]
        #print(predictions)
        #predictions = [(0 if sents_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if topics_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if opflow_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if cropped_30_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if mem_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if med_hue_svm_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if topics_dt_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if sents_dt_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if cropped_60_class == true_label else 1) + (random.random() - 0.5)*.6, (0 if avg_hue_svm_class == true_label else 1) + (random.random() - 0.5)*.6]
        predictions = [0 if sents_svm_class == true_label else 1, 0 if topics_svm_class == true_label else 1, 0 if opflow_svm_class == true_label else 1, 0 if cropped_30_class == true_label else 1, 0 if mem_svm_class == true_label else 1, 0 if med_hue_svm_class == true_label else 1, 0 if topics_dt_class == true_label else 1, 0 if sents_dt_class == true_label else 1, 0 if cropped_60_class == true_label else 1, 0 if avg_hue_svm_class == true_label else 1]
        #print(predictions)
        predictions_sub_out[0].append(predictions[0])
        predictions_sub_out[1].append(predictions[1])
        predictions_sub_out[2].append(predictions[2])
        predictions_sub_out[3].append(predictions[3])
        predictions_sub_out[4].append(predictions[4])
        predictions_sub_out[5].append(predictions[5])
        predictions_sub_out[6].append(predictions[6])
        predictions_sub_out[7].append(predictions[7])
        predictions_sub_out[8].append(predictions[8])
        predictions_sub_out[9].append(predictions[9])
        predictions_sub_train[0].append([predictions_sub[0]])
        predictions_sub_train[1].append([predictions_sub[1]])
        predictions_sub_train[2].append([predictions_sub[2]])
        predictions_sub_train[3].append([predictions_sub[3]])
        predictions_sub_train[4].append([predictions_sub[4]])
        predictions_sub_train[5].append([predictions_sub[5]])
        predictions_sub_train[6].append([predictions_sub[6]])
        predictions_sub_train[7].append([predictions_sub[7]])
        predictions_sub_train[8].append([predictions_sub[8]])
        predictions_sub_train[9].append([predictions_sub[9]])
        predictions_in.append(predictions)
        predictions_out.append(true_label)
        predictions_sub_train.append(predictions_sub)

    #class_counts = collections.Counter([opflow_svm_class, cropped_30_class, cropped_60_class])
    class_counts = collections.Counter([opflow_svm_class, sents_svm_class, topics_svm_class, exciting_class, total_svm_class])
    predicted_class = class_counts.most_common(1)[0][0]
    total += 1

    if predicted_class == true_label:
        topics_correct[topic] += 1
        sents_correct[sent] += 1
        correct += 1

predictions_clf = SVC()
predictions_clf.fit(predictions_in, predictions_out)
predictions_sub_clfs = [SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC(), SVC()]
#predictions_sub_clfs = [LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression()]
for i in range(len(predictions_sub_clfs)):
    predictions_sub_clfs[i].fit(predictions_sub_train[i], predictions_sub_out[i])
print("Combiner accuracy: %.4f (%d correct, %d total)" % (correct/total, correct, total))
sents_svm_correct = np.array([sents_svm_correct[i] / sents_totals[i] for i in range(30)])
topics_svm_correct = np.array([topics_svm_correct[i] / topics_totals[i] for i in range(38)])
opflow_topics_correct = np.array([opflow_topics_correct[i] / topics_totals[i] for i in range(38)])
opflow_sents_correct = np.array([opflow_sents_correct[i] / sents_totals[i] for i in range(30)])
cropped_topics_correct = np.array([cropped_topics_correct[i] / topics_totals[i] for i in range(38)])
cropped_sents_correct = np.array([cropped_sents_correct[i] / sents_totals[i] for i in range(30)])
text_len_topics_correct = np.array([text_len_topics_correct[i] / topics_totals[i] for i in range(38)])
text_len_sents_correct = np.array([text_len_sents_correct[i] / sents_totals[i] for i in range(30)])
meaningfulness_topics_correct = np.array([meaningfulness_topics_correct[i] / topics_totals[i] for i in range(38)])
meaningfulness_sents_correct = np.array([meaningfulness_sents_correct[i] / sents_totals[i] for i in range(30)])
avg_word_len_topics_correct = np.array([avg_word_len_topics_correct[i] / topics_totals[i] for i in range(38)])
avg_word_len_sents_correct = np.array([avg_word_len_sents_correct[i] / sents_totals[i] for i in range(30)])
sent_anal_topics_correct = np.array([sent_anal_topics_correct[i] / topics_totals[i] for i in range(38)])
sent_anal_sents_correct = np.array([sent_anal_sents_correct[i] / sents_totals[i] for i in range(30)])
duration_topics_correct = np.array([duration_topics_correct[i] / topics_totals[i] for i in range(38)])
duration_sents_correct = np.array([duration_sents_correct[i] / sents_totals[i] for i in range(30)])
word_count_topics_correct = np.array([word_count_topics_correct[i] / topics_totals[i] for i in range(38)])
word_count_sents_correct = np.array([word_count_sents_correct[i] / sents_totals[i] for i in range(30)])
topics_correct = np.array([topics_correct[i] / topics_totals[i] for i in range(38)])
sents_correct = np.array([sents_correct[i] / sents_totals[i] for i in range(30)])

sents_dt_correct = np.array([sents_dt_correct[i] / sents_totals[i] for i in range(30)])
topics_dt_correct = np.array([topics_dt_correct[i] / topics_totals[i] for i in range(38)])
mem_topics_correct = np.array([mem_topics_correct[i] / topics_totals[i] for i in range(38)])
mem_sents_correct = np.array([mem_sents_correct[i] / sents_totals[i] for i in range(30)])
med_hue_topics_correct = np.array([med_hue_topics_correct[i] / topics_totals[i] for i in range(38)])
med_hue_sents_correct = np.array([med_hue_sents_correct[i] / sents_totals[i] for i in range(30)])
cropped_60_topics_correct = np.array([cropped_60_topics_correct[i] / topics_totals[i] for i in range(38)])
cropped_60_sents_correct = np.array([cropped_60_sents_correct[i] / sents_totals[i] for i in range(30)])

audio_topics_correct = np.array([audio_topics_correct[i] / topics_totals[i] for i in range(38)])
audio_sents_correct = np.array([audio_sents_correct[i] / sents_totals[i] for i in range(30)])
objects_topics_correct = np.array([objects_topics_correct[i] / topics_totals[i] for i in range(38)])
objects_sents_correct = np.array([objects_sents_correct[i] / sents_totals[i] for i in range(30)])
places_topics_correct = np.array([places_topics_correct[i] / topics_totals[i] for i in range(38)])
places_sents_correct = np.array([places_sents_correct[i] / sents_totals[i] for i in range(30)])
expressions_topics_correct = np.array([expressions_topics_correct[i] / topics_totals[i] for i in range(38)])
expressions_sents_correct = np.array([expressions_sents_correct[i] / sents_totals[i] for i in range(30)])
emotions_topics_correct = np.array([emotions_topics_correct[i] / topics_totals[i] for i in range(38)])
emotions_sents_correct = np.array([emotions_sents_correct[i] / sents_totals[i] for i in range(30)])
climax_topics_correct = np.array([climax_topics_correct[i] / topics_totals[i] for i in range(38)])
climax_sents_correct = np.array([climax_sents_correct[i] / sents_totals[i] for i in range(30)])
'''print(sents_svm_correct)
print(topics_svm_correct)
print(opflow_topics_correct)
print(opflow_sents_correct)
print(cropped_topics_correct)
print(cropped_sents_correct)
print(text_len_topics_correct)
print(text_len_sents_correct)
print(sent_anal_topics_correct)
print(sent_anal_sents_correct)
print(duration_topics_correct)
print(duration_sents_correct)
print("\n\n")
print(topics_totals)
print(sents_totals)
print("\n\n")
print(topics_correct)
print(sents_correct)'''
for i in range(len(topics_correct)):
    if topics_correct[i] < correct/total:
        pass#print("Topic %s" % TOPICS[i-1])
for i in range(len(sents_correct)):
    if sents_correct[i] < correct/total:
        pass#print("Sentiment %s" % SENTIMENTS[i-1])

correct = 0
total = 0
predictions_correct = 0
predictions_total = 0
classifications = {i : 0 for i in range(1, 6)}
misclassifications = {i : 0 for i in range(1, 6)}
for sample in test_ids:
    sample_index = test_ids.index(sample)
    topics_svm_class = topics_pred[sample_index]
    sents_svm_class = sents_pred[sample_index]
    topics_dt_class = topics_DT_pred[sample_index]
    sents_dt_class = sents_DT_pred[sample_index]
    mem_svm_class = mem_SVM_pred[sample_index]
    opflow_svm_class = opflow_SVM_pred[sample_index]
    avg_hue_svm_class = avg_hue_SVM_pred[sample_index]
    med_hue_svm_class = med_hue_SVM_pred[sample_index]
    cropped_30_class = cropped_30_SVM_pred[sample_index]
    cropped_60_class = cropped_60_SVM_pred[sample_index]
    text_length_class = text_length_pred[sample_index]
    meaningful_words_class = meaningfulness_pred[sample_index]
    avg_word_len_class = avg_word_len_pred[sample_index]
    sent_anal_class = sent_anal_pred[sample_index]
    word_count_class = word_count_pred[sample_index]
    audio_class = audio_pred[sample_index]
    objects_class = objects_pred[sample_index]
    places_class = places_pred[sample_index]
    expressions_class = expressions_pred[sample_index]
    emotions_class = emotions_pred[sample_index]
    climax_class = climax_pred[sample_index]
    true_label = test_out[sample_index]
    predicted_class = -1

    topic = list(test_topics[sample_index]).index(1)
    sent = list(test_sents[sample_index]).index(1)
    total += 1

    if sample in predictions_test:
        predictions_sub_test = [topics_svm_class, sents_svm_class, topics_dt_class, sents_dt_class, mem_svm_class, opflow_svm_class, avg_hue_svm_class, med_hue_svm_class, cropped_30_class, cropped_60_class]
        predictions_sub = [predictions_sub_clfs[0].predict(predictions_sub_test[0])[0], predictions_sub_clfs[1].predict(predictions_sub_test[1])[0], predictions_sub_clfs[2].predict(predictions_sub_test[2])[0], predictions_sub_clfs[3].predict(predictions_sub_test[3])[0], predictions_sub_clfs[4].predict(predictions_sub_test[4])[0], predictions_sub_clfs[5].predict(predictions_sub_test[5])[0], predictions_sub_clfs[6].predict(predictions_sub_test[6])[0], predictions_sub_clfs[7].predict(predictions_sub_test[7])[0], predictions_sub_clfs[8].predict(predictions_sub_test[8])[0], predictions_sub_clfs[9].predict(predictions_sub_test[9])[0]]
        #print(predictions_sub)
        #predictions = [0 if sents_svm_class == true_label else 1, 0 if topics_svm_class == true_label else 1, 0 if opflow_svm_class == true_label else 1, 0 if cropped_30_class == true_label else 1, 0 if mem_svm_class == true_label else 1, 0 if med_hue_svm_class == true_label else 1, 0 if topics_dt_class == true_label else 1, 0 if sents_dt_class == true_label else 1, 0 if cropped_60_class == true_label else 1, 0 if avg_hue_svm_class == true_label else 1]
        predicted_label = predictions_clf.predict([predictions_sub])
        if predicted_label == true_label:
            classifications[true_label] += 1
            predictions_correct += 1
        else:
            misclassifications[true_label] += 1
            #print("sample: %s, predicted: %d, ground-truth: %d" % (sample, predicted_label, true_label))
        predictions_total += 1

    sents_scores = [sents_svm_correct[sent], opflow_sents_correct[sent], cropped_sents_correct[sent], sents_dt_correct[sent], mem_sents_correct[sent], med_hue_sents_correct[sent], duration_sents_correct[sent], word_count_sents_correct[sent], meaningfulness_sents_correct[sent], avg_word_len_sents_correct[sent], sent_anal_sents_correct[sent], audio_sents_correct[sent], objects_sents_correct[sent], places_sents_correct[sent], expressions_sents_correct[sent], emotions_sents_correct[sent], climax_sents_correct[sent]]
    topics_scores = [topics_svm_correct[topic], opflow_topics_correct[topic], cropped_topics_correct[topic], topics_dt_correct[topic], mem_topics_correct[topic], med_hue_topics_correct[topic], duration_topics_correct[topic], word_count_topics_correct[topic], meaningfulness_topics_correct[topic], avg_word_len_topics_correct[topic], sent_anal_topics_correct[topic], audio_topics_correct[topic], objects_topics_correct[topic], places_topics_correct[topic], expressions_topics_correct[topic], emotions_topics_correct[topic], climax_topics_correct[topic]]
    #classes = [sents_svm_class, topics_svm_class, opflow_svm_class, cropped_30_class]
    classes = [sents_svm_class, opflow_svm_class, cropped_30_class, sents_dt_class, mem_svm_class, med_hue_svm_class, duration_class, word_count_class, meaningful_words_class, avg_word_len_class, sent_anal_class, audio_class, objects_class, places_class, expressions_class, emotions_class, climax_class, topics_svm_class, opflow_svm_class, cropped_30_class, topics_dt_class, mem_svm_class, med_hue_svm_class, duration_class, word_count_class, meaningful_words_class, avg_word_len_class, sent_anal_class, audio_class, objects_class, places_class, expressions_class, emotions_class, climax_class]
    high_sents_index = 0
    high_topics_index = 0
    for i in range(len(sents_scores)):
        if sents_scores[i] > sents_scores[high_sents_index]:
            high_sents_index = i
    for i in range(len(topics_scores)):
        if topics_scores[i] > topics_scores[high_topics_index]:
            high_topics_index = i
    if sents_scores[high_sents_index] > topics_scores[high_topics_index]:
        predicted_class = classes[high_sents_index]
    else:
        predicted_class = classes[len(sents_scores) + high_topics_index]

    if predicted_class == true_label:
        topics_correct[topic] += 1
        sents_correct[sent] += 1
        correct += 1
print("Combiner accuracy (NEW): %.4f (%d correct, %d total)" % (correct/total, correct, total))
print("Predictions SVC accuracy: %.4f" % (predictions_correct/predictions_total))
print("Train len: %d" % len(predictions_train))
print("Test len: %d" % len(predictions_test))
print(classifications)
print(misclassifications)
#print(predictions_clf.coef_)

print("Number of video ids: %d" % (len(video_ids)))
print("Topics score: %.4f" % (topics_score))
print("Sents score: %.4f" % (sents_score))
