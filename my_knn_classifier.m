function predicted_labels = my_knn_classifier(training_data, training_labels, test_data, k)
    % Initialize predicted_labels
    predicted_labels = strings(size(test_data, 1), 1);

    % Calculate pairwise distances between training and test instances
    distances = pdist2(test_data, training_data);

    % Find k nearest neighbors for each test instance
    [~, sorted_indices] = sort(distances, 2);

    % Perform voting for each test instance
    for i = 1:size(test_data, 1)
        % Get indices of k nearest neighbors
        k_indices = sorted_indices(i, 1:k);

        % Get labels of k nearest neighbors
        k_labels = training_labels(k_indices);

        allWords = {};
        % Flatten the cell array into a single cell 
        for j = 1:numel(k_labels)
            allWords = [allWords; k_labels{j}];
        end
        
        % Get unique words and their counts
        [uniqueWords, ~, idx] = unique(allWords);
        wordCounts = accumarray(idx, 1);
        
        % Find the word with the highest count
        [maxCount, maxIndex] = max(wordCounts);
        mostRecurringWord = uniqueWords{maxIndex};
        %Convert the output into a string rather than characters
        predicted_labels(i) = convertCharsToStrings(mostRecurringWord);
    end
end