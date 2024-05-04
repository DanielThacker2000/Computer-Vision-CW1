% Based on James Hays, Brown University 

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters. 
function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, lambda)
    categories = unique(train_labels); 
    num_categories = length(categories);
    predicted_categories = cell(size(test_image_feats, 1), 1);

    % Train 1-vs-all SVM classifiers
    svm_models = cell(num_categories, 1);
    %Treat spatial pyramids differently since it's a variable size cell
    
    for i = 1:num_categories
        % Create binary labels for the current category
        binary_labels = strcmp(train_labels, categories{i});
        svm_models{i} = fitcsvm(train_image_feats, binary_labels, 'KernelFunction', 'linear', 'BoxConstraint', lambda, 'Standardize', true, 'ClassNames', [false true]);
        % Enable posterior probability estimation
        svm_models{i} = fitPosterior(svm_models{i});
    end

    % Test the SVM classifiers
    for j = 1:size(test_image_feats, 1)
        scores = zeros(num_categories, 1);
        for i = 1:num_categories
            % Predict the probability of the sample belonging to the current class
            [~, score] = predict(svm_models{i}, test_image_feats(j, :));
            % Store the confidence score for this class
             %Returns index of positive class
            scores(i) = score(svm_models{i}.ClassNames == true);
        end
        % Select the class with the highest confidence score
        [~, idx] = max(scores);
        predicted_categories{j} = categories{idx};
    end
end
% function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, lambda)
%     run('vlfeat-0.9.21/toolbox/vl_setup.m');
%     categories = unique(train_labels); 
%     num_categories = length(categories);
%     predicted_categories = cell(size(test_image_feats, 1), 1);
% 
%     % Train 1-vs-all SVM classifiers
%     svm_models = cell(num_categories, 1);
%     for i = 1:num_categories
%         % Create binary labels for the current category
%         binary_labels = strcmp(train_labels, categories{i});
%         % Train SVM classifier for the current category
%         svm_models{i} = fitcsvm(train_image_feats, double(binary_labels), 'KernelFunction', 'linear', 'BoxConstraint', lambda);
%     end
% 
%     for j = 1:size(test_image_feats, 1)
%         scores = zeros(num_categories, 1);
%         for i = 1:num_categories
%             % Predict confidence score for the current category
%             [~, scores(i)] = predict(svm_models{i}, test_image_feats(j,:));
%         end
%         % Find the category with the highest confidence score
%         [~, max_idx] = max(scores);
%         predicted_categories{j} = categories{max_idx};
%     end
% end



% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in coursework_starter,
%because unique() sorts them. This shouldn't really matter, though.
