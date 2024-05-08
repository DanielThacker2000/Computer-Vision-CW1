
function cw2_run_all_params(k_numbers, colour_spaces, vocab_sizes, is_weighteds, is_tfidfs, lambdas, distance_metrics, features, classifiers, disp_cf,output_name)

    num_train_per_cat = 100; 
    data_path = 'data/data';
    categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
           'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
           'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
    abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
        'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    fprintf('Getting paths and labels for all train and test data\n')
    [train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
    %generate all combinations of param to determine size of output csv
    %needed
    combinations = combvec(k_numbers, colour_spaces, vocab_sizes, is_weighteds, is_tfidfs, lambdas, distance_metrics, features, classifiers);
    parameter_matrix_new  = cell(size(combinations,2),size(combinations,1)+1);
    %%
    
    current_iteration=1;
    headers = {...
        'lambda', ...
        'distance_metric', ...
        'is_tfidf', ...
        'vocab_size', ...
        'is_weighted', ...
        'col_space', ...
        'k_number', ...
        'feature', ...
        'classifier', ...
        'accuracy'
    };
    parameter_matrix_new(current_iteration,:) = headers;
    current_iteration=2;
    for FEATURE = 1:numel(features)
        for col_space = 1:numel(colour_spaces)
            for vocab_size = 1:numel(vocab_sizes)
                for is_tfidf = 1:numel(is_tfidfs)
                    for is_weighted = 1:numel(is_weighteds)
                        switch lower(vocab_sizes(vocab_size))
                            case 50
                                vocab_file_name = "vocab.mat";
                                vocab_file_name_col = "vocab_col.mat";
                            case 100
                                vocab_file_name = "vocab_100.mat";
                                vocab_file_name_col = "vocab_col_100.mat";
                            case 150
                                vocab_file_name = "vocab_150.mat";
                                vocab_file_name_col = "vocab_col_150.mat";
                            case 200
                                vocab_file_name = "vocab_200.mat";
                                vocab_file_name_col = "vocab_col_200.mat";
                        end
    switch lower(features(FEATURE))    
         case 'bag of sift'
            % BAG OF SIFTS COLOUR
            train_image_feats = get_bags_of_sifts(train_image_paths,vocab_sizes(vocab_size)); 
            test_image_feats  = get_bags_of_sifts(test_image_paths,vocab_sizes(vocab_size)); 

        case 'bag of sift col'
            %BAG OF SIFTS COLOUR
            train_image_feats = get_bags_of_sifts_col_dist(train_image_paths,vocab_sizes(vocab_size)); 
            test_image_feats  = get_bags_of_sifts_col_dist(test_image_paths,vocab_sizes(vocab_size));

          case 'spatial_pyramids'
              train_image_feats = spatial_pyramid(train_image_paths, 3, 5, is_tfidfs(is_tfidf), vocab_sizes(vocab_size), is_weighteds(is_weighted),vocab_file_name);
              disp("done training")
              
              test_image_feats = spatial_pyramid(test_image_paths, 3, 5, is_tfidfs(is_tfidf), vocab_sizes(vocab_size), is_weighteds(is_weighted),vocab_file_name);
              disp("done test feats")
    
          case 'spatial_pyramids_col'
              train_image_feats = spatial_pyramid_col(train_image_paths, 3, 5, is_tfidfs(is_tfidf), vocab_sizes(vocab_size), is_weighteds(is_weighted),vocab_file_name_col);
              disp("done training")
              
              test_image_feats = spatial_pyramid_col(test_image_paths, 3, 5, is_tfidfs(is_tfidf), vocab_sizes(vocab_size), is_weighteds(is_weighted),vocab_file_name_col);
              disp("done test feats")
    
    end
    %% Step 2: Classify each test image by training and using the appropriate classifier
        
    for lambda = 1:numel(lambdas)
        for distance_metric = 1:numel(distance_metrics)
            for k_number = 1:numel(k_numbers)
                for CLASSIFIER = 1:numel(classifiers)
                    switch lower(classifiers(CLASSIFIER))    
                        case 'nearest neighbor'
                            %predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats);
                            predicted_categories = my_knn_classifier(train_image_feats, train_labels, test_image_feats, k_numbers(k_number), distance_metrics(distance_metric));
                        case 'support vector machine'
                            predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, lambdas(lambda));
                    end
    
    %calculate accuracy
    confusion_matrix = zeros(15, 15);
    for i = 1:length(predicted_categories)
        row = find(strcmp(test_labels{i}, categories));
        column = find(strcmp(predicted_categories{i}, categories));
        confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
    end

    num_test_per_cat = length(test_labels) / 15;
    confusion_matrix = confusion_matrix ./ num_test_per_cat;   
    accuracy = mean(diag(confusion_matrix));
    
    disp(['Accuracy: ', num2str(accuracy)]);
    
    if disp_cf
        accuracy = create_results_webpage( train_image_paths, ...
                                test_image_paths, ...
                                train_labels, ...
                                test_labels, ...
                                categories, ...
                                abbr_categories, ...
                                predicted_categories);
    end
    
    %define params and accuracy cell to be outputted as excel file
    params = {...
        sprintf('%d',lambdas(lambda)), ...
        sprintf('%s',distance_metrics(distance_metric)), ...
        sprintf('%d',is_tfidfs(is_tfidf)), ...
        sprintf('%d',vocab_sizes(vocab_size)), ...
        sprintf('%d',is_weighteds(is_weighted)), ...
        sprintf('%s',colour_spaces(col_space)), ...
        sprintf('%d',k_numbers(k_number)), ...
        sprintf('%s',features(FEATURE)), ...
        sprintf('%s',classifiers(CLASSIFIER)), ...
        sprintf('%d', accuracy)
        
        };
    
    parameter_matrix_new(current_iteration,:) = params;
    current_iteration=current_iteration+1;
                end
            end
                            end
                        end
                    end
                end
            end
        end
    end
    %%
    writecell(parameter_matrix_new,output_name)
end
