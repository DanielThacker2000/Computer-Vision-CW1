%feats = spatial_pyramid_yuck('data\data\test\bedroom\sun_abllxrmlmfgdbepz.jpg', 3, 5);
function image_feats = spatial_pyramid_col(image_paths, num_layers, step_size, tfidf_weighted, num_words, weight,vocab_file_name_col)
    % Initialize VLFeat toolbox
    run('vlfeat-0.9.21/toolbox/vl_setup.m'); 
    
    num_images = numel(image_paths);
    %num_images = 1 %DEBUG
    %image_feats = cell(num_images, 1);
    image_feats = zeros(num_images, num_words*(4+16+1));
    
    for i = 1:num_images
        % Read image
        image = imread(image_paths{i}); 
        %image = imread(image_paths);DEBUG
        %image = rgb2gray(image);
        pyramid = cell(num_images, 1);
        % Split image into spatial pyramid
        for layer = 1:num_layers
            pyramid{layer} = split_image(image, layer);
        end
        
        % Extract features for each region of the pyramid
        feats = [];
        weights = [0.25,0.3,0.45];
        for j = 1:num_layers

            region_feats = extract_sift_features(pyramid{j}, step_size,vocab_file_name_col);
            %Weight each histogram by the layer
            if weight
                region_feats = region_feats*weights(j);
            end
            %returns hist of visual words - now concatinate each histogram
            %together
            feats = [feats, region_feats(:)'];

        end
        
        % Store features for current image
        image_feats(i,:) = feats;

    end 
    %Apply tfidf weighting if requested
     if tfidf_weighted
         image_feats = get_tfidf_weights(image_feats, num_words, num_images);
     end
end

function pyramid = split_image(image, num_layers)
    % Split image into spatial pyramid recursively
    
    [h, w, ~] = size(image);
    % Base case: if num_layers is 1, divide image into 4 quarters
    if num_layers == 1
        pyramid = cell(1, 4);
        h_half = floor(h / 2);
        w_half = floor(w / 2);
        
        % Split image into quarters
        pyramid{1} = image(1:h_half, 1:w_half,:);
        pyramid{2} = image(1:h_half, w_half+1:end,:);
        pyramid{3} = image(h_half+1:end, 1:w_half,:);
        pyramid{4} = image(h_half+1:end, w_half+1:end,:);
        
    elseif num_layers == 2
        % Define number of rows and columns for splitting
        rows = 4;
        cols = 4;
        
        h_sixteenth = floor(h / rows);
        w_sixteenth = floor(w / cols);
        
        % Create coordinates using linspace
        y_coords = linspace(1, h, rows+1);
        x_coords = linspace(1, w, cols+1);
        
        % Loop through each sixteenth
        for i = 1:16
          % Calculate row and column indices from i
          row_idx = floor((i - 1) / cols) + 1;
          col_idx = mod(i - 1, cols) + 1;
          
          % Get starting and ending coordinates for this sixteenth
          y_start = floor(y_coords(row_idx));
          y_end = ceil(min(y_coords(row_idx + 1) - 1, h));
          x_start = floor(x_coords(col_idx));
          x_end = ceil(min(x_coords(col_idx + 1) - 1, w));
        
          % Extract sub-image
          pyramid{i} = image(y_start:y_end, x_start:x_end,:);
        end
    else
        pyramid = cell(1, 1);
        % Return original image if num_layers is greater than 2
        pyramid{1} = image;
    end
end

function histogram = extract_sift_features(images, step_size,vocab_file_name)
    vocab = load(vocab_file_name);
    num_words = size(vocab.vocab, 2);
    num_images = numel(images);
    
    histograms = zeros(num_images, num_words); % Initialize histogram matrix
    
    for i = 1:num_images
        % Normalize each color channel
        r = single(images{i}(:,:,1));
        g = single(images{i}(:,:,2));
        b = single(images{i}(:,:,3));

        r = (r - mean(r(:))) / std(r(:));
        g = (g - mean(g(:))) / std(g(:));
        b = (b - mean(b(:))) / std(b(:));

        % Extract SIFT features for each color channel
        [~, sift_features_r] = vl_dsift(r, 'step', step_size, 'Fast');
        [~, sift_features_g] = vl_dsift(g, 'step', step_size, 'Fast');
        [~, sift_features_b] = vl_dsift(b, 'step', step_size, 'Fast');
        
        % Concatenate SIFT features from all color channels
        sift_features = [sift_features_r sift_features_g sift_features_b];
        distances = vl_alldist2(single(sift_features), single(vocab.vocab));
    
        % Find the nearest visual word for each SIFT feature (each row)
        [~, assignments] = min(distances, [], 2);
    

        % Compute histogram of visual word occurrences
        histogram = histcounts(assignments, 1:(num_words+1));
        
        histograms(i, :) = histogram; 

    end
    
    % Concatenate histograms from all regions
    histogram = reshape(histograms.', 1, []);
    histogram = histogram / sum(histogram); % Normalize histogram
    %Concatinate all hists together then normalise
    %histogram = histogram / sum(histogram);
    %calc hist with vocab WEIGHT ACCORDING TO SPATIAL LEVEL - HIGHER
    %WEIGHT FOR MORE GRANULARITY
    %disp(size(feats))
end

function features = get_tfidf_weights(features, step_size, num_images)

  [num_rows, num_cols] = size(features);

  %initialize list for non-zero counts
  nonZeroCounts = zeros(1, step_size);

  %loop through columns in steps of step_size
  for col_start = 1:step_size:num_cols
    %define ending column for this step (ensure it doesn't exceed total columns)
    col_end = min(col_start + step_size - 1, num_cols);
    
    %extract current block of columns
    current_block = features(:, col_start:col_end);
    
    %count non-zero elements in each row
    nonZeroCounts = nonZeroCounts + sum(current_block ~= 0, 1);
  end
    for col_start = 1:step_size:num_cols
    %define ending column for this step 
    col_end = min(col_start + step_size - 1, num_cols);
    %calculate apply scaling factory to the block
    scaling_factor = log(num_images ./ nonZeroCounts);
    features(:, col_start:col_end) = features(:, col_start:col_end) .* scaling_factor;
   end
end