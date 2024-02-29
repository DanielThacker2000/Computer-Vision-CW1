%S&B Colour indexing method
%Input params take sin the 1500x1 list of images plus the various
%extraction methods, including colour space e.g. 'RGB' or 'HSV', and number
%of bins in the histogram

% Example usage:
%data_path = 'data/data/train/bedroom/sun_aaajwnfblludyasb.jpg';


%num_bins = 8;
%normalize = true;
%features = extract_color_features(data_path, num_bins, normalize, "RGB");

function feature_matrix = my_colour_indexing(image_paths, num_bins, normalize, colour_space)
    num_images = numel(image_paths);
    %If greyscale, set the feature_matrix with less bins
    if colour_space == "Greyscale"
        feature_matrix = zeros(num_images, num_bins);
    else
        feature_matrix = zeros(num_images, num_bins * 3); % Initialize feature matrix
    end
    %Iterate over the images and return add to the feature matrix
    for i = 1:num_images
        % Read image
        image = imread(image_paths{i});
        %If greyscale, don't separate colour space into 3
        if colour_space == "Greyscale"
            % Convert image to grayscale
            if size(image, 3) == 3
                image = rgb2gray(image);
            end
        else
            % Convert image to specified color space
            if colour_space == "HSV"
                image = rgb2hsv(image);
            elseif colour_space =="LAB"
                image = rgb2lab(image);
            end
        end
        
        % Calculate histograms
        if colour_space == "Greyscale"
            hist_features = my_hist_function(image, num_bins);
        else
            space_1 = image(:,:,1);
            space_2 = image(:,:,2);
            space_3 = image(:,:,3);
            hist_1 = my_hist_function(space_1, num_bins);
            hist_2 = my_hist_function(space_2, num_bins);
            hist_3 = my_hist_function(space_3, num_bins);
            % Concatenate histograms
            hist_features = [hist_1(:); hist_2(:); hist_3(:)];
        end
        
        % Optionally normalize histograms
        if normalize
            hist_features = hist_features / sum(hist_features);
        end
        
        feature_matrix(i, :) = hist_features';
    end
end


