%Tiny image extraction features
%Takes in Nx1 path of images where N is the number of image paths
%Returns Nxd matrix where N is number of images and d is the lenght of the
%feature vector
%Take image- resize to 16x16. Either ignore aspect ratio or crop centre
%square. Normalized so it has 0 mean and unit length (sd 1?)
%Stretch out so it has 
%data_path = 'data/data/train/bedroom/sun_aaajwnfblludyasb.jpg';

function feature_matrix = my_tiny_image(image_paths, normalise, col_space)
    num_images = numel(image_paths);
    %If greyscale, change feature matrix size
    if col_space == "Greyscale" 
        feature_matrix = zeros(num_images, 16*16);
    else
        feature_matrix = zeros(num_images, 16*16*3); % Initialize feature matrix
    end
    for i = 1:num_images
        % Read the image
        img = imread(image_paths{i});

        if col_space == "Greyscale"
            img = rgb2gray(img);
        else
            % Convert image to specified color space
            if col_space == "HSV"
                img = rgb2hsv(img);
            elseif col_space =="LAB"
                img = rgb2lab(img);
            end
        end

        % Resize the image to 16x16
        img_resized = imresize(img, [16, 16],'bilinear');
    
        % Convert the resized image to a single long 1D vector
        feature_vector = double(reshape(img_resized, 1, []));
        %Normalise the vectors if chosen
        if normalise
            % Normalize the feature vector
            feature_vector = feature_vector - mean(feature_vector); % Subtract mean
            feature_vector = feature_vector / norm(feature_vector); % Normalize to unit length
        end
        % Assign the normalized feature vector to the feature matrix
        feature_matrix(i, :) = feature_vector;
    end
end
