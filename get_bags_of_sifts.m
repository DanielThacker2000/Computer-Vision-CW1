% Implementated according to the starter code prepared by James Hays, Brown
% University 
% Michal Mackiewicz, UEA

%%
function [bos_features_normalized, labels] = get_bags_of_sifts(image_paths, vocab_file_name)
    vocab = load(vocab_file_name);
    vocab = vocab.vocab;

    disp("getting bag of sifts")
    disp(vocab)
    num_images = numel(image_paths);
    num_visual_words = size(vocab, 2);
    bos_features = zeros(length(image_paths), size(vocab, 2));
    sift_descriptors = cell(num_images, 1);
    labels = zeros(num_images, 1);
    step_size = 5;
    run('vlfeat-0.9.21/toolbox/vl_setup.m'); 
    
   % Iterate over the images 
    for i = 1:num_images
        % Read image
        image = imread(image_paths{i});
        image = single(rgb2gray(image));
        
        % Extract SIFT features
        %disp("Extracting sift features")
        [~, sift_descriptors{i}] = vl_dsift(image, 'step', step_size,'Fast');
        
        % Assign label
        [~, name, ~] = fileparts(image_paths{i});
        labels(i) = str2double(name(1:2)); % Extracting label from the filename
    end
    
    % Concatenate sift descriptors from all images into a single matrix
    all_sift_descriptors = single(cell2mat(sift_descriptors'));
    
    % Compute pairwise distances between descriptors and vocab
    disp(['all sift: '  num2str(size(all_sift_descriptors))]);
    disp(['vocab: '  num2str(size(vocab))]);
    distances = pdist2(vocab', all_sift_descriptors', 'euclidean');
    
    % For each image, find the nearest visual word and create histogram
    start_idx = 1;
    for i = 1:num_images
        end_idx = start_idx + size(sift_descriptors{i}, 2) - 1;
        nearest_vocab_indices = zeros(1, size(sift_descriptors{i}, 2));
        for j = 1:size(sift_descriptors{i}, 2)
            [~, min_idx] = min(distances(:, start_idx + j - 1));
            nearest_vocab_indices(j) = min_idx;
        end
        histogram = histcounts(nearest_vocab_indices, 1:num_visual_words + 1);
        bos_features(i, :) = histogram;
        start_idx = end_idx + 1;
    end
    
    % L2 normalization of bag of sifts features
    bos_features_normalized = bsxfun(@rdivide, bos_features, sqrt(sum(bos_features.^2, 2)));
end 
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or a visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.
%}