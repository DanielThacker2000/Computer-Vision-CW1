
FEATURE = 'spatial pyramids col';
vocab_size = 50;
weight_layers = true;
tfidf_weighted = false;
%FEATURE ='research';
%FEATURE = 'colour histogram';

CLASSIFIER = 'nearest neighbor';
% CLASSIFIER = 'support vector machine';
lambda = 0.01;
k_number = 11;
distance_metric = 'euclidian';
    



  
%%

output_name= "testing.xls"; %DONT OVERWRITE THE FILE!

%PARAMETERS
%choose whether to show a matrix or not
disp_cf = false;
k_numbers = [1];
%colour_spaces = ["RGB", "HSV","GREY"];
colour_spaces = ["RGB"];
vocab_sizes = [50];
is_weighteds = [1];
is_tfidfs = [1];
lambdas=[0.1];
distance_metrics = ["euclidian"];
fprintf('Using %s representation for images\n', FEATURE)
parameters = [];
accuracy_vector = [];
features = ["spatial_pyramids_col","spatial_pyramids"];
classifiers = ["nearest neighbor","support vector machine"];

%Run the iterations - saves an output excel
cw2_run_all_params(k_numbers, colour_spaces, vocab_sizes, is_weighteds, is_tfidfs, lambdas, distance_metrics, features, classifiers,disp_cf,output_name);