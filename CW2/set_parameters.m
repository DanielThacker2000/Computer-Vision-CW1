%THIS CODE ALLOWS YOU TO SET ALL OF THE ITERATION PARAMETERS AND CALLS
%CW2_RUN_ALL_PARAMS WHICH WILL ITERATE THROUGH THEM AND OUTPUT AN EXCEL
%SHEET AT THE END

output_name= "bow_all.xls"; %DONT OVERWRITE THE FILE!

%PARAMETERS
%choose whether to show a matrix or not
disp_cf = false;
k_numbers = [1,11,21,31,41];
%colour_spaces = ["RGB", "HSV","GREY"];
colour_spaces = ["RGB"];
vocab_sizes = [50,100,150,200];
is_weighteds = [0];
is_tfidfs = [0];
lambdas=[0.1,0.01,0.001,0.0001];
distance_metrics = ["euclidian","cheby"]; %NOT CURRENTLY TESTING MAHAL
features = ["bag of sift", "bag of sift col"];
classifiers = ["nearest neighbor","support vector machine"];

%Run the iterations - saves an output excel
cw2_run_all_params(k_numbers, colour_spaces, vocab_sizes, is_weighteds, is_tfidfs, lambdas, distance_metrics, features, classifiers,disp_cf,output_name);