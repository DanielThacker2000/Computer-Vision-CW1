
function hist = my_hist_function(data, numBins)

    data = double(data);
    % Calculate the bin edges
    minData = min(data(:));
    %Convert to double to prevent rounding 
    minData = double(minData);
    maxData = max(data(:));
    maxData = double(maxData);
    binWidth = (maxData - minData) / numBins;
    binEdges = minData:binWidth:maxData;
    binEdges(end) = maxData; % Ensures last bin edge is correct
    
    % Initialize bin counts
    binCounts = zeros(1, numBins);
    
    % Calculate bin counts
    for i = 1:numBins
        binCounts(i) = sum(data(:) >= binEdges(i) & data(:) <= binEdges(i+1));
    end

    hist = binCounts;
    %Plot the histogram for debugging
    % bar(binEdges(1:end-1), binCounts, 'histc');
    % xlabel('Bins');
    % ylabel('Frequency');
    % title('Histogram of 1D Data Array');


end