function rs = performRotatedPCA(histo_all)
    num_comp = 6; 
    % Normalize firing rates
    histo_all = permute(histo_all,[3,1,2]);
    frRange = range(histo_all,[2,3]);
    normConstant = round(median(frRange)*(3/17)*4)/4;

    frRange = range(histo_all, [2,3]);
    normFactor = frRange + normConstant;
    histo_all = histo_all ./ repmat(normFactor, [1, size(histo_all, 2), size(histo_all, 3)]);

    % Baseline subtraction
    baseline = repmat(mean(histo_all, 3), 1, 1, size(histo_all, 3));
    histos = histo_all - baseline;
    
    % Perform PCA
    data = reshape(histos, size(histos, 1), []);
    [COEFF, SCORE] = pca(data');
    
    % Rotate PCA factors
    [Rotated_Score, rmatrix] = rotatefactors(SCORE(:, 1:num_comp), 'Method', 'promax', 'power', 4, 'Normalize', false, 'Coeff', 3, 'Maxit', 5000);
    rs = reshape(Rotated_Score, size(histo_all, 2), size(histo_all, 3), num_comp);
end

