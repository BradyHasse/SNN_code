function outPCA = doPCA(data)
    % data: sample x variable
    %       variables should have been centered
    [coeff, score, latent, tsquared, explained, mu] = pca(data, 'Centered', false);
    
    outPCA.coeff = coeff;
    outPCA.score = score;
    outPCA.explained = explained;
    outPCA.mu = mu;
end