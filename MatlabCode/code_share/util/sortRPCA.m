function outRPCA = sortRPCA(inRPCA, numBin, numCondition, numPC)
    score = inRPCA.score;
    coeff = inRPCA.coeff;
    
    matScore = reshape(score, numBin, numCondition, numPC);
    
    binPeak = nan(1, numPC);
    for i = 1:numPC
        rpcScore = squeeze(matScore(:, :, i)); % time x condition
        scoreRange = range(rpcScore, 2);
        [~, binPeak(i)] = max(scoreRange);
    end
    
    [binPeak, rpcOrder] = sort(binPeak, 'ascend');
    
    outRPCA.score = score(:, rpcOrder);
    outRPCA.coeff = coeff(:, rpcOrder);
    outRPCA.matScore = matScore(:, :, rpcOrder);
    outRPCA.binPeak  = binPeak;
    outRPCA.rpcID    = rpcOrder;
end