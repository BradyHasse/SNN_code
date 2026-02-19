function outRPCA = rotatePCA(outPCA, param)
    if nargin < 2
        numPC = 6;
        power = 4;
        gamma = numPC/2;
    else
        numPC = param.numPC;
        power = param.power;
        gamma = param.gamma;
    end
    
    pcaScore = outPCA.score(:, 1:numPC);
    pcaCoeff = outPCA.coeff(:, 1:numPC);
    
    % rScore = score*T
    %  score = data*coeff
    % rScore = data*(coeff*T)  project data onto coeff*T to get rScore;
    %                          axes in coeff*T are orthogonal to the other
    %                          rCoeff axes
    [rScore, T] = rotatefactors(pcaScore, ...
                                'Method', 'promax', 'power', power, ...
                                'Normalize', false, ...
                                'Coeff', gamma, ...
                                'Maxit', 5000);
                            
    % data = score*coeff'
    %      = (score*T)*(coeff*inv(T)')'
    %      = rScore*rCoeff'
    % therefore rCoeff = coeff * inv(T)';
    rCoeff = pcaCoeff * inv(T)';
    
    
    outRPCA.score = rScore;
    outRPCA.coeff = rCoeff;
end