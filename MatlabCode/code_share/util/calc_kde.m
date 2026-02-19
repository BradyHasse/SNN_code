%% Estimate probability density function from univariant data samples
%  Wrap density values around for a given sub-region of data sample values.
%  For example, original data samples (the difference between two variables of angles) 
%      range between -360 and 360 degrees, but the actually range should be 
%      between -180 and 180 degrees.
%      This function first use kde to find pdf over [-360, 360]. 
%      Then map [-360, -180) to [0, 180), (180, 360] to (-180, 0].
%  
% Hongwei Mao, 8/1/2025.

function [outF, outX] = calc_kde(x, varargin)

    %% set parameters
    % initialize parameters
    xTicks = [];
    xCenter = [];
    xWidth = [];
    xWrappedWidth = [];
    isVerbose = false;
    makePlot = false;

    % update parameters with input arguments
    numOptions = size(varargin, 2);
    for i = 1:2:numOptions
        switch varargin{i}
            case 'EvaluationPoints'
                xTicks = varargin{i+1};
            case 'EvaluationCenter'
                xCenter = varargin{i+1};
            case 'EvaluationWidth'
                xWidth = varargin{i+1};
            case 'WrappedEvalWidth'
                xWrappedWidth = varargin{i+1};
            case 'isVerbose'
                isVerbose = varargin{i+1};
            case 'makePlot'
                makePlot = varargin{i+1};
            otherwise
                assert(false, ['Error in my_kde.m: input argument ' varargin{i} ' is not supported.']);
        end
    end

    if isempty(xWidth)
        xWidth = range(x);
        scaleFactor = 10 ^ (floor(log10(xWidth)) - 1);
        xWidth = (ceil(xWidth / scaleFactor) + 4) * scaleFactor;  % rounding and expand a bit
        if isVerbose
            fprintf('\tmy_kde.m: Changed the range of estimation points from %f to %f.\n', ...
                    range(x), xWidth);
        end
    end

    if isempty(xCenter)
        xCenter = mean([min(x) max(x)]);
        scaleFactor = 10 ^ (floor(log10(xWidth)) - 1);
        xCenter = round(xCenter / scaleFactor) * scaleFactor;  % rounding
        if isVerbose
            fprintf('\tmy_kde.m: Changed the center of estimation points from %f to %f.\n', ...
                    mean([min(x) max(x)]), xCenter);
        end
    end

    if isempty(xTicks)
        dx = 10 ^ ceil(log10(xWidth)) / 1e3;
        xTicks = xCenter + [(-xWidth/2):dx:(xWidth/2)];
    else
        xCenter = mean(xTicks([1 end]));
        xWidth = range(xTicks);
        dx = median(diff(xTicks));
    end

    if min(x) < xTicks(1)
        fprintf('\tmy_kde.m: Warning - %d out of %d samples are to the left of evaluation region.\n', ...
                sum(x < xTicks(1)), length(x));
    end
    if max(x) > xTicks(end)
        fprintf('\tmy_kde.m: Warning - %d out of %d samples are to the right of evaluation region.\n', ...
                sum(x > xTicks(end)), length(x));
    end

    %% estimate pdf using kde
    % [f, xf] = kde(x, 'Kernel', 'normal', 'EvaluationPoints', xTicks, 'Bandwidth', 'plug-in');
    [f, xf] = kde(x, 'Kernel', 'normal', 'EvaluationPoints', xTicks);

    % estimate pdf using histcounts
    xEdges = [xTicks-dx/2 xTicks(end)+dx/2];
    counts = histcounts(x, xEdges);
    probabilities = counts / length(x);  % sum(probabilities) should be 1
    densities = probabilities / dx;

    %% wrap around pdf - circularly shift over the segments of pdf outside the given x range
    % wrappedTicks           [ x(1) x(2) ... x(N-1) x(N) ]
    % xticks:  l(1) ... l(M) | x(1) x(2) ... x(N-1) x(N) | r(1) ... r(K)
    %  -->          l(1) ... ... ... ... ... l(M)
    %  -->                          r(1) ... ... ... ... ... r(K)

    if ~isempty(xWrappedWidth)
        xLower = xCenter - xWrappedWidth/2;
        xUpper = xCenter + xWrappedWidth/2;

        indLower = find(abs(xTicks - xLower) < eps);
        indUpper = find(abs(xTicks - xUpper) < eps);

        wrappedTicks = xTicks(indLower:indUpper);
        wrappedLen = indUpper - indLower + 1;
        wrappedPDF = f(indLower:indUpper);

        % move over the part from the left
        leftPDF = f(1:(indLower-1));
        while ~isempty(leftPDF)
            currlen = length(leftPDF);
            if currlen >= wrappedLen
                idx = (currlen-(wrappedLen-1)+1):currlen;
                wrappedIdx = 1:(wrappedLen-1);
        
                wrappedPDF(wrappedIdx) = wrappedPDF(wrappedIdx) + leftPDF(idx);
        
                leftPDF = leftPDF(1, 1:(idx(1)-1));
            else
                idx = 1:currlen;
                wrappedIdx = (wrappedLen-currlen):(wrappedLen-1);
        
                wrappedPDF(wrappedIdx) = wrappedPDF(wrappedIdx) + leftPDF(idx);
        
                leftPDF = [];
            end
        end

        % move over the part from the right
        rightPDF = f((indUpper+1):end);
        while ~isempty(rightPDF)
            currlen = length(rightPDF);
            if currlen >= wrappedLen
                idx = 1:(wrappedLen-1);
                wrappedIdx = 2:wrappedLen;
        
                wrappedPDF(wrappedIdx) = wrappedPDF(wrappedIdx) + rightPDF(idx);

                rightPDF = rightPDF(1, wrappedLen:end);
            else
                idx = 1:currlen;
                wrappedIdx = 2:(2+currlen-1);
        
                wrappedPDF(wrappedIdx) = wrappedPDF(wrappedIdx) + rightPDF(idx);
        
                rightPDF = [];
            end
        end

        outX = wrappedTicks;
        outF = wrappedPDF;
    else
        outX = xf;
        outF = f;
    end

    % add extra evaluation points on both sides where pdf is zero
    if outF(1) > eps
        outF = [0 outF];
        outX = [outX(1)-dx outX];
    end
    
    if outF(end) > eps
        outF = [outF 0];
        outX = [outX outX(end)+dx];
    end

    % % % assert(range(diff(outX)) < 1e-10, 'my_kde.m: evaluation point intervals are not consistent.');
    % % % sum(outF * median(diff(outX)))


    %% make plots
    if makePlot
        figure; hold on;
        % % % bar(xTicks, densities);
        plot(xf, f, '-m', 'LineWidth', 2);
        plot(outX, outF, '-b', 'LineWidth', 2);
        legend({'Histogram', 'Sample PDF', 'Output PDF'});
        xlabel(sprintf('Values (bin size = %f)', dx)); ylabel('Probability density');
    end
    
end
