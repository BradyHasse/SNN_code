%% Make a violin plot for data samples in each column of 'data'
%  at the corresponding location defined in 'pos'.
%  If 'Orientation' is 'vertical',   'pos' defines the x-axis locations of all plots.
%  If 'Orientation' is 'horizontal', 'pos' defines the y-axis locations of all plots.
%
% Hongwei Mao, 7/29/2025.

function [prctX, prctY] = make_violin_plot(data, pos, varargin)
    [numSample, numGroup] = size(data);

    % location of plots
    if nargin < 2
        pos = 1:numGroup;
    elseif length(pos) < numGroup
        pos = 1:numGroup;
    end

    %% set parameters
    Orientation = 'vertical';
    DensityDirection = 'both';
    EvaluationPoints = [];
    WrappedEvalWidth = [];
    DensityScaleFactor = 1;
    PrctLevels = [10 25 50 75 90];
    PrctLineStyles = {':', '--' , '-', '--', ':'};
    FaceColor = colororder;
    FaceAlpha = 0.4;
    EdgeColor = 'none';

    % update parameters with input arguments
    numOptions = size(varargin, 2);
    for i = 1:2:numOptions
        switch varargin{i}
            case 'Orientation'
                Orientation = varargin{i+1};
            case 'DensityDirection'
                DensityDirection = varargin{i+1};
            case 'EvaluationPoints'
                EvaluationPoints = varargin{i+1};
            case 'WrappedEvalWidth'
                WrappedEvalWidth = varargin{i+1};
            case 'DensityScaleFactor'
                DensityScaleFactor = varargin{i+1};
            case 'PrctLevels'
                PrctLevels = varargin{i+1};
            case 'PrctLineStyles'
                PrctLineStyles = varargin{i+1};
            case 'FaceColor'
                FaceColor = varargin{i+1};
            case 'FaceAlpha'
                FaceAlpha = varargin{i+1};
            case 'EdgeColor'
                EdgeColor = varargin{i+1};
            otherwise
                assert(false, ['Error in my_violin_plot.m: don''t recognize ' varargin{i} '.']);
        end
    end

    switch DensityDirection
        case 'positive'
            densityDir = 1;
        case 'negative'
            densityDir = -1;
        case 'both'
            densityDir = [1 -1];
    end
    

    %% estimate PDFs
    pdfX = cell(1, numGroup);
    pdfP = cell(1, numGroup);

    prctX = nan(length(PrctLevels), numGroup);
    prctY = nan(length(PrctLevels), numGroup);
    for igp = 1:numGroup
        a = data(:, igp);

        [f, xf] = calc_kde(a, 'EvaluationPoints', EvaluationPoints, 'WrappedEvalWidth', WrappedEvalWidth, ...
                            'makePlot', false);

        pdfP{igp} = f' * DensityScaleFactor;
        pdfX{igp} = xf';

        % for plotting lines to show percentile levels
        p = f .* median(diff(xf));
        cumsumP = cumsum(p);
        for iprct = 1:length(PrctLevels)
            [minVal, minLoc] = min(abs(cumsumP * 100 - PrctLevels(iprct)));
            prctX(iprct, igp) = xf(minLoc);
            prctY(iprct, igp) = f(minLoc) * DensityScaleFactor;
        end
    end
        
    %% calc. patch vertices from PDFs
    X = cell(1, numGroup);
    Y = cell(1, numGroup);
    switch Orientation
        case 'vertical'
            for igp = 1:numGroup
                X{igp} = [];
                Y{igp} = [];
                for idir = 1:length(densityDir)
                    tmpX = pdfP{igp} * densityDir(idir);
                    tmpY = pdfX{igp};
    
                    if idir == 2
                        tmpX = tmpX(end:-1:1, :);
                        tmpY = tmpY(end:-1:1, :);
                    end
                    X{igp} = [X{igp}; tmpX];
                    Y{igp} = [Y{igp}; tmpY];
                end
                X{igp} = X{igp} + pos(igp);  % arrange vertical plots horizontally along x axis
            end
            
        case 'horizontal'
            for igp = 1:numGroup
                X{igp} = [];
                Y{igp} = [];
                for idir = 1:length(densityDir)
                    tmpX = pdfX{igp};
                    tmpY = pdfP{igp} * densityDir(idir);
    
                    if idir == 2
                        tmpX = tmpX(end:-1:1);
                        tmpY = tmpY(end:-1:1);
                    end
                    X{igp} = [X{igp}; tmpX];
                    Y{igp} = [Y{igp}; tmpY];
                end
                Y{igp} = Y{igp} + pos(igp);  % arrange plots vertically along y axis
            end
            
        otherwise
            error(sprintf('my_violin_plot.m - %s is not a supported Orientation option. Should be either vertical or horizontal', Orientation));
    end


    %% make one patch per plot
    for igp = 1:numGroup
        icolor = mod(igp, size(FaceColor, 1));
        if icolor == 0
            icolor = size(FaceColor, 1);
        end
        patch(X{igp}, Y{igp}, FaceColor(icolor, :), 'EdgeColor', EdgeColor, 'FaceAlpha', FaceAlpha);
    end

    % % % %% plot lines to show percentiles
    % % % if ~isempty(prctY)
    % % %     switch Orientation
    % % %         case 'vertical'
    % % %             for igp = 1:numGroup
    % % %                 tmpY = repmat(prctX(:, igp), 1, 2);
    % % %                 tmpX = [];
    % % %                 for idir = 1:length(densityDir)
    % % %                     tmpX = [tmpX prctY(:, igp) * densityDir(idir)];
    % % %                 end
    % % %                 if size(tmpX, 2) == 1
    % % %                     tmpX = [zeros(length(tmpX), 1) tmpX] + pos(igp);
    % % %                 else
    % % %                     tmpX = tmpX + pos(igp);  % arrange vertical plots horizontally along x axis
    % % %                 end
    % % %                 for iprct = 1:size(tmpY, 1)
    % % %                     plot(tmpX(iprct, :), tmpY(iprct, :), PrctLineStyles{iprct}, 'Color', 'k');
    % % %                 end
    % % %             end
    % % % 
    % % %         case 'horizontal'
    % % %             for igp = 1:numGroup
    % % %                 tmpX = repmat(prctX(:, igp), 1, 2);
    % % %                 tmpY = [];
    % % %                 for idir = 1:length(densityDir)
    % % %                     tmpY = [tmpY prctY(:, igp) * densityDir(idir)];
    % % %                 end
    % % %                 if size(tmpY, 2) == 1
    % % %                     tmpY = [zeros(length(tmpY), 1) tmpY] + pos(igp);
    % % %                 else
    % % %                     tmpY = tmpY + pos(igp);  % arrange horizontal plots vertically along y axis
    % % %                 end
    % % %                 for iprct = 1:size(tmpY, 1)
    % % %                     plot(tmpX(iprct, :), tmpY(iprct, :), PrctLineStyles{iprct}, 'Color', 'k');
    % % %                 end
    % % %             end
    % % % 
    % % %         otherwise
    % % %             error(sprintf('my_violin_plot.m - %s is not a supported Orientation option. Should be either vertical or horizontal', Orientation));
    % % %     end
    % % % end

end
