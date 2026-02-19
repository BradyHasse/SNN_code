function plotPCAScore(Score, times, colors, scoreID, markerTimes, markerYPos)
% Score: # sample x # condition x # PC
%
% One subplot per PC. In each subplot, one plot per condition.
% 

    [~, numCondition, numDim] = size(Score);
    
    if numDim > 9
        warning('Too many subplots! Only plotting first 9.');
        
        numDim = 9;
        Score = Score(:, :, 1:numDim);
    end
    
    numFigCol = 3; % 3 subplot columns
    numFigRow = ceil(numDim/numFigCol);
    
    yMax = ceil(max(Score(:))*2)/2;
    yMin = floor(min(Score(:))*2)/2;

    % subplot layout: dim-1 dim-3 dim-5
    %                     2     4     6
    % ---------------------------------
    % or                  1     4     7
    for icol = 1:numFigCol
        for irow = 1:numFigRow
            plotInd = icol+(irow-1)*numFigCol;
            subplot(numFigRow, numFigCol, plotInd); hold on;
            
            % plot scores
            dimInd  = (icol-1)*numFigRow+irow;
            for icond = 1:numCondition
                plot(times, squeeze(Score(:, icond, dimInd)), ...
                     'Color', colors(icond, :), 'LineWidth', 1);
            end
            if exist('markerTimes', 'var')
                if ~exist('markerYPos', 'var')
                    markerYPos = yMin+(yMax-yMin)/55;
                end
                plot(markerTimes, markerYPos*ones(1, length(markerTimes)), ...
                     '.', 'Color', 'k', 'MarkerSize', 8);
            end
            hold off; ylim([yMin yMax]);
            
            % modify axis 
            daspect([times(end) yMax-yMin 1]);
            set(gca, 'XTick', 0:0.2:0.8, 'YTick', -2:2, ...
                     'XLim', [times(1) times(end)], 'YLim', [yMin yMax], ...
                     'TickDir', 'out', 'TickLength', [0.03, 0.03], 'LineWidth', 1, ...
                     'FontSize', 8, 'FontName', 'Helvetica', ...
                     'TitleFontSizeMultiplier', 1, 'TitleFontWeight', 'normal');
            if exist('scoreID', 'var')
                title(sprintf('Component %d', scoreID(dimInd)));
            else
                title(sprintf('Component %d', dimInd));
            end
            
            % add labels
            if icol==1 && irow==numFigRow
                xlabel('Time (s)'); ylabel('Score (a.u.)');
            else
                h = gca;
                h.YAxis.Visible = 'off';
                set(gca, 'XTickLabel', '');
            end
        end %irow
    end %icol 
end
