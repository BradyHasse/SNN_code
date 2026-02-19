%% Draw axis for figure
%
%    Inputs
%           axisLim         [start finish] of axis line
%           axisPos         Y-position for x-axis or
%                           X-position for y-axis
%           axisLoc         'bottom', 'top'  -  x axis
%                           'left', 'right'  -  y axis
%
%           params - struct with the following fields:
%           axisColor       [r g b]
%           axisWidth       lineWidth for ticks and axis line
%           fontSize        font size for tick label and axis label
%           ticks           
%           tickLabel
%           tickDir         tick direction: 1 for inside; -1 for outside
%           axisLabel
%

function draw_axis_3d(axisLim, axisPos, axisLoc, params)
    %% horizontal (x) or vertical (y) axis?
    switch axisLoc
        case {'front', 'back'}
            axisType = 'x';
        case {'left', 'right'}
            axisType = 'y';
        case {'bottom', 'top'}
            axisType = 'z';
        otherwise
            error(['Error in draw_axis.m: ''' axisLoc ''' - not a valid axis location.']);
    end

    if ~exist('params', 'var')
        params = [];
    end
    
    %% calc. full range of axis (only plotting the part defined in axisLim)
    axisRange = diff(axisLim);

    if ~exist('rangeUnit', 'var')
        rangeUnit = 1;
    end

                              
    %% parse input parameters
    axisColor = [0 0 0];            % default color is black
    if isfield(params, 'color')
        axisColor = params.color;
    end
    
    axisWidth = 1;                  % default axis line width is 1
    if isfield(params, 'lineWidth')
        axisWidth = params.lineWidth;
    end
    
    fontSize  = 10;
    if isfield(params, 'fontSize')
        fontSize = params.fontSize;
    end
    
    tickDir   = -1;                 % 1: inside, -1: outside (default)
    if isfield(params, 'tickDir')
        tickDir = params.tickDir;
    end
    if strcmp(axisLoc, 'top') || strcmp(axisLoc, 'right')
        tickDir = -tickDir;
    end
    
    tickPos = [];
    if ~isfield(params, 'ticks')
        tickFirst = rangeUnit * ceil(axisLim(1)/rangeUnit);
        tickLast  = rangeUnit * floor(axisLim(end)/rangeUnit);
        tickPos = tickFirst:rangeUnit:tickLast;
    else
        tickPos = params.ticks;
    end
    numTick = length(tickPos);
    
    if ~isfield(params, 'tickLabel')
        if ~isempty(tickPos)
            tickLabel = cell(1, numTick);
            for i = 1:numTick
                tickLabel(i) = {num2str(tickPos(i), '%g')};
            end
        else
            tickLabel = {};
        end
    else
        tickLabel = params.tickLabel;
    end
    
    if isfield(params, 'axisLabel')
        axisLabel = params.axisLabel;
    else
        switch axisType
            case 'x'
                axisLabel = 'X axis';
            case 'y'
                axisLabel = 'Y axis';
            case 'z'
                axisLabel = 'Z axis';
        end
    end


    if isfield(params, 'axisLabelPos')
        axisLabelPos = params.axisLabelPos;
    end
    
%     tickLengthFactor = 0.015;
    tickLengthFactor = 0.200;
    if isfield(params, 'tickLengthFactor')
        tickLengthFactor = params.tickLengthFactor;
    end
    
%     tickLabelOffsetFactor = 0.020;
    tickLabelOffsetFactor = 0.50;
    if isfield(params, 'tickLabelOffsetFactor')
        tickLabelOffsetFactor = params.tickLabelOffsetFactor;
    end
    
%     axisLabelOffsetFactor = 0.060;
    axisLabelOffsetFactor = 1.0;
    if isfield(params, 'axisLabelOffsetFactor')
        axisLabelOffsetFactor = params.axisLabelOffsetFactor;
    end
    
    yOffsetMultiplier = 1.5;  % Y-axis tick label and axis label might need extra spacing
    if isfield(params, 'yOffsetMultiplier')
        yOffsetMultiplier = params.yOffsetMultiplier;
    end

    zOffsetMultiplier = 1.5;  % Z-axis tick label and axis label might need extra spacing
    if isfield(params, 'zOffsetMultiplier')
        zOffsetMultiplier = params.zOffsetMultiplier;
    end

    % parse input parameters - end
    
    
    %% set default length/offset which scale with range of the axis
    tickLength      = tickLengthFactor      * axisRange;  % line length for ticks
    tickLabelOffset = tickLabelOffsetFactor * axisRange;  % tick label to axis distance
    axisLabelOffset = axisLabelOffsetFactor * axisRange;  % axis label to axis distance
                                          
    
    %% plot axis line
    switch axisType
        case 'x'
            axisX = axisLim;
            axisY = [axisPos(1) axisPos(1)];
            axisZ = [axisPos(2) axisPos(2)];
        case 'y'
            axisY = axisLim;
            axisZ = [axisPos(1) axisPos(1)];
            axisX = [axisPos(2) axisPos(2)];
        case 'z'
            axisZ = axisLim;
            axisX = [axisPos(1) axisPos(1)];
            axisY = [axisPos(2) axisPos(2)];
    end

    plot3(axisX, axisY, axisZ, '-', 'Color', axisColor, 'LineWidth', axisWidth);
    
    %% plot axis ticks
    if ~isempty(tickPos)
        switch axisType
            case 'x'
                tickX = [tickPos; tickPos];
                tickY = [repmat(axisPos(1), 1, numTick);
                         repmat(axisPos(1), 1, numTick) + tickDir*tickLength];
                tickZ = repmat(axisPos(2), 2, numTick);
            case 'y'
                tickY = [tickPos; tickPos];
                tickZ = repmat(axisPos(1), 2, numTick);
                tickX = [repmat(axisPos(2), 1, numTick);
                         repmat(axisPos(2), 1, numTick) + tickDir*tickLength];
            case 'z'
                tickZ = [tickPos; tickPos];
                tickX = [repmat(axisPos(1), 1, numTick);
                         repmat(axisPos(1), 1, numTick) + tickDir*tickLength];
                tickY = repmat(axisPos(2), 2, numTick);
        end
        for i = 1:numTick
            plot3(tickX(:, i)', tickY(:, i)', tickZ(:, i)', '-', 'Color', axisColor, ...
                  'LineWidth', axisWidth);
        end
    end
    
    %% plot axis tick labels
    if ~isempty(tickLabel)
        if strcmp(axisType, 'x')
            tickLabelX = tickPos;
            tickLabelY = repmat(axisPos(1) + tickDir*tickLabelOffset, 1, numTick);
            tickLabelZ = repmat(axisPos(2), 1, numTick);
        elseif strcmp(axisType, 'y')
            tickLabelY = tickPos;
            tickLabelZ = repmat(axisPos(1), 1, numTick);
            tickLabelX = repmat(axisPos(2) + yOffsetMultiplier*tickDir*tickLabelOffset, 1, numTick);
        elseif strcmp(axisType, 'z')
            tickLabelZ = tickPos;
            tickLabelX = repmat(axisPos(1) + zOffsetMultiplier*tickDir*tickLabelOffset, 1, numTick);
            tickLabelY = repmat(axisPos(2), 1, numTick);
        end

        for i = 1:numTick
            t = text(tickLabelX(i), tickLabelY(i), tickLabelZ(i), tickLabel{i}, 'FontSize', fontSize);
            switch axisLoc
                case 'back'
                    set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
                case 'front'
                    set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
                case 'bottom'
                    set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
                case 'top'
                    set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
                case 'left'
                    set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
                case 'right'
                    set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            end
        end
    end
    
    %% plot axis label
    if ~isempty(axisLabel)
        if exist('axisLabelPos', 'var')
            axisLabelX = axisLabelPos(1);
            axisLabelY = axisLabelPos(2);
            axisLabelZ = axisLabelPos(3);
        else
            if strcmp(axisType, 'x')
                axisLabelX = axisLim(1);
                axisLabelY = axisPos(1) + tickDir*axisLabelOffset;
                axisLabelZ = axisPos(2);
            elseif strcmp(axisType, 'y')
                axisLabelY = axisLim(end);
                axisLabelZ = axisPos(1);
                axisLabelX = axisPos(2) + yOffsetMultiplier*tickDir*axisLabelOffset;
            elseif strcmp(axisType, 'z')
                axisLabelZ = axisLim(end);  % mean(axisLim);
                axisLabelX = axisPos(1) + zOffsetMultiplier*tickDir*axisLabelOffset;
                axisLabelY = axisPos(2);
            end
        end
        
        t = text(axisLabelX, axisLabelY, axisLabelZ, axisLabel, 'FontSize', fontSize);
        
        switch axisLoc
            case 'back'
                set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            case 'front'
                set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            case 'bottom'
                set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            case 'top'
                set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            case 'left'
                set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 0);
            case 'right'
                set(t, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 0);
        end
        
        set(t, 'Position', [axisLabelX, axisLabelY, axisLabelZ]);
    end
    
end
