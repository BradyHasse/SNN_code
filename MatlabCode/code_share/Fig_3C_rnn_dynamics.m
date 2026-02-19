%% Fig 3C, S7C.  RNN neural trajectories and flow field.
%  e.g. Fig_3C_rnn_dynamics('MonkeyC')

function Fig_3C_rnn_dynamics(subject)

    addpath('.\util\');
    
    if ~exist('subject', 'var')
        subject = 'MonkeyC';    % monkeyC  monkeyN
    end
    
    %% Load data
    load(['./data/' sprintf('%s_RNN_Data.mat', subject)]);
    
    numBlock  = Data.numBlock;
    numTarget = Data.numTarget;
    numTrial  = numBlock * numTarget;
    
    Inputs    = Data.inputGau;  % gaussian shaped inputs
    numInput  = size(Inputs{1}, 1);
    Targets   = Data.fr;
    numOutput = size(Targets{1}, 1);
    ICs       = Data.ic;
    landmarkTimes = Data.landmarkTime;  % 1: Cue-0.2s; 2: Cue; 3: MoveOn; 4: Peak; 5: MoveOff; 6: End
    
    load('target_colormap.mat');
    colors = colors([8:numTarget 1:7], :);

    assert(numInput == 54);  % there are 3 x 18 inputs
    inputGroups = {[1:18]; 18+[1:18]; 2*18+[1:18]};  
    
    %% load trained networks and get network states and firing rates
    irun = 1;
    netsDir = './rnn_results';

    resultsFileName = sprintf('Results_%s_Run%d_SingleTrial.mat', subject, irun);
    resultsFilePath = [netsDir '/' resultsFileName];
    netResults = load(resultsFilePath);

    rnnXs = netResults.rnnXs;
    numRNNUnit = size(rnnXs{1}, 1);

    %% align trials at behavioral landmarks
    alignedFileName = sprintf('Results_%s_Run%d_Aligned.mat', subject, irun);
    alignedFilePath = [netsDir '/' alignedFileName];
    resAligned = load(alignedFilePath);

    matX = resAligned.matX;
    numBin = size(matX, 3);
    landmarkBinInd = resAligned.landmarkBinInd;
    
    blockNos = reshape(repmat(1:numBlock, numTarget, 1), 1, numTrial);
    tgtNos = repmat(1:numTarget, 1, numBlock);

    %============================== choose one block of trials for display ==================================================
    switch subject
        case 'MonkeyC'
            blocks2Plot = 25;  % choose a block of trials
            trialInd2Highlight = 4;
            ssAxisIDs = [1 4 5];  % choose rPCA axes for plotting 3-D state space trajectories
            flowFigViewAngle = [-60 30];
            segPlaneOffset = [-6 -5 -10];
            segPlaneAxisScalar = [ 7   5  10
                                  18  10  30];
            xLimMultiplier      = 1.2;
            yLimMultiplier      = 1.2;
            zLimMultiplier      = 1.2;
        case 'MonkeyN'
            blocks2Plot         = 25;
            trialInd2Highlight  = 1;
            ssAxisIDs           = [1 4 6];
            flowFigViewAngle    = [160 25];
            segPlaneOffset      = [-6 -5 -10];
            segPlaneAxisScalar  = [ 7   8  15
                                   20  14  80];
            xLimMultiplier      = 1.2;
            yLimMultiplier      = 1.2;
            zLimMultiplier      = 1.2;
    end

    trials2Plot = [];
    for i = 1:length(blocks2Plot)
        trials2Plot = [trials2Plot find(blockNos == blocks2Plot(i))];
    end

    % figure out in which CV fold the chosen block was used for testing
    for i = 1:size(netResults.trialMask, 1)
        testTrialMask = netResults.trialMask{i, 3};
        testTrialNos = find(testTrialMask);
        trials2PlotCVTrialIdx = find(testTrialNos == trials2Plot(1));
        if ~isempty(trials2PlotCVTrialIdx)
            icv = i;
            break;
        end
    end
    % % % trials2PlotCVTrialIdx = trials2PlotCVTrialIdx + trials2Plot - trials2Plot(1);  % trial No.'s in that CV fold

    % load the corresponding RNN net file
    net_file_name = netResults.allNetworkFileNames{icv};
    net_file_path = [netsDir '/' net_file_name];
    load(net_file_path);
    % % % fprintf('*** File %s loaded.\n', net_file_path);

    % RNN network parameters
    dt  = net.dt;
    tau = net.tau;
    dt_o_tau = dt / tau;
    [n_Wru_v, n_Wrr_n, ~, ~, n_bx_1, ~] = unpackRNN(net, net.theta);

    
    %% dimensionality reduction to find state-space axes for RNN
    binIdx = landmarkBinInd(1) : (landmarkBinInd(4) + 250/5);  % use the first 250 ms of Hold

    % rPCA of X
    avgX    = squeeze(mean(matX(:, :, binIdx, :), 1));  % trial averaging
    nondirX = mean(avgX, 1);
    dirX    = avgX - nondirX;  % remove non-directional component

    tmpX  = permute(dirX, [2 1 3]);
    longX = reshape(tmpX, length(binIdx)*numTarget, size(tmpX, 3));
    pcaX  = doPCA(longX);

    % rotate scores - d x m; d = numBin*numCondition, m = numPC
    numPC = 6;
    d = length(binIdx)*numTarget;    m = numPC;
    rpcaParam.numPC = numPC;
    rpcaParam.power = 4;
    rpcaParam.gamma = m/2;
    outRPCA = rotatePCA(pcaX, rpcaParam);
    outRPCA = sortRPCA(outRPCA, length(binIdx), numTarget, numPC);


    % get 3 orthogonal axes from the 6 rPCA axes
    newAxis12 = outRPCA.coeff(:, 1:2);
    nullOf12  = null(newAxis12');
    tmpProjs  = nullOf12' * outRPCA.coeff(:, 3:4);
    newAxis34 = nullOf12 * tmpProjs;

    nullof1234 = null([newAxis12 newAxis34]');
    tmpProjs = nullof1234' * outRPCA.coeff(:, 5:6);
    newAxis56 = nullof1234 * tmpProjs;

    newAxes = [newAxis12 newAxis34 newAxis56];
    newAxes = newAxes ./ repmat(vecnorm(newAxes), numRNNUnit, 1);  
    ssAxes  = newAxes(:, ssAxisIDs);  

    % project aligned X data onto the 3-D rPCA state space
    avgX    = squeeze(mean(matX, 1));
    tmpX  = permute(avgX, [2 1 3]);
    longX = reshape(tmpX, size(tmpX, 1)*numTarget, size(tmpX, 3));
    ssScores = longX * ssAxes;

    % projection of X data onto the remaining 1000-3 dimensional subspace
    nullAxes = null(ssAxes');
    nullScores = longX * nullAxes;
    meanNullScore = mean(nullScores, 1);  % mean scores (997 x 1) across time and directions
    meanNullX = nullAxes * meanNullScore';  % X values coming from the mean 997-D scores


    %% calc the X, Y, and Z range of data in the 3-D state space
    %  and compute the gradients on the middle 2-D plane due to intrinsic RNN dynamics
    [minX, maxX] = bounds(ssScores(:, 1));
    [minY, maxY] = bounds(ssScores(:, 2));
    [minZ, maxZ] = bounds(ssScores(:, 3));

    minX = floor(xLimMultiplier * minX * 2) / 2;     maxX = ceil(xLimMultiplier * maxX * 2) / 2; 
    minY = floor(yLimMultiplier * minY * 2) / 2;     maxY = ceil(yLimMultiplier * maxY * 2) / 2; 
    minZ = floor(zLimMultiplier * minZ * 2) / 2;     maxZ = ceil(zLimMultiplier * maxZ * 2) / 2; 
    
    % make the grids where gradients will be calculated
    Xticks = minX : 0.5 : maxX;
    Yticks = minY : 0.5 : maxY;
    Zticks = mean(ssScores(:, 3));

    [X, Y, Z] = meshgrid(Xticks, Yticks, Zticks);  % coordinates for grid points in state space
    coords = [X(:) Y(:) Z(:)]';
    Mones  = ones(length(Yticks), length(Xticks), length(Zticks));
    Mzeros = zeros(length(Yticks), length(Xticks), length(Zticks));
    
    gridXs = ssAxes * coords + meanNullX;  % partial X from the 3-D rPCA space + partial X from remaining 997-D space

    gridXdots = calc_dx(gridXs, net);

    if size(gridXdots, 3) > 1
        gridXdots = sum(gridXdots, 3);
    end
    
    ssXdots = ssAxes' * gridXdots;  % gradients (dx) projected onto the 3-D state space
    U = reshape(ssXdots(1, :), size(X));
    V = reshape(ssXdots(2, :), size(Y));
    W = reshape(ssXdots(3, :), size(Z));

    
    %% plot neural traj and intrinsic flow field
    arrowSize = 4;
    arrowLength = arrowSize * 0.12 / 6;

    figure('Position', [100 100 4*250 3*250]); hold on;
    
    % plot flow field (gradients due to intrinsic dynamics) on the bottom plane
    quiverScale = 1.0;
    flowZ = minZ - 1.0;  % z position of the 2-D plane used to plot the flow chart
    quiver3(X, Y, flowZ*Mones, U*quiverScale, V*quiverScale, Mzeros, 'Color', [158,154,200]/255, 'AutoScale', 'off');  

    % plot  border of the bottom plane
    patch([Xticks(1) Xticks(end) Xticks(end) Xticks(1)], ...
          [Yticks(1) Yticks(1) Yticks(end) Yticks(end)], ...
          flowZ*ones(1, 4), 'k', 'FaceColor', 'none', 'EdgeColor', 'k');
    

    % plot single-trial RNN neural traj in state space
    quiverScale = 1.0;
    for i = 1:1:length(trials2Plot)
        itrial = trials2Plot(i);
        itgt = tgtNos(itrial);
        trialTimes = Data.times{itrial};
        landmarks = landmarkTimes(:, itrial)';

        Xs = rnnXs{itrial};
        Ss = Xs' * ssAxes;
    
        bins = find(trialTimes >= landmarks(2) & trialTimes <= landmarks(5)+0.30);  % from target onset to 300 ms after movement offset
            
        if i == trialInd2Highlight
            trialColor = colors(itgt, :);
            trialLineWidth = 2;
            trajArrowSize = 8;
        else
            trialColor = 0.50*[1 1 1];
            trialLineWidth = 1;
            trajArrowSize = 4;
        end

        plot3(Ss(bins, 1), Ss(bins, 2), Ss(bins, end), 'Color', trialColor, 'LineWidth', trialLineWidth);

        % plot a arrow to indicate direction of flow
        if strcmp(subject, 'MonkeyC')
            medBin = floor(median(bins));
        elseif strcmp(subject, 'MonkeyN')
            medBin = floor(median(bins)) - 25;
        end
        draw_arrow_3d(Ss(medBin, :), Ss(medBin, :)-Ss(medBin-1, :), 'faceColor', trialColor, 'arrowSize', trajArrowSize, 'centerIsTip', true);

        % plot shadow of trajectories on the bottom plane
        plot3(Ss(bins, 1), Ss(bins, 2), flowZ*ones(length(bins), 1), 'Color', trialColor, 'LineWidth', trialLineWidth);  
    end

    
    % for highlighted trials, plot gradient arrows along the trajectory, and make three planes to represent the epochs
        itrial = trials2Plot(trialInd2Highlight);
        Xs = rnnXs{itrial};
        Ss = Xs' * ssAxes;

        trialTimes = Data.times{itrial};
        landmarks = landmarkTimes(:, itrial)';
    
        bins = find(trialTimes >= landmarks(2) & trialTimes <= landmarks(5)+0.30);  
        diffSs = diff(Ss([bins(1)-1 bins], :), 1, 1);
        diffSs = sqrt(sum(diffSs .^ 2, 2));
        cumsumSs = cumsum(diffSs);
    
        % calc and plot gradients
        trialInputs = Inputs{itrial};
        trialInputs = [trialInputs(:, 2:end) trialInputs(:, end)];    % dx(t) for x(t+1) was calculated with u(t+1) - HW 6/11/24
        [dX, dX_u]  = calc_dx(Xs, net, trialInputs, inputGroups);    

        % calc gradients from recurrent connections
        ssDx  = dX' * ssAxes;  % gradients (dx) projected into the 3-D state space
        ssDx_len = vecnorm(ssDx, 2, 2);
        ssDx_dir = ssDx ./ ssDx_len;

        % calc gradients from input groups
        ssDxU_len = nan(size(dX_u, 2), size(dX_u, 3));
        ssDxU_dir = nan(size(dX_u, 2), 3, size(dX_u, 3));
        for k = 1:size(dX_u, 3)
            tmp_dX_u = squeeze(dX_u(:, :, k));
            ssDxU = tmp_dX_u' * ssAxes;
    
            ssDxU_len(:, k) = vecnorm(ssDxU, 2, 2);
            ssDxU_dir(:, :, k) = ssDxU ./ ssDxU_len;
        end

        % find locations where each input has the greatest contribution
        [maxLenU, maxLenULoc] = max(ssDxU_len, [], 1);  

        % plot 3 planes to fit the three segments of the state-space trajectory
        segLoc = maxLenULoc + segPlaneOffset;
        for k = 1:length(maxLenULoc)
            ibin = segLoc(k);
            tmpSs = Ss((ibin-5):(ibin+5), :);
            [coeff,score,latent,tsquared,explained,mu] = pca(tmpSs);
    
            tmpDiff = diff(tmpSs, 1, 1);
            tmpScore = tmpDiff * coeff(:, 1:2);
    
            axis1 = coeff(:, 1);
            if mean(tmpScore(:, 1)) < 0
                axis1 = -coeff(:, 1);
            end
    
            axis2 = coeff(:, 2);
            if mean(tmpScore(:, 2)) < 0
                axis2 = -coeff(:, 2);
            end
    
            mag1 = mean(abs(tmpScore(:, 1)), 1);
            mag2 = mean(abs(tmpScore(:, 2)), 1);
    
            corners = Ss(ibin, :)' + segPlaneAxisScalar(1, k) * mag1 * axis1 * [1 -1 -1 1 1] + ...
                                     segPlaneAxisScalar(2, k) * mag2 * axis2 * [1 1 -1 -1 1];
    
            patch(corners(1, :), corners(2, :), corners(3, :), 0.5*[1 1 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end

        % find bin locations where the arrows will be plotted
        numArrowPerSeg = 3; 
        arrowBins = [];
        tmpLoc = [bins(1) maxLenULoc bins(end)] - bins(1) + 1;
        for iloc = 2:length(tmpLoc)
            loc1 = tmpLoc(iloc - 1);
            loc2 = tmpLoc(iloc);

            segLen = cumsumSs(loc2) - cumsumSs(loc1);
            
            for p = 1:numArrowPerSeg
                tmpS = cumsumSs(loc1) + segLen * (p / numArrowPerSeg);
                [~, idx] = min(abs(cumsumSs - tmpS));
                arrowBins = [arrowBins bins(idx)];
            end
        end

        for iarrow = 1:length(arrowBins)
            ibin = arrowBins(iarrow);
            vectorDir    = ssDx_dir(ibin, :);
            vectorLength = ssDx_len(ibin) * quiverScale;

            posStart = Ss(ibin, :);
            posEnd = Ss(ibin, :) + vectorDir * vectorLength;
            if vectorLength > 0.5*arrowLength  % plot arrow stick
                posEnd = Ss(ibin, :) + vectorDir * (vectorLength - 0.5*arrowLength);  % make stick of arrow a bit shorter so it won't go out of the arrow head
                plot3([posStart(1) posEnd(1)], [posStart(2) posEnd(2)], [posStart(3) posEnd(3)], ...
                      'Color', [106,81,163]/255, 'LineWidth', 1);

                posEnd = Ss(ibin, :) + vectorDir * vectorLength;
                draw_arrow_3d(posEnd, vectorDir, 'faceColor', [106,81,163]/255, 'arrowSize', arrowSize, 'centerIsTip', false);
            end
        end


        % plot gradients from input groups
        tmp_dX_u = squeeze(sum(dX_u, 3));  % all input groups combined
        ssDxU = tmp_dX_u' * ssAxes;

        ssDxU_len = vecnorm(ssDxU, 2, 2);
        ssDxU_dir = ssDxU ./ ssDxU_len;

        for iarrow = 1:length(arrowBins)
            ibin = arrowBins(iarrow);
            vectorDir    = ssDxU_dir(ibin, :);
            vectorLength = ssDxU_len(ibin) * quiverScale;

            posStart = Ss(ibin, :);
            posEnd = Ss(ibin, :) + vectorDir * vectorLength;
            if vectorLength > 0.5*arrowLength  % plot arrow stick
                posEnd = Ss(ibin, :) + vectorDir * (vectorLength - 0.5*arrowLength);
                plot3([posStart(1) posEnd(1)], [posStart(2) posEnd(2)], [posStart(3) posEnd(3)], ...
                      'Color', [49,163,84]/255, 'LineWidth', 1);
                
                posEnd = Ss(ibin, :) + vectorDir * vectorLength;
                draw_arrow_3d(posEnd, vectorDir, 'faceColor', [49,163,84]/255, 'arrowSize', arrowSize, 'centerIsTip', false);
            end
        end


    daspect([1 1 1]); 
    axis([minX maxX minY maxY flowZ maxZ]);
    view(flowFigViewAngle);
    xlabel('Dim 1'); ylabel('Dim 2'); zlabel('Dim 3'); 


    %% draw X- and Y-axis manually    
    set(gca, 'visible', 'off');
    set(gcf, 'color', [1 1 1]);
    
    % draw X-axis
    axisParams.axisLabel = sprintf('rPC_{%d}', ssAxisIDs(1));
    axisParams.ticks = [];  % [-1.2 -0.6];
    axisParams.lineWidth = 2;
    axisParams.axisLabelOffsetFactor = 0.4;

    draw_axis_3d(maxX + [-1 0], [minY flowZ], 'back', axisParams);
    
    % draw Y-axis
    axisParams.axisLabel = sprintf('rPC_{%d}', ssAxisIDs(2));
    draw_axis_3d(minY + [0 1], [flowZ maxX], 'right', axisParams);
    
    % draw Z-axis
    axisParams.axisLabel = sprintf('rPC_{%d}', ssAxisIDs(3));
    draw_axis_3d(flowZ + [0 1], [maxX minY], 'top', axisParams);

    
    %% cleanup
    rmpath('.\util');

end




function [dX, dX_u] = calc_dx(X, net, U, inputGroups)
    [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);

    rec_trans_fun = net.layers(2).transFun;
    R = rec_trans_fun(X);
    
    
    dt_o_tau = net.dt / net.tau;

    dX = dt_o_tau * ( -X + n_Wrr_n*R + repmat(n_bx_1, 1, size(X, 2)) );

    if exist('U', 'var')
        numInputGroup = length(inputGroups);
        for k = 1:numInputGroup
            inputDims = inputGroups{k};
            dX_u(:, :, k) = dt_o_tau * n_Wru_v(:, inputDims) * U(inputDims, :);
        end
    else
        dX_u = [];
    end
end
