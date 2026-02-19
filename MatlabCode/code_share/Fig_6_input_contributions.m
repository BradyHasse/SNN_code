
subjectName = 'MonkeyC';        % monkeyC  monkeyN
%% load data
addpath('.\util\');
dataDir  = '.\data\';

if ~exist('subjectName', 'var')
    subjectName = 'MonkeyC';  % MonkeyC MonkeyN
end
dataFileName = sprintf('%s_SNN_Data.mat', subjectName);
Data = importdata([dataDir dataFileName]);
colors = importdata('target_colormap.mat');  % colors correspond to condition [10:16 1:9]

weights_u = Data.weights_u;
STA = Data.STA;
causal_in = Data.causal_in;

sta = STA.sta_pot;
threshold = STA.threshold;
%% Make 20ms STA figure

XTick = -20:2.5:0;
XTickL = cellstr(string(XTick));

f1 = figure;
plot((-199:0)/10, mean(sta), 'k', 'LineWidth', 5);
hold on
%     plot(xRange, sta, 'LineWidth', 1);
yline(threshold, ':', 'LineWidth', 3)
set(f1.Children, 'box', 'off', 'XTickLabel', XTickL, 'XTick', XTick, 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Helvetica');
ylabel('Membrane Potential (a.u.)');
xlabel('Time (ms)');

set(gca, 'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 20, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 2, 4, 4]);

% savePath = sprintf('%s\\Figures\\Monk_%s\\STA', CodeDir, Monk);
% saveas(f1, sprintf('%s\\emf\\STA_%s.emf', savePath, titleSuffix), 'meta');
% saveas(f1, sprintf('%s\\png\\STA_%s.png', savePath, titleSuffix));
% close all;


%% Make 5ms STA figure

XTick = -5:1:0;
XTickL = cellstr(string(XTick));

f1 = figure;

plot((-49:0)/10, mean(sta(:, end-49:end)), 'k', 'LineWidth', 5);
hold on
%     plot(xRange, sta, 'LineWidth', 1);
yline(threshold, ':', 'LineWidth', 3)
set(f1.Children, 'box', 'off', 'XTickLabel', XTickL, 'XTick', XTick, 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Helvetica');
ylabel('Membrane Potential (a.u.)');
xlabel('Time (ms)');

set(gca, 'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 20, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 2, 4, 4]);
pause(.5)
% savePath = sprintf('%s\\Figures\\Monk_%s\\STA', CodeDir, Monk);
% saveas(f1, sprintf('%s\\emf\\STA_%s.emf', savePath, titleSuffix), 'meta');
% saveas(f1, sprintf('%s\\png\\STA_%s.png', savePath, titleSuffix));
% close all;

%% surface plots for causal inputs
% savePath = sprintf('%s\\Figures\\Monk_%s\\SurfacePlots', CodeDir, Monk);
fields = fieldnames(causal_in);
causal_in2 = [];
for g = 1:length(fields)
    causal_in2 = cat(5,causal_in2,causal_in.(fields{g}));
end
causal_in2 = permute(causal_in2,[2,3,1,4,5]);
SCI2 = size(causal_in2);%16x90x2x4x3 - target x input x range (trigger or build-up) x input gorup x analysis epoch

weights2 = permute(weights_u,[3,2,4,1,5]);
weights2 = repmat(weights2, SCI2(1),1,SCI2(3),1,SCI2(5));

causal_in2w = weights2.*causal_in2;%with weights, membrane potential
causal_in2(isnan(causal_in2))=0;

causal_smoothw = circularsmooth(causal_in2w,10);
causal_smooth =  circularsmooth(causal_in2 ,10);


%Plot individual imagesc and as subplots
for k = 1:size(causal_smoothw,3)%for each window length (trigger and build-up)
    mat_pre = squeeze(causal_smoothw(:,:,k,:,:));
        % mkdir([savePath '\png\']);
    % mkdir([savePath '\emf\']);
    f1 = figure;
    pause (1)
    maxval = max(mat_pre, [],'all');
    minval = min(mat_pre, [],'all');
    for i = 1:size(mat_pre,3)%for all input groups
        for j = 1:size(mat_pre,4)%for each epoch
            mat_ = mat_pre(:,:,i,j);
            subplot(3,4,(i+(3-j)*4))
            colormap("turbo")
            imagesc(imresize(mat_, 4))
            axis image
            set(gca,'YDir','normal')
            xlabel('input prefered direction (degrees)')
            ylabel('target direction (degrees)')
            clim([minval, maxval])
                
            if i ==4
                colorbar
            end
        end
        pause(.1)
    end
        
    % saveas(f1, sprintf('%s\\emf\\%02ims_subplots.emf', savePath, k), 'meta');
    % saveas(f1, sprintf('%s\\png\\%02ims_subplots.png', savePath, k));
    % 
    % close all
end
rmpath('.\util\');