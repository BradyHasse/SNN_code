function plotSTA(sta, threshold, XTick, XTickL, xRange, yLabel, savePath, titleSuffix)
    mkdir([savePath '\png\']);
    mkdir([savePath '\emf\']);
    f1 = figure('Position', [-1918 42 1409 954]);
    plot(xRange, mean(sta), 'k', 'LineWidth', 5);
    hold on
%     plot(xRange, sta, 'LineWidth', 1);
    yline(threshold, ':', 'LineWidth', 3)
    set(f1.Children, 'box', 'off', 'XTickLabel', XTickL, 'XTick', XTick, 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Arial');
    ylabel(yLabel);
    xlabel('Time (ms)');

    saveas(f1, sprintf('%s\\emf\\STA_%s.emf', savePath, titleSuffix), 'meta');
    saveas(f1, sprintf('%s\\png\\STA_%s.png', savePath, titleSuffix));
    close all;
end