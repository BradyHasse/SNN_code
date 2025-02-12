function plotSTA(sta, threshold, XTick, XTickL, xRange, yLabel, savePath, Monk, titleSuffix)
    f1 = figure('Position', [-1918 42 1409 954]);
    plot(xRange, mean(sta), 'k', 'LineWidth', 5);
    hold on
    yline(threshold, ':', 'LineWidth', 3)
    set(f1.Children, 'box', 'off', 'XTickLabel', XTickL, 'XTick', XTick, 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Arial');
    ylabel(yLabel);
    xlabel('Time (ms)');
    saveas(f1, [savePath, 'Monk', Monk, '_STA_', titleSuffix, '.emf'], 'meta');
    saveas(f1, [savePath, 'Monk', Monk, '_STA_', titleSuffix, '.png']);
    close all;
end