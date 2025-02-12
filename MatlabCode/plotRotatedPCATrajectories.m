function plotRotatedPCATrajectories(rs, mean_ev, xax_labelsms,colors, savePath)
    num_comp = 6;
    ylimset = max(abs(rs), [], 'all');
    mevms = (mean_ev([7,12,10]) * 1000) - 200;
    ylab = 'Firing Rate (a.u.)';
    xlab = 'Time (msec)';
    
    for i = 1:num_comp
        f1 = figure('Position',[-1919 41 1920 963]);
        colororder(colors(:,1:3))
        plot(xax_labelsms, rs(:,:,i), 'LineWidth', 5);
        hold on;
        plot(mevms, -ones(size(mevms)), 'pentagram', 'Color', 'k', 'LineWidth', 8);
        axis tight;
        set(f1.Children, 'box', 'off', 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Arial', 'YLim', [-ylimset ylimset], 'TickDir', 'out');
        ylabel(ylab);
        xlabel(xlab);
        % Save figures
        saveas(f1, [savePath 'rotPCA_Actual_comp_' num2str(i) '.emf'], 'meta');
        saveas(f1, [savePath 'rotPCA_Actual_comp_' num2str(i) '.png']);
        close all;
    end
end