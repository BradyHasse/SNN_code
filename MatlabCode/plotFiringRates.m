function plotFiringRates(xax_labelsms, histo_data, ylimset, mevms, ylab, xlab, savePath, AvsP,colors)
    mkdir([savePath 'png\']);
    mkdir([savePath 'emf\']);
    
    for j = 1:size(histo_data, 3)
        histoo = squeeze(histo_data(:,:,j));
        f1 = figure('Position',[-1919 41 1920 963]);
        colororder(colors(:,1:3))
        plot(xax_labelsms, histoo, 'LineWidth', 5);
        hold on;
        plot(mevms, zeros(size(mevms)), 'pentagram', 'Color', 'k', 'LineWidth', 8);
        axis tight;
        set(f1.Children, 'box', 'off', 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Arial', 'YLim', ylimset(j,:), 'TickDir', 'out');
        ylabel(ylab);
        xlabel(xlab);
        
        % Save figures
        saveas(f1, sprintf('%s\\emf\\histo%s_unit_%02i.emf', savePath, AvsP, j), 'meta');
        saveas(f1, sprintf('%s\\png\\histo%s_unit_%02i.png', savePath, AvsP, j));
        % pause(1)
        close all;
    end
end