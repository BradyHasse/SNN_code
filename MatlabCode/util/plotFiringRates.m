function plotFiringRates(xax_labelsms, histo_data, ylimset, mevms, ylab, xlab, savePath, AvsP,colors, MI, Mkind)
    mkdir([savePath 'png\']);
    mkdir([savePath 'emf\']);
    if nargin == 11
        digits = regexp(MI(1,Mkind), '\d+', 'match', 'once');
        ex_unit = int32(str2double(digits));
        histo_data = histo_data(:,:,ex_unit + 1);
        unit_nums = ex_unit +1;
    else
        unit_nums = 1:size(histo_data, 3);
    end
    for j = 1:size(histo_data, 3)
        histoo = squeeze(histo_data(:,:,j));
        f1 = figure('Position',[-1919 41 1920 963]);
        colororder(colors(:,1:3))
        plot(xax_labelsms, histoo, 'LineWidth', 5);
        hold on;
        plot(mevms, zeros(size(mevms)), 'pentagram', 'Color', 'k', 'LineWidth', 8);
        axis tight;
        set(f1.Children, 'box', 'off', 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Arial', 'YLim', ylimset(unit_nums(j),:)+[0,.0001], 'TickDir', 'out');
        ylabel(ylab);
        xlabel(xlab);
        
        % Save figures
        saveas(f1, sprintf('%s\\emf\\histo%s_unit_%02i.emf', savePath, AvsP, unit_nums(j)+1), 'meta');
        saveas(f1, sprintf('%s\\png\\histo%s_unit_%02i.png', savePath, AvsP, unit_nums(j)+1));
        % pause(1)
        close all;
    end
end