%Used to plot firing rates with patches showing when each analysis period is

function plotRatesPatches(xax_labelsms, histo_data, ylimset, mevms, ylab, xlab, savePath, AvsP,colors, STA)
    mkdir([savePath 'png\']);
    mkdir([savePath 'emf\']);
    
    centers = STA.centers;
    centersms = (squeeze(mean(centers,[1,2]))/10)-200;

    for j = 1:size(histo_data, 3)
        histoo = squeeze(histo_data(:,:,j));
        PL = [min(histoo,[],'all') max(histoo,[],'all')];
        PL = ((PL-mean(PL))*1.25)+mean(PL);

        f1 = figure('Position',[-1919 41 1920 963]);
        colororder(colors(:,1:3))
        plot(xax_labelsms, histoo, 'LineWidth', 5);
        hold on;
        plot(mevms, zeros(size(mevms)), 'pentagram', 'Color', 'k', 'LineWidth', 8);
        
        for i = 1:3
            p = patch(centersms(i)+[60,60,-60,-60], [PL(1) PL(2) PL(2) PL(1)]+0.01*diff(PL)-(0.01*(i-1))*diff(PL),'r');
            set(p,'FaceColor',[.5,1-(.25*i),.25*i],'EdgeColor','none','FaceAlpha',.3)
        end

        axis tight;
        set(f1.Children, 'box', 'off', 'LineWidth', 3, 'FontSize', 32, 'fontname', 'Arial', 'YLim', ylimset(j,:)+[0,.0001], 'TickDir', 'out');
        ylabel(ylab);
        xlabel(xlab);
        
        % Save figures
        saveas(f1, sprintf('%s\\emf\\histo%s_unit_%02i_patch.emf', savePath, AvsP, j), 'meta');
        saveas(f1, sprintf('%s\\png\\histo%s_unit_%02i_patch.png', savePath, AvsP, j));
%         pause(1)
        close all;
    end
end










