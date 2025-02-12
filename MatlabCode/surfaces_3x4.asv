%need to add in filename %s\\Figures\\Monk_%s\\Monk%s_WeightedContribution_%02ims_ig%i_e%i
%to allow for its use across seperate use cases

function surfaces_3x4(mat_pre, k,Monk, FileName)
f1 = figure('Position', [-1919 41 1920 963]);
    maxval = max(mat_pre, [],'all');
    minval = min(mat_pre, [],'all');
    for i = 1:size(mat_pre,3)%for all input groups
        for j = 1:size(mat_pre,4)%for each epoch
            mat_ = mat_pre(:,:,i,j);
            for jj = 1:2
            if jj == 1
            subplot(3,4,(i+(3-j)*4))
            else
                f2 = figure('Position', [-1919 41 1920 963]);
            end
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
            if jj == 2
                saveas(f2, sprintf('%s\\Figures\\Monk_%s\\Monk%s_WeightedContribution_%02ims_ig%i_e%i.emf', CodeDir, Monk,Monk, k,i,j), 'meta');
                saveas(f2,[CodeDir '\Figures\' 'Monk_' Monk '\' 'Monk' Monk '_' sprintf('WeightedContribution_%02ims_ig%i_e%i', k,i,j) '.png'])
                close(f2)
            end
            end
            pause(.1)
        end
        
    end
    pause(1)
    saveas(f1,[CodeDir '\Figures\' 'Monk' Monk '_' sprintf('WeightedContribution_%02ims', k) '.emf'],'meta')
    saveas(f1,[CodeDir '\Figures\' 'Monk' Monk '_' sprintf('WeightedContribution_%02ims', k) '.png'])
    close all



end