%Used to plot surfaces (actually imagesc) on a subplot figure, and in their own figures.
function surfaces_3x4(mat_pre, k,savePath)
    mkdir([savePath '\png\']);
    mkdir([savePath '\emf\']);
    f1 = figure('Position', [-1919 41 1920 963]);
    maxval = max(mat_pre, [],'all');
    minval = min(mat_pre, [],'all');
    for i = 1:size(mat_pre,3)%for all input groups
        for j = 1:size(mat_pre,4)%for each epoch
            mat_ = mat_pre(:,:,i,j);
            for jj = 1:2 %for subplot or full figure
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

                saveas(f2, sprintf('%s\\emf\\%02ims_inputG%i_epoch%i.emf', savePath, k,i,j), 'meta');
                saveas(f2, sprintf('%s\\png\\%02ims_inputG%i_epoch%i.png', savePath, k,i,j));
                close(f2)
            end
            end
            pause(.1)
        end
        
    end
    pause(1)
    saveas(f1, sprintf('%s\\emf\\%02ims_subplots.emf', savePath, k), 'meta');
    saveas(f1, sprintf('%s\\png\\%02ims_subplots.png', savePath, k));

    close all

end