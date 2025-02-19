function plotRotatedPCATrajectories2d(rs,colors, savePath)
    mkdir([savePath 'png\']);
    mkdir([savePath 'emf\']);
    num_comp = 6;
    ylimset = max(abs(rs),[],'all');

    maxrs = squeeze(max(abs(rs),[],2));%first find the paired PCs
    [~,ind] = max(maxrs);
    pairedPC = zeros(num_comp/2,2);
    avlbPC = true(num_comp,1);
    for i = 1:size(pairedPC,1)
        [valind, ind2chk] = min(ind);
        avlbPC(ind2chk) = 0;
        ind(ind2chk) = NaN;
        [~,ind2] = min(abs(valind-ind));
        pairedPC(i,:) = [ind2chk, ind2];
        ind(ind2) = NaN;
        avlbPC(ind2) = 0;
    end
    
    MT = zeros(size(rs(:,:,1)));
    dimorder = [3,1,2;2,3,1;1,2,3];
    lineg = [1,1,-1,-1,1;1,-1,-1,1,1;0,0,0,0,0]*ylimset;
    views = [200,-5;0,0;-30,15];
    
    for i = 1:size(pairedPC,1)
        prs = cat(3,rs(:,:,pairedPC(i,:)),MT);
        prs = prs(:,:,dimorder(i,:));
        linegp = lineg(dimorder(i,:),:);
        f1 = figure('Position',[-1919 41 1920 963]);
        colororder(colors(:,1:3))
        plot3(prs(:,:,1),prs(:,:,2),prs(:,:,3),'linewidth',3)
        hold on
        plot3(linegp(1,:),linegp(2,:),linegp(3,:),'Color','k','linewidth',3)
        axis image off ij;
        view(views(i,:))
        set(f1,'Color','None')

        saveas(f1, sprintf('%s\\emf\\rotPCA_Paired_2D_%01i.emf', savePath, i), 'meta');
        saveas(f1, sprintf('%s\\png\\rotPCA_Paired_2D_%01i.png', savePath, i));

        close all
    end
end



