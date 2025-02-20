function trigger_histogram(triggerFRW,Monk, CodeDir)

FRWc = cell(max(triggerFRW.infoint8All(:,end)),2);
for unit = 1:max(triggerFRW.infoint8All(:,end))
    FRW = triggerFRW.FRsAll;
    FRWi = triggerFRW.infoint8All;
    FRWunitmask = FRWi(:,end)==unit;
    FRW = FRW(FRWunitmask,:);
    FRWi = FRWi(FRWunitmask,:);
    w = FRW(:,2);
    FRWc{unit,2} = w./max(abs(w));
    FRWc{unit,1} = FRW(:,1);
end
FRW = cell2mat(FRWc);
f1 = figure();
h= histogram2(FRW(:,1),FRW(:,2),linspace(0,100,101),linspace(-1,1,101),'FaceColor','flat');
xlabel('FR')
ylabel('weight')
zlabel('count')
h.BinCounts;
h.YBinEdges;

FRW = struct('Counts',h.BinCounts,'FRbinedges',h.XBinEdges,'Wbinedges',h.YBinEdges,...
    'FR',h.XBinEdges(1:end-1)+diff(h.XBinEdges(1:2)),'W',h.YBinEdges(1:end-1)+diff(h.YBinEdges(1:2)));

save(sprintf('%s\\Data\\Monk%s_FR_W_histdata.mat', CodeDir, Monk),'FRW')
saveas(f1, sprintf('%s\\Figures\\Monk_%s\\Trigger_Histogram.png', CodeDir, Monk))

close all

end