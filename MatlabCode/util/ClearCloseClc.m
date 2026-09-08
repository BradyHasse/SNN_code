function ClearCloseClc()
evalin( 'base', 'clearvars -except CodeDir colors MI Mkind Monk TargetDir_N TargetDir_N2 PD_N PD_N2' )
close all
clc
end