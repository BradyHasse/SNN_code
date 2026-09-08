function [colors, MI, Mkind, TargetDir_N, PD_N, PD_N2] = setupEnvironment(Monk, CodeDir)
    mkdir([CodeDir '\Figures']);
    colors = importdata([CodeDir,'\Data\rgbColorMap.mat']);
    MI = {'U58', 'U47'; '28-06-2024-15-09-50', '03-07-2024-13-30-06'};
    if Monk == 'C'
        Mkind = 1;
    else
        Mkind = 2;
    end
    TargetDir_N = linspace(0, 2*pi*3-((2*pi)/16), 16*3);
    PD_N = linspace(0, 2*pi*3-((2*pi)/90), 90*3);
    PD_N2 = linspace(0, 2*pi-((2*pi)/90), 90);
end
