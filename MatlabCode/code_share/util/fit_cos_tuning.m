%% Filter cosine tuning curve
%    fit cosine function: 
%        fr = b0 + MD * cos(PD - ang)
%
%    input:     fr              a column vector of firing rates
%               ang             a column vector of angles in degrees
%
%    output:    b0              baseline firing rate
%               MD              modulation depth
%               PD              preferred direction
%
% Created by Hongwei Mao, 10/14/2021
% ----- ----- -----
% Solve the problem with multiple linear regression using 'regress'.
% This approach runs faster than fitting a sine curve.
%
% Hongwei Mao, 8/7/2025

function [PD, MD, b0, R2] = fit_cos_tuning(FR, ang)

    % % % for i = 1:size(FR, 2)
    % % %     fr = FR(:, i);
    % % % 
    % % %     % fr = a1 * sin(b1 * ang + c1)
    % % %     [fitObj, gof] = fit(deg2rad(ang), fr - mean(fr), 'sin1', 'Lower', [0 1 0], ...
    % % %                                            'Upper', [Inf 1 2*pi]);
    % % %     GoF(i) = gof;
    % % % 
    % % %     fitCoeff = coeffvalues(fitObj);
    % % % 
    % % %     % % % frFit = fitCoeff(1) * sin(fitCoeff(2) * deg2rad(ang) + fitCoeff(3));
    % % % 
    % % %     % fr = b0 + a1 * sind(ang + c1)
    % % %     %    = b0 + a1 * cosd(ang + c1 - 90)
    % % %     %    = b0 + a1 * cosd(ang - (-c1 + 90))
    % % %     % fr = b0 + MD * cosd(ang - PD)
    % % %     b0(1, i) = mean(fr);
    % % %     MD(1, i) = fitCoeff(1);
    % % %     PD(1, i) = -rad2deg(fitCoeff(3)) + 90;
    % % % 
    % % %     % % % frFit = b0 + MD * cosd(ang - PD);
    % % %     % % % 
    % % %     % % % figure(1); clf; hold on; 
    % % %     % % % plot(fr, '-k'); plot(frFit, '--rx');
    % % % end


    % fr = b0 + b1*cosd(ang) + b2*sind(ang)
    X = [ones(size(FR, 1), 1) cosd(ang) sind(ang)];
    for i = 1:size(FR, 2)
        fr = FR(:, i);

        % fr = b0 + b1*cosd(ang) + b2*sind(ang)
        %    = b0 + MD*cosd(PD)cosd(ang) + MD*sind(PD)sind(ang)
        %    = b0 + MD*cosd(PD - ang)
        % b1 = MD*cosd(PD), b2 = MD*sind(PD)
        % MD = norm([b1 b2])
        [b, bint, r, rint, stats] = regress(fr, X);

        b0(1, i) = b(1);
        [theta, rho] = cart2pol(b(2), b(3)); 
        PD(1, i) = rad2deg(theta);
        MD(1, i) = rho;
        R2(1, i) = stats(1);
    end
end
