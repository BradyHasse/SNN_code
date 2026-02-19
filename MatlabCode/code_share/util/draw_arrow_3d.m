%% Draw an arrow
%
%   Inputs:
%           center          [x, y, z] position of center
%           mainAxis        [x1, y1, z1] defines direction of the arrow
%
%   Optional inputs:
%           arrowSize       default = 6; similar to how MarkerSize works
%           lineColor,  lineWidth
%           faceColor,  faceAlpha
%
% Created by Hongwei Mao, 11/5/2021

function draw_arrow_3d(center, mainAxis, varargin)
    if nargin < 1
        center = [0 0 0];
    elseif nargin < 2
        mainAxis = [0 0 1];
    end
    
    if isempty(mainAxis)
        mainAxis = [0 0 1];
    end
    
    % parse additional options
    m = 50;
    arrowSize = 6;          % Matlab default 'MarkerSize' = 6 pt
    faceColor = [0 0 1];
    centerIsTip = false;    % default: 'center' is center of base of arrow head
    
    numOptions = size(varargin, 2);
    for i = 1:2:numOptions
        switch varargin{i}
            case 'arrowSize'
                arrowSize = varargin{i+1};
            case 'faceColor'
                faceColor = varargin{i+1};
            case 'centerIsTip'
                centerIsTip = true;
            otherwise
                assert(false, ['Error in draw_arrow_3d.m: don''t recognize ' varargin{i} '.']);
        end
    end
    
    
    % arrow size scale by arrowSize
    %   similarly to what 'MarkerSize' does for plot function
    sizeFactor = arrowSize * 0.12 / 6;  % length of arrow head
    
    
    % create 3-D arrow centered at [0 0 0], pointing in [0 0 1] direction
    theta = (0:m)*360/m;
    %               center(0, 0, 0); base of cone;  tip of cone
    x = center(1) + sizeFactor * [zeros(1, m+1); cosd(theta)/4; zeros(1, m+1)];
    y = center(2) + sizeFactor * [zeros(1, m+1); sind(theta)/4; zeros(1, m+1)];
    if centerIsTip
        z = center(3) - sizeFactor + sizeFactor * [zeros(1, m+1); zeros(1, m+1); ones(1, m+1)];  % center is tip of arrow head
    else
        z = center(3) + sizeFactor * [zeros(1, m+1); zeros(1, m+1); ones(1, m+1)];  % center is base of arrow head
    end

    % draw arrow in default direction
    s = surface(x, y, z, 'FaceColor', faceColor, 'EdgeColor', 'none');
    
    % rotate arrow to point in the direction defined by mainAxis
    defaultAxis = [0 0 1];
    rotAxis = cross(defaultAxis, mainAxis); % rotation axis is perpendicular to 
                                            % arrow axis before and after rotation
    alpha = atan2d(norm(cross(mainAxis, defaultAxis)), dot(mainAxis, defaultAxis));
    if alpha ~= 0
        rotate(s, rotAxis, alpha, center);
    end
    
end
