function o = create_object(type,varargin)
%CREATE_OBJECT Create object handle for a time-stepping test.
%
%   O = CREATE_OBJECT('BOX',[x1 y1 x2 y2]) creates a 2D box. Lower left
%       corner is (x1,y1), upper right is (x2,y2).
%   O = CREATE_OBJECT('SPHERE',[x y],RADIUS) creates a sphere of radius
%       RADIUS centered at (x,y) (a 2D coordinate).
%   
%   See also CREATE_MOVEMENT, TEST_CASE, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created

o.type      = type;

switch (type)
case 'box',                                                     % Create a box
    o.coords    = varargin{1};                                  % Box coordinates
case 'ball',                                                    % Create a ball
    o.center    = varargin{1};                                  % Ball center
    o.radius    = varargin{2};                                  % Ball radius
end
