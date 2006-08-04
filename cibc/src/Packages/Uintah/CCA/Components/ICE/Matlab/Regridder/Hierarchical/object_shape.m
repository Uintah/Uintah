function object = object_shape(type,varargin)
%OBJECT_SHAPE Create object handle for a time-stepping test.
%
%   OBJECT = OBJECT_SHAPE('BOX',[x1 y1 x2 y2]) creates a 2D box. Lower left
%       corner is (x1,y1), upper right is (x2,y2).
%   OBJECT = OBJECT_SHAPE('SPHERE',[x y],RADIUS) creates a sphere of radius
%       RADIUS centered at (x,y) (a 2D coordinate).
%   
%   See also OBJECT_MOVEMENT, TEST_CASE, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created

object.type      = type;

switch (type)
case 'box',                                                     % Create a box    
    object.center    = varargin{1};                             % Box center coordinate
    object.size      = varargin{2};                             % Box size
case 'ball',                                                    % Create a ball
    object.center    = varargin{1};                             % Ball center
    object.radius    = varargin{2};                             % Ball radius
end
