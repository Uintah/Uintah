function d = object_movement(type,varargin)
%OBJECT_MOVEMENT Create object handle for a time-stepping test.
%
%   D = OBJECT_MOVEMENT('LINE',[x1 y1 x2 y2]) creates a 2D box. Lower left
%       corner is (x1,y1), upper right is (x2,y2).
%   D = OBJECT_MOVEMENT('CIRC',[x y],RADIUS,THETA0,DTHETA) creates a movement along a sphere of radius
%       RADIUS centered at (x,y) (a 2D coordinate). THETA0 is the initial angle of the
%       object w.r.t. the sphere it rotates around, and DTHETA the relative angular velocity
%       (per one time-step).
%   
%   See also OBJECT_CREATE, TEST_CASE, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created

d.type      = type;

switch (type)
case 'line',                                                        % Create a linear movement
    d.direction = varargin{1};                                      % Direction of movement in one time-step [cells]
case 'circ',                                                        % Create a circular movement
    d.center    = varargin{1};                                      % Center of sphere around which we move
    d.radius    = varargin{2};                                      % Radius of sphere around which we move
    d.t0        = varargin{3};                                      % Initial angle
    d.dtheta    = varargin{4}*acos(1./d.radius);                    % Relative anglular velocity
case 'expd',                                                        % Expanding ball (use with object.type == 'ball')
    d.dradius   = varargin{1};                                      % Rate of radius growth
end
