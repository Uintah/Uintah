function tin = test_case(title)
%TEST_CASE Prepare parameters structure for TEST_MOVEMENT.
%
%   TIN = TEST_CASE(TITLE) creates the parameter structure for the test case specified by
%   the title string TITLE (objects types and their movements with time).
%   
%   See also CREATE_CLUSTER, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created

%%%%% Set parameters
tin.plot_flag   = 1;                                            % Flag for generating plots of the boxes and cells
tin.num_tsteps  = 100;                                           % Number of timesteps to be performed
tin.delay       = 0.;                                           % Delay time [secs] between plots of consequtive timesteps
tin.regrid_max  = 10000;                                        % Maximum number of timesteps allowed between re-gridding
tin.init_t      = 0;                                            % Initial time [time_unit]
tin.dt          = 1;                                            % Delta_t [time_unit]
tin.dilate_create   = 5;                                        % #safety layers when we create a set of boxes (5-10)
tin.dilate_check    = 2;                                        % #safety layers when we check whether error is out of boxes (1-2)

tin.title       = title;                                        % Title string of this test case
tin.num_objects = 0;                                            % Counter of synthetic objects in this test case
tin.object      = [];                                           % Object array; each cell decribes the object and its movement

%%%%% Add objects - initial locations and movements
switch (lower(title))
case {'ball_x'}
    tin.domain_size = [100 100];                                % Size of the entire domain we live in [cells]
    o   = create_object('ball',[-10 30],30);                   % Ball of size 20 at (100,200)
    d   = create_movement('line',[1 0]);                        % Moving along the x direction
    tin = add_object(tin,o,d);                                  % Add this object
case {'ball_mx'}
    tin.domain_size = [400 400];                                % Size of the entire domain we live in [cells]
    o   = create_object('ball',[100 200],20);                   % Ball of size 20 at (100,200)
    d   = create_movement('line',[-1 0]);                       % Moving along negative x direction
    tin = add_object(tin,o,d);                                  % Add this object
case {'ball_y'}
    tin.domain_size = [400 400];                                % Size of the entire domain we live in [cells]
    o   = create_object('ball',[100 200],20);                   % Ball of size 20 at (100,200)
    d   = create_movement('line',[0 1]);                        % Moving along the y direction
    tin = add_object(tin,o,d);                                  % Add this object
case {'ball_my'}
    tin.domain_size = [400 400];                                % Size of the entire domain we live in [cells]
    o   = create_object('ball',[100 200],20);                   % Ball of size 20 at (100,200)
    d   = create_movement('line',[0 -1]);                       % Moving along negative y direction
    tin = add_object(tin,o,d);                                  % Add this object
case {'ball_diag'}
    tin.domain_size = [100 100];                                % Size of the entire domain we live in [cells]
    o   = create_object('ball',[20 20],30);                   % Ball of size 20 at (100,200)
    d   = create_movement('line',[1 1]);                        % Moving along a diagonal line of 45 degrees
    tin = add_object(tin,o,d);                                  % Add this object
case {'ball_circ'}
    tin.domain_size = [200 200];                                % Size of the entire domain we live in [cells]
    o   = create_object('ball',[],30);                          % Ball of size 30
    d   = create_movement('circ',[100 100],30,0,0.1);           % Circular motion in a circle of size 30, center (100,100).
    tin = add_object(tin,o,d);                                  % Initial angle: 0; relative angular step = 0.1. Add this object
otherwise
    error(sprintf('Unknown title ''%s'' for test_case',title)); % Unknown test case, exit
end
