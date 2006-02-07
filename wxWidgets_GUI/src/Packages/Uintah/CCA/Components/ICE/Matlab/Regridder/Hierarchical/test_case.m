function tin = test_case(title)
%TEST_CASE Prepare parameters structure for TEST_MOVEMENT.
%   TIN = TEST_CASE(TITLE) creates the parameter structure for the test case specified by
%   the title string TITLE (objects types and their movements with time).
%   
%   See also CREATE_CLUSTER, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created

%%%%% Set global parameters
tin.init_t      = 1;                                            % Initial time [time_unit]
tin.num_tsteps  = 30;                                           % Number of timesteps to be performed
tin.dt          = 1;                                            % Delta_t [time_unit]

tin.safe_bdry   = 1;                                            % #safety cell layers of L{k} from a boundary of L{k+1} patch
tin.safe_create = 1;%1;                                         % #safety cell layers when we create patches (2-5)
tin.safe_delete = 2;%1;                                         % #safety cell layers when we delete patches (5-10)

tin.num_procs   = 4;                                            % Number of processors
tin.min_side    = 5;                                            % Minimum box side length allowed (in all directions)
tin.max_volume  = 100;                                          % Maximum box volume allowed. Need to be >= (2*min_side)^dim.

tin.plot        = 1;%1;                                         % Flag for generating plots of the boxes and cells
tin.print       = 1; %1;                                        % Flag for generating plots of the boxes and cells
tin.delay       = 0.0;                                          % Delay time [secs] between plots of consequtive timesteps
tin.tsteps_save = [3 7 12 15];                                    % Timesteps for which we save a snapshot of the composite grid
    
%%%%% Define the geometry and a hierarchy of levels
tin.domain      = [32 32];                                      % Size of domain [meter]
tin.max_levels  = 4;                                            % Maximum number of levels
tin.cell_num    = [17 17];                                      % Number of cells at coarsest level [cell]
tin.patch_size  = [8 8];                                        % Number of cells in a patch at the coarsest level [cell], possibly except the last patch in every row

tin.rat_cell    = [ ...                                         % Grid refinement ratio between L{k}->L{k+1}, k=o..max_levels+o-2
        [2 2]; ...
        [2 2]; ...
        [2 2]; ...
    ];

tin.rat_patch   = [ ...                                         % Lattice refinement ratio between L{k}->L{k+1}, k=o..max_levels+o-2
        [4 4]; ...
        [2 2]; ...
        [2 2]; ...
    ];

%%%%% Add objects - initial locations and movements
tin.title       = title;                                        % Title string of this test case
tin.num_objects = 0;                                            % Counter of synthetic objects in this test case
tin.object      = [];                                           % Object array; each cell decribes the object and its movement
switch (lower(title))
case {'ball_x'}
    ob  = object_shape('ball',[-4 8],[4 4]);                    % Ball of size 4
    d   = object_movement('line',[1 0]);                        % Moving along positive x direction
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_mx'}
    ob  = object_shape('ball',[24 8],[4 4]);                    % Ball of size 4
    d   = object_movement('line',[-1 0]);                       % Moving along negative x direction
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_y'}
    ob  = object_shape('ball',[8 -4],[4 4]);                    % Ball of size 4
    d   = object_movement('line',[0 1]);                        % Moving along positive y direction
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_my'}
    ob  = object_shape('ball',[8 24],[4 4]);                    % Ball of size 4
    d   = object_movement('line',[0 -1]);                       % Moving along negative y direction
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_diag'}
    ob  = object_shape('ball',[-4 -4],[4 4]);                   % Ball of size 4
    d   = object_movement('line',[1 1]);                        % Moving along diagonal direction (+x+y)
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_circ'}
    ob  = object_shape('ball',[-4 -4],[4 4]);                   % Ball of size 4
    d   = object_movement('circ',[8 8],8,0,0.1);                % Circular motion around (8,8), circle size=10, initial angle=0, angular timestep=0.1 radians relative to radius
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_collide'}
    ob  = object_shape('ball',[-4 8],[4 4]);                    % Ball of size 4
    d   = object_movement('line',[1 0]);                        % Moving along positive x direction
    tin = object_add(tin,ob,d);                                 % Add this object
    ob  = object_shape('ball',[24 8],[4 4]);                    % Ball of size 4
    d   = object_movement('line',[-1 0]);                       % Moving along negative x direction
    tin = object_add(tin,ob,d);                                 % Add this object
case {'ball_expand'}
    ob  = object_shape('ball',[8 8],[4 4]);                     % Ball of size 4
    d   = object_movement('expd',[1 1]);                          % Circular motion around (8,8), circle size=10, initial angle=0, angular timestep=0.1 radians relative to radius
    tin = object_add(tin,ob,d);                                 % Add this object
otherwise
    error(sprintf('Unknown title ''%s'' for test_case',title)); % Unknown test case, exit
end
