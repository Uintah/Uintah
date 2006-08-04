%function tout = test_movement(tin)
%TEST_MOVEMENT a driver for moving objects and creating moving box collections around them.
%
%   TOUT = TEST_MOVEMENT(TIN) is a driver for CREATE_CLUSTER and UPDATE_CLUSTER functions, 
%   that takes shapes of flagged cells (e.g., circles, boxes) and their direction of movement,
%   and moves them over a domain as in "time-stepping". The objects and their movements are
%   specified by the structure TIN, that can be prepared by the function TEST_CASE. For each 
%   timestep, we generate the boxes around the flagged cells by CREATE_CLUSTER, or only update
%   them with UPDATE_CLUSTER. Statistics on how many re-boxing are actually needed, etc.,
%   are printed and plotted at the end of the run. A sample of timestep configurations 
%   of the cells and their covering boxes, is also plotted. The statistics and original 
%   parameters of TIN are output in the structure TOUT.
%   
%   See also CREATE_CLUSTER, TEST_CASE, UPDATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/13/2004    Version 1: moving a circle in a horizontal direction
%         05/27/2004    Version 2: encapsulated plots, added test case structure of input (tin), output (tout)

%%%%% Set and print parameters
tin             = test_case('ball_x');                          %---------- Parameters of this test case ----------

opts            = [];                                           %---------- Parameters for cluster creation ----------
opts.efficiency = 0.8;                                          % Lowest box efficiency allowed
opts.low_eff    = 0.5;                                          % Efficiency threshold for boxes that don't have holes/inflections
opts.min_side   = 10;                                           % Minimum box side length allowed (in all directions)
opts.max_volume = 400;                                          % Maximum box volume allowed. Need to be >= (2*min_side)^dim.
opts.print      = 0;                                            % Printouts flag
opts.plot       = 0;                                            % Plots flag
opts_create     = opts;

opts            = opts_create;                                  %---------- Parameters for cluster update ----------
opts.print      = 0;                                            % Printouts flag7
opts.plot       = 0;                                            % Plots flag
opts_update     = opts;

fprintf('<<<<<<<<<<<<<< TEST_MOVEMENT: moving points and boxes around them >>>>>>>>>>>>>>\n');
if (opts.print)
    fprintf('Parameters:\n');                                   % Print parameters
    fprintf('Test case title   : %s\n',tin.title);              % Title string
end

%%%%% Initial time: create the initial boxes, initialize stats array
count           = 1;                                            % Counter of number of timesteps
saved_tstep     = 0;                                            % Counter for saved timesteps
t               = tin.init_t;                                   % Initial time
regrid          = 1;                                            % Save initial regridding status
points_old      = position_objects(tin,t);                      % Create the flagged cells array
points_dilated  = dilate(points_old,tin.dilate_create);             % Dilate by a big amount when creating boxes
[rect,s]        = create_cluster(points_dilated,opts_create);   % Create the initial set of boxes (t=0)
all_rect        = rect;                                         % Accumulates boxes from all times
%%%%% Accumulate statistics
stats.t             = [];
stats.regrid_status = [];
stats.efficiency    = [];
stats.num_boxes     = [];
stats.avg_volume    = [];
stats.avg_side_rat  = [];
if (~isempty(s))
    stats.t             = [stats.t; t];
    stats.regrid_status = [stats.regrid_status regrid];
    stats.efficiency    = [stats.efficiency; s.efficiency];
    stats.num_boxes     = [stats.num_boxes ; s.num_boxes ];
    stats.avg_volume    = [stats.avg_volume; s.avg_volume];
    stats.avg_side_rat  = [stats.avg_side_rat; s.avg_side_rat];
end

%%%%% Plot-outs of the points and current boxes
fprintf('Initial time = %d\n',t);
if (tin.plot_flag)
    figure(1);
    clf;
    plot_points(points_old,'red');
    hold on;
    plot_boxes(rect,'black');
    axis equal;
    axis([0 tin.domain_size(1) 0 tin.domain_size(2)]);
    %    pause
    shg;
    pause(tin.delay);
    eval(sprintf('print -depsc t%d.eps',t));    
end
saved_tstep = saved_tstep + 1;
saved{saved_tstep}.t            = t;
saved{saved_tstep}.points_old   = points_old;
saved{saved_tstep}.rect         = rect;

%%%%% Main loop over time steps
for count = [1:tin.num_tsteps]+1,
    t           = t+tin.dt;
    fprintf('time step = %f\n',t);
    ind_old     = find(points_old);                             % Old flagged cells in a 1D array
    points_new  = position_objects(tin,t);                      % Create the flagged cells array at the new timestep    
    
    points_out  = dilate(points_new,tin.dilate_check);          % Dilate error by a small amount to check if it's still within boxes
    for k = 1:size(rect,1)                                      % Get rid of new points that are covered by old boxes
        r = rect(k,:);                                          % Old box no. k
        points_out(r(1):r(3),r(2):r(4)) = 0;                    % Delete new points in this box
    end
    
    if (isempty(find(points_new)))                              % No points inside the domain
        fprintf('No error in domain, empty box covering\n');
        regrid = 0;
        rect   = [];
        s = [];
    elseif (isempty(find(points_out)))                              % Nothing is outside
        regrid = 0;
        fprintf('Error still inside boxes\n');
    
        %%% LATER: MAKE SURE THIS PART ONLY DELETES BOXES, NOT FULLY REGRIDS
        % Update because some rectangles are now empty, maybe need to delete them                
        points_new_dilated = dilate(points_new,tin.dilate_create);             % Dilate by a big amount when creating boxes
        fprintf('Checking for empty rectangles\n');
        for k = 1:size(rect,1)                                      % Get rid of new points that are covered by old boxes
            r = rect(k,:);                                          % Old box no. k
            points_local = points_new(r(1):r(3),r(2):r(4));
            fprintf('Rectangle %d, #points = %d\n',k,length(find(points_local)));
            if (isempty(find(points_local)))
                regrid = 1;
                break;
            end
        end
        fprintf('regrid = %d\n',regrid);
    else
        points_new_dilated = dilate(points_new,tin.dilate_create);             % Dilate by a big amount when creating boxes
        ind_new     = find(points_new_dilated);                     % New flagged cells in a 1D array
        points_add  = zeros(size(points_old));                      % An array for the cells added in this time step
        points_add(setdiff(ind_new,ind_old)) = 1;                   % The additional cells of the new time step in a 2D array
        
        points_old(setdiff(ind_old,ind_new)) = 0;                   % The old cells that are deleted in the new time step
        [rect_new,status,s] = update_cluster(...
            points_old,rect,points_add,opts_update);                % Try to add the new points using the update function    
        if ((status >= 0) & (mod(count,tin.regrid_max) ~= 0))       % Update suceeded and we are not forcing a re-gridding
            rect    = rect_new;                                     % Update box set
            regrid  = 0;
        else                                                        % Update failed, re-grid
            fprintf('Re-gridding\n');
            [rect,s] = create_cluster(points_new_dilated,opts_create);      % Create the set of boxes from the current points
            regrid  = 1;
        end
    end
    
    points_old  = points_new;                                   % Advance points_old to the next timestep
    %%%%% Accumulate statistics
    if (~isempty(s))
        all_rect            = [all_rect; rect];
        stats.t             = [stats.t; t];
        stats.regrid_status = [stats.regrid_status regrid];
        stats.efficiency    = [stats.efficiency; s.efficiency];
        stats.num_boxes     = [stats.num_boxes ; s.num_boxes ];
        stats.avg_volume    = [stats.avg_volume; s.avg_volume];
        stats.avg_side_rat  = [stats.avg_side_rat; s.avg_side_rat];
    end
    
    %%%%% Plot-outs of the points and current boxes
    if (tin.plot_flag)
        figure(1);
        clf;
        plot_points(points_old,'red');
        hold on;
        plot_boxes(rect,'black');
        axis equal;
        axis([0 tin.domain_size(1) 0 tin.domain_size(2)]);
        %    pause
        shg;
        pause(tin.delay);
%        if ((t == 19) | (t == 20) | (t == 21))
            saved_tstep = saved_tstep + 1;
            saved{saved_tstep}.t            = t;
            saved{saved_tstep}.points_old   = points_old;
            saved{saved_tstep}.rect         = rect;
            eval(sprintf('print -dtiff -r50 t%03d.tif',t));
%            eval(sprintf('print -djpeg td.jpg',t));
%            eval(sprintf('print -depsc t%d.eps',t));
%        end
    end    
end

%%%%% Final plots, prepare output structure
final_plots(stats,all_rect);

tout.input      = tin;
tout.stats      = stats;
tout.saved      = saved;
