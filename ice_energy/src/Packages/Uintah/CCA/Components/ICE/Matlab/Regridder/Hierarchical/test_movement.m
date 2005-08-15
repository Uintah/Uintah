%function tout = test_movement(tin)
%TEST_MOVEMENT Driver for moving objects and creating moving box collections around them.
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
%   See also MARK_PATCHES, OBJECT_RENDER, TEST_CASE.

% Author: Oren Livne
% Date  : 05/13/2004    Version 1: moving a circle in a horizontal direction
%         05/27/2004    Version 2: encapsulated plots, added test case structure of input (tin), output (tout)
%         06/28/2004    Simplified, in accordance with marked_patches().

%%%%% Set and print parameters
global_params;
o               = 1;                                            % List offset for MATLAB
tin             = test_case('ball_x');                          %---------- Parameters of this test case ----------

fprintf('<<<<<<<<<<<<<< TEST_MOVEMENT: moving flagged and boxes around them >>>>>>>>>>>>>>\n');
if (tin.print)
    fprintf('Parameters:\n');                                   % Print parameters
    fprintf('\tTest case title   : %s\n',tin.title);            % Title string    
    fprintf('\tDomain size\t\t= ');
    print_vector(tin.domain,'x','float');
    fprintf('\n');
    fprintf('\tNumber of timesteps\t= %d\n' ,tin.num_tsteps);
    fprintf('\tInitial time\t\t= %f [sec]\n',tin.init_t);
    fprintf('\tDelta_t\t\t\t= %f [sec]\n',tin.dt);
    fprintf('\n');
    fprintf('\t# safety layers, boundary= %d [cell]\n',tin.safe_bdry);
    fprintf('\t# safety layers, create\t= %d [cell]\n',tin.safe_create);
    fprintf('\t# safety layers, delete\t= %d [cell]\n',tin.safe_delete);
    fprintf('\n');
    fprintf('\t# processors\t\t= %d\n',tin.num_procs);
    fprintf('\tMin. patch side length\t= %d [cell] \n',tin.min_side);
    fprintf('\tMax. patch volume\t= %d [cell^d]\n',tin.max_volume);
    fprintf('\n');
    fprintf('\tPlot flag\t\t= %d\n'         ,tin.plot);
    fprintf('\tPrint flag\t\t= %d\n'         ,tin.print);
    fprintf('\tDelay time\t\t= %f [sec]\n'   ,tin.delay);
    fprintf('\n');    
    fprintf('\tMax. # levels\t\t= %d\n' ,tin.max_levels);
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('\n');
end

%%%%% Create the list of levels and init stats array (tout)
dim                 = length(tin.domain);                       % Dimension of the problem
L                   = cell(tin.max_levels,1);                   % List of levels; L{o} is the coarsest, L{o+1}-finer, etc.
levels_create(tin.max_levels);                                  % Init levels o..max_levels+o-1
print_level_info;
tout                = [];
tout.in             = tin;                                      % Save input parameters in the output data
tout.t              = zeros(tin.num_tsteps,1);
tout.tstep          = zeros(tin.num_tsteps,1);
tout.data           = zeros(tin.num_tsteps,length(L),9);
tout.sum_data       = zeros(tin.num_tsteps,4);

%%%%% Main loop over time steps
for count = [0:tin.num_tsteps-1],
    t = tin.init_t+count*tin.dt;
    fprintf('Time step = %d, Time = %f [sec]\n',count+o,t);
    for k = o:length(L)+o-1                                     % Initialize some counters
        L{k}.num_created    = 0;
        L{k}.num_deleted    = 0;
    end
    
    for k = o:length(L)+o-2                                     % Mark patches for refinement at all levels
        fprintf('---- Level %d: RENDERING AND MARKING PATCHES FOR REFINEMENT ----\n',k);
        L{k}.cell_err           = object_render(t,k);           % Synthesize the flagged cells array at this level
        L{k}.cell_err_create    = dilate_list(L{k}.cell_num,...
            L{k}.cell_err,tin.safe_create);                     % Dilate by a big amount when creating patches
        L{k}.cell_err_delete    = dilate_list(L{k}.cell_num,...
            L{k}.cell_err,tin.safe_delete);                     % Dilate by a small amount when checking patches
        if (tin.plot >= 2)
            fprintf('Before marking\n');
            for l = o:length(L)+o-1                             % Display all finer grids
                plot_grid(l);
            end
%            pause
        end
        mark_patches(k);                                        % Mark patches that need refinement, based on L{k}.flagged
        if (tin.plot >= 2)
            fprintf('After marking\n');
            for l = o:length(L)+o-1                             % Display all finer grids
                plot_grid(l);
            end
%            pause
        end    
    end

    %%%%% Accumulate statistics, printout
    plot_composite_grid(t);
    if (tin.plot >= 1)
        if (ismember(count+o,tin.tsteps_save))
            fprintf('Saving grid\n');
            eval(sprintf('print -dtiff %s_grid_t%d.tiff',tin.title,count+o));
%            eval(sprintf('print -depsc %s_grid_t%d.eps',tin.title,count+o));
        end
    end
    if (tin.print >= 2)
        print_level_info;
    end    
    tout.t(count+o)             = t;
    tout.tstep(count+o)         = count+o;
    tout.data(count+o,:,:)      = levels_stats;
    tout.sum_data(count+o,:)    = sum(squeeze(tout.data(count+o,:,[1 2 8 9])),1); % Sum over levels the following columns of tout.data: 1. #patches 2. #cells 8. #created patches 9. #deleted patches
    fprintf('###########################################################################################\n');
end

%%%%% Final plots, prepare output structure
eval(sprintf('save %s_data.mat tout',tin.title));
final_plots(tout);
