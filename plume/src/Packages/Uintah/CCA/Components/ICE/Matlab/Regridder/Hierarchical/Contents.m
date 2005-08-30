% AMR Gridding Algorithm - Hierarchical Approach.
% CSAFE MATLAB Toolbox  Version 4.6    28-Jun-2004
%
% Display and statistics functions.
%   final_plots         - final plots of statistics
%   plot_composite_grid - Plot the composite AMR grid at a certain timestep.
%   plot_flagged        - Plot flaged cells.
%   plot_flagged2       - Plot flaged cells.
%   plot_grid           - Plot an AMR level.
%   print_level_info    - Print levels' information.
%   
% Index handling of cells and patches.
%   convert_c2f         - Convert coarse level to fine level coordinates.
%   convert_c2p         - Convert cell to patch coordinates.
%   convert_f2c         - Convert fine level to coarse level coordinates.
%   patch_find          - Return patch index from its coordinates at a given level.
%   patch_finish        - Return patch "last" cell coordinate.
%   patch_start         - Return patch "first" cell coordinate.
%
% Object definition and movement.
%   object_add          - Add object to input parameter structure.
%   object_movement     - Create object handle for a time-stepping test.
%   object_render       - Position objects in the binary image of flagged cells, at time t.
%   object_shape        - Create object handle for a time-stepping test.
%
% Hierchical algorithm: level manipulation.
%   levels_create       - Initialize levels.
%   levels_stats        - Compute statistics at all levels.
%   mark_patches        - Mark patches for refinement (main hierchical algorithm code).
%
% Drivers, main routines, global parameters.
%   global_params       - Global parameters for all routines in this code.
%   test_case           - Prepare parameters structure for TEST_MOVEMENT.
%   test_movement       - Driver for moving objects and creating moving box collections around them.

% Author: Oren Livne
% Date  : 06/28/2004    Version 4
%                       First version of the hierarchical approach that has a fully
%                       working simulation of object movement. Plots are 2D specific,
%                       rest of the code already applies to d-D.
