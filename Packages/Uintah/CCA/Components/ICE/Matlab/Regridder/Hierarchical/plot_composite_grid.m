function plot_composite_grid(t)
%PLOT_COMPOSITE_GRID Plot the composite AMR grid at a certain timestep.
%   PLOT_COMPOSITE_GRID(T) creates a figure with the cell lines of the composite
%   grid (i.e., the union of all levels), at the current timestep T.
%   This also includes the flagged cells (plotted as "x"'s inside the cells).
%   
%   See also PLOT_GRID, TEST_MOVEMENT.

% Author: Oren Livne
%         06/28/2004    Version 1: Created

global_params;

if (~tin.plot)
    return;
end

%%%%% Initialize offsets, djust image title, labels, position, axis
%%%%% Units for x- and y- for all plots are cells of the current level
offset_cell = 0.5;
offset_dom  = 0.1;
offset_text = [0.3 0.4];
offset_axis = 2;
ocell       = -1.0;
figure(50);
clf;
hold on;
axis equal;
zoom on;
axis([-0.5-offset_axis tin.domain(1)+offset_axis+0.5 0.5-offset_axis tin.domain(2)+offset_axis+0.5]);
xlabel('x [meter]');
ylabel('y [meter]');
title(sprintf('Composite Level Layout, Time = %f',t));
set(gcf,'Position',[931 432 664 696]);

%%%%% Plot domain
k=o;
dim                 = length(L{o}.cell_num);                    % Dimension of the problem
start = ones(1,dim);
finish = L{k}.cell_num;
siz = finish - start + 1;
a = rectangle('Position',[(start+ocell-offset_dom).*L{k}.cell_size,(siz+2*offset_dom).*L{k}.cell_size]);
set(a,'linewidth',2);
set(a,'edgecolor','black');

%%%%% Plot levels
for k = o:length(L)+o-1,
    s                   = L{k}.cell_num;
    flagged             = L{k}.cell_err;                        % Flagged cells d-D indices in array form
    flagged_d           = L{k}.cell_err_create;                 % Dilated (for check) flagged cells d-D indices in array form
    % Plot patches
    for p = o:size(L{k}.patch_active,1)+o-1,
        % Patch information
        j       = L{k}.patch_active(p,:);                       % d-D coordinates of the patch
        start   = patch_start(k,j);                             % Start cell of patch j
        finish  = patch_finish(k,j);                            % Finish cell of patch j
        siz     = finish - start + 1;                           % Size of this patch

        % Plot patch
        a = rectangle('Position',[(start+ocell).*L{k}.cell_size,siz.*L{k}.cell_size]);
        set(a,'linewidth',3);
        
        % Plot gridlines within each patch
        for d = 1:dim,
            for i = [start(d):start(d)+siz(d)],
                if (d == 1)
                    x = ([i i]+ocell).*L{k}.cell_size;
                    y = ([start(2) start(2)+siz(2)]+ocell).*L{k}.cell_size;
                else
                    y = ([i i]+ocell).*L{k}.cell_size;
                    x = ([start(1) start(1)+siz(1)]+ocell).*L{k}.cell_size;
                end
                line(x,y);
            end
        end
    end
    
    plot_flagged(k,'coord');    
end

shg;
pause(tin.delay);
