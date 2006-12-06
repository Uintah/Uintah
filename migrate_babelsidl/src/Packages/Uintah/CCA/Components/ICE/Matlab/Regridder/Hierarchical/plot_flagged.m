function plot_flagged(k,type)
%PLOT_FLAGGED Plot flaged cells.
%   PLOT_FLAGGED(K,TYPE) plots in the current figure "x"'s at flagged
%   cells, at level K. We can plot various sets of cells (originally flagged,
%   dilated, etc.). This routine is used in PLOT_COMPOSITE_GRID.
%   
%   See also PLOT_COMPOSITE_GRID, TEST_MOVEMENT.

% Author: Oren Livne
%         06/28/2004    Version 1: Created

global_params;

%%%%% Plot flagged cells
f_range     = {...
        L{k}.cell_err, ...
%        setdiff(L{k}.cell_err_create,L{k}.cell_err,'rows'), ...
%         L{k}.cell_err_create, ...
%         L{k}.new_cells_bdry ...
%        setdiff(L{k}.cell_err_delete,L{k}.cell_err_create,'rows'), ...
%        setdiff(L{k}.new_cells,L{k}.cell_err,'rows'), ...
%        setdiff(L{k}.new_cells_safe,L{k}.new_cells,'rows'), ...
    };
marker_size = [0.6, 0.6 , 0.2, 0.0, 0.0];
line_width  = [4  , 2   , 2  , 0.0, 0.0];

for count = 1:length(f_range),
    f = f_range{count};
    m = marker_size(count);
    w = line_width(count);
    bot = 0.5*(1-m);
    top = 0.5*(1+m);       
    if (isempty(f))
        continue;
    end
    switch (type)
    case 'coord',
        for i = 1:size(f,1),
            x   = [f(i,1)+bot f(i,1)+top];
            y1  = [f(i,2)+bot f(i,2)+top];
            y2  = [f(i,2)+top f(i,2)+bot];
            x   = (x-o)*L{k}.cell_size(1);
            y1  = (y1-o)*L{k}.cell_size(2);
            y2  = (y2-o)*L{k}.cell_size(2);
            a = line(x,y1);
            set(a,'color','blue');
            set(a,'linewidth',w);
            a = line(x,y2);
            set(a,'color','blue');
            set(a,'linewidth',w);
        end
    case 'cell',
        for i = 1:size(f,1),
            x   = [f(i,1)+bot f(i,1)+top];
            y1  = [f(i,2)+bot f(i,2)+top];
            y2  = [f(i,2)+top f(i,2)+bot];
            a = line(x,y1);
            set(a,'color','blue');
            set(a,'linewidth',w);
            a = line(x,y2);
            set(a,'color','blue');
            set(a,'linewidth',w);
        end
    end
end
