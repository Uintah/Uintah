function plot_grid(k)
%PLOT_GRID Plot an AMR level.
%   PLOT_GRID(K) creates a figure with the cell lines, patches and
%   lattice lines of level K, at the current timestep. This also
%   includes the flagged cells (plotted as "x"'s inside the cells).
%   
%   See also PLOT_COMPOSITE_GRID, TEST_MOVEMENT.

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
figure(k);
clf;
hold on;
axis equal;
zoom on;
axis([o-offset_axis L{k}.cell_num(1)+o-1+offset_axis o-offset_axis L{k}.cell_num(2)+o-1+offset_axis]);
xlabel('x [cell]');
ylabel('y [cell]');
title(sprintf('Level %d Layout (%d x %d cells), Time = %f',k,L{k}.cell_num,tout.t));
set(gcf,'Position',[931 432 664 696]);

%%%%% Plot domain
dim         = length(L{k}.cell_num);                            % Dimension of the problem
if (k < length(L)+o-1)
    b       = box_list(zeros(1,dim),L{k}.rat_patch);            % Prepare all combinations for sub-patch index offsets
    siz_sub = L{k+1}.patch_size./L{k}.rat_cell;                 % Size of a sub-patch [L{k} cells]
end
start       = zeros(1,dim)+o;
siz         = L{k}.cell_num;
a           = rectangle('Position',[start-offset_dom,siz+2*offset_dom]);
set(a,'linewidth',3);
set(a,'edgecolor','black');

%%%%% Plot patches
for p = o:size(L{k}.patch_active,1)+o-1,
    % Patch information
    j       = L{k}.patch_active(p,:);                           % d-D coordinates of the patch
    start   = patch_start(k,j);                                 % Start cell of patch j
    finish  = patch_finish(k,j);                                % Finish cell of patch j
    siz     = finish - start + 1;                               % Size of this patch
    
    % Plot sub-patches within each patch
    if (k < length(L)+o-1)
        for q = o:size(b,1)+o-1,                                % Loop over all possible patches; that might be 3x3 for a "normal" patch and 4x4 for a "big" last patch. So exit if we're outside the range of the L{k}-patch
            start_sub   = start + b(q,:).*siz_sub;              % Sub-patch start cell [L{k} cells]
            if (~isempty(find(start_sub > finish)))             % We are outside the L{k}-patch, skip this one
                continue;
            end
            start_f     = convert_c2f(k,start_sub,'cell');      % Sub-patch start cell [L{k+1} cells]
            j_f         = convert_c2p(k+1,start_f);             % Sub-patch patch coordinate [L{k+1} patches]
            finish_sub  = start_sub + siz_sub - 1;
            last        = find(j_f == L{k+1}.patch_num);
            finish_sub(last) = L{k}.cell_num(last);
            ind         = patch_find(k+1,j_f);                  % Index of sub-patch at level k+1 (if exists)
            a = rectangle('Position',[start_sub,finish_sub-start_sub+1]);
            set(a,'linewidth',3);
            set(a,'edgecolor','red');
            if (ind >= o)
                set(a,'facecolor','yellow');
            end
%            shg
%            pause
        end
    end    
    
    % Plot patch
    a = rectangle('Position',[start,siz]);
    set(a,'linewidth',4);
    set(a,'edgecolor','blue');

    % Mark newly patches in green    
    if (find(ismember(L{k}.new_patch,j,'rows')))
        set(a,'facecolor','green');
    end
    
    % Plot gridlines within each patch
    for d = 1:dim,
        for i = [start(d):start(d)+siz(d)],
            if (d == 1)
                x = [i i];
                y = [start(2) start(2)+siz(2)];
            else
                y = [i i];
                x = [start(1) start(1)+siz(1)];
            end
            line(x,y);
        end
    end
    
    % Print patch id at the starting cell
    a = text(start(1)+offset_text(1),start(2)+offset_text(2),sprintf('%d',p));
    set(a,'fontsize',20);
    set(a,'color','b');                
end

% Mark recently deleted patches in gray
for p = o:size(L{k}.deleted_patches,1)+o-1,
    j       = L{k}.deleted_patches(p,:);                        % d-D coordinates of the patch
    start   = patch_start(k,j);                                 % Start cell of patch j
    finish  = patch_finish(k,j);                                % Finish cell of patch j
    siz     = finish - start + 1;                               % Size of patch
    a = rectangle('Position',[start,siz]);
    set(a,'linewidth',4);
    set(a,'edgecolor','blue');
    set(a,'facecolor','cyan');
end
   
%plot_flagged(k,'cell');
plot_flagged2(k,'cell');

%shg;
%pause(tin.delay);
