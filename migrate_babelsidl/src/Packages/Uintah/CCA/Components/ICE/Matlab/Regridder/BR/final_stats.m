function stats = final_stats(points,rect,opts)
%FINAL_PLOTS Final printouts of overall statistics and plot-outs of the clustering algorithm.
%   At the end of the clustering algorithm, we print and plot some statistics
%   box efficiency, number of boxes, average volume, minimum and maximum box size, etc.
%   Input: POINTS = the array of flagged cells; RECT = box collection. opts = flags for prints/plots.
%   Output: STATS = structure with the statistics.
%
%   See also CREATE_CLUSTER, FINAL_PLOTS.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

if (nargin < 3)
    opts.print  = 0;
    opts.plot   = 0;
end

x                   = (rect(:,3)-rect(:,1)+1);
y                   = (rect(:,4)-rect(:,2)+1);
rect_area           = x.*y;

stats.flagged_area  = length(find(points));
stats.box_area      = sum(rect_area);
stats.efficiency    = 100*stats.flagged_area/stats.box_area;
stats.num_boxes     = size(rect,1);
stats.min_side      = min(min(rect(:,3)-rect(:,1)+1,rect(:,4)-rect(:,2)+1));
stats.max_volume    = max(rect_area);
stats.avg_volume    = mean(rect_area);
stats.avg_side_rat  = 100*mean(min(x,y)./max(x,y));

if (opts.print)
    flagged_area    = length(find(points));
    fprintf('Number of flagged pts  : %d\n'    ,stats.flagged_area);
    fprintf('Area of boxes          : %d\n'    ,stats.box_area);
    fprintf('Efficiency             : %.1f%%\n',stats.efficiency);
    fprintf('Number of boxes        : %d\n'    ,stats.num_boxes);
    fprintf('Minimum box edge       : %d\n'    ,stats.min_side);
    fprintf('Maximum box volume     : %d\n'    ,stats.max_volume);
    fprintf('Average box volume     : %f\n'    ,stats.avg_volume);
    fprintf('Average box side ratio : %.1f%%\n',stats.avg_side_rat);
end

if (opts.plot)
    figure(1);
    clf;
    plot_points(points);
    hold on;
    plot_boxes(rect);
    print -depsc cover.eps
    
    % figure(2);
    % clf;
    % plot_points(points);
    % print -depsc cells.eps
end
