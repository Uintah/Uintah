function final_plots(stats,all_rect)
%FINAL_PLOTS final plots of statistics for the clustering algorithm in time-stepping mode.
%   At the end of a test of moving points in time and creating patches around them, we plot some statistics
%   versus time: box efficiency, number of boxes, average volume.
%   Input: STATS = structure with the statistics.. ALL_RECT = box collection (union over all time steps).
%
%   See also CREATE_CLUSTER, TEST_MOVEMENT.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

fig = 1;

fig = fig+1;
figure(fig);
clf;
plot(stats.t,stats.efficiency);
xlabel('time [sec]');
ylabel('Efficiency [%]');
print -depsc teff.eps

fig = fig+1;
figure(fig);
clf;
plot(stats.t,stats.num_boxes);
xlabel('time [sec]');
ylabel('Number of boxes');
print -depsc tnbox.eps

fig = fig+1;
figure(fig);
clf;
plot(stats.t,stats.avg_volume);
xlabel('time [sec]');
ylabel('Average Box Volume [cell^d]');
print -depsc tavgvol.eps

dim         = size(all_rect,2)/2;
a           = sortrows(all_rect,[1:2*dim]);
base        = max(a(:))+1;
hash        = sum(a.*repmat(base.^[0:size(a,2)-1],size(a,1),1),2);
cross       = find(diff(hash) ~= 0);
histogram   = [cross(1) ; diff(cross) ; length(hash)-cross(length(cross))];
ind         = [cross ; length(hash)];

stats.avg_box_life  = mean(histogram);
num_tsteps  = length(stats.t)-1;
stats.avg_regrid    = (length(find(stats.regrid_status))-1)/num_tsteps;        % -1 because we can ignore the initial time
fprintf('Average efficiency                    : %.1f%%\n',mean(stats.efficiency));
fprintf('Average #timesteps of a box''s life    : %f\n',stats.avg_box_life);
fprintf('Average #timesteps between regriddings: %f\n',1/stats.avg_regrid);
