function final_plots(tout)
%FINAL_PLOTS final plots of statistics for the hierarhical gridding algorithm in time-stepping mode.
%   At the end of a test of moving points in time and creating patches around them, we plot some statistics
%   versus time: patch efficiency, number of patches, etc.
%   Input: TOUT = structure containing the output statistics.
%
%   See also LEVELS_STATS, TEST_MOVEMENT.

% Author: Oren Livne
% Date  : 06/22/2004    Adapted from the Berger-Rigostous code.

labels_data = {...
        '# Patches',...
        '# Cells',...
        '% Active Patches',...
        'Average Patch Efficiency',...
        'Median Patch Efficiency',...
        'Maximum Patch Efficiency',...
        '# Empty Patches', ...
        '# Created Patches',...
        '# Deleted Patches',...
    };
sub_size = [3 3];

labels_sum_data = {...
        'Total # Patches',...
        'Total # Cells',...
        'Total # Created Patches',...
        'Total # Deleted Patches',...
    };
sub_size_sum = [2 2];

num_levels  = size(tout.data,2);
leg         = cell(num_levels,1);
for k = 1:num_levels,
    leg{k} = sprintf('Level %d',k);
end

%%%%% For each indicator: a list of graphs of the indicator at each level vs. t
fig = 100;
figure(fig);
clf;
set(gcf,'Position',[931 432 664 696]);
for col = 1:length(labels_data),
    subplot(sub_size(1),sub_size(2),col);
    plot(tout.t,squeeze(tout.data(:,:,col)));
    xlabel('time [sec]');
    ylabel(labels_data{col});
    if (col == 1)
        legend(leg{:});
    end
end
eval(sprintf('print -depsc %s_stats.eps',tout.in.title));

%%%%% Summed-over-level statistics vs. t
fig = fig+1;
figure(fig);
clf;
set(gcf,'Position',[931 432 664 696]);
for col = 1:length(labels_sum_data),
    fig = fig+1;
    subplot(sub_size_sum(1),sub_size_sum(2),col);
    plot(tout.t,squeeze(tout.sum_data(:,col)));
    xlabel('time [sec]');
    ylabel(labels_sum_data{col});
end
eval(sprintf('print -depsc %s_stats_sum.eps',tout.in.title));
