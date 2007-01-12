function plot_points(points,color)
%PLOT_POINTS Plot the current flagged cells in the current figure
%   PLOT_POINTS(POINTS,COLOR) plots the array of flagged cells POINTS in the current figure.
%   The marker color is COLOR.
%
%   See also FINAL_PLOTS, FINAL_STATS, PLOT_BOXES.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

if (nargin < 2)
    color = 'black';
end
[i,j]   = find(points);
h       = plot(i,j,'.');
set(h,'color',color);
set(h,'MarkerSize',15);
%set(h,'linewidth',0.001);
%axis([min(i)-1 max(i)+1 min(j)-1 max(j)+1]);
%axis off;
