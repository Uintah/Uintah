function plot_boxes(rect,color)
%PLOTS_BOXES Plot the current box covering in the current figure
%   PLOT_BOXES(RECT,COLOR) plots the box collection RECT in the current figure. The edge color
%   is COLOR.
%
%   See also FINAL_PLOTS, FINAL_STATS.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

if (nargin < 2)
    color = 'black';
end
offset = 0.2;
for i = 1:size(rect,1)
    h = rectangle('Position',[rect(i,1:2)-offset,[rect(i,3:4)-rect(i,1:2)]+2*offset]);
    set(h,'EdgeColor',color);
    set(h,'LineWidth',3);
%    axis off;
    set(gcf,'Position',[520 520 300 300]);
end
