%BOUNDING_FIG Illustrative figures of dissecting a box in the CREATE_CLUSTER algorithm.
%   BOUND_FIG is a script that creates several figures showing the stages of dissecting a
%   box in CREATE_CLUSTER. Used in Clustering Report 2.
% 
%   See also CREATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/13/2004    Version 1: five figures created and saved to color EPS files.

%%%%% Original box
figure(1);
clf;
hold on;
axis image
h = rectangle('Position',[0 5 4 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[3 0 1 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[0 0 4 10]);
set(h,'linewidth',3);
h = text(2,-1.5,'4');
set(h,'fontsize',16);
h = text(-1,5,'10');
set(h,'fontsize',16);
axis([-1 5 -1 11])
axis off
print -depsc bounding_a.eps

%%%%% Dissection into two smaller boxes
figure(2);
clf;
hold on;
axis image
h = rectangle('Position',[0 5 4 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[3 0 1 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[0 0 4 5]);
set(h,'linewidth',3);
h = rectangle('Position',[0 5 4 5]);
set(h,'linewidth',3);
h = text(2,-1.5,'4');
set(h,'fontsize',16);
h = text(-0.5,2.5,'5');
set(h,'fontsize',12);
h = text(-0.5,7.5,'5');
set(h,'fontsize',12);
axis([-1 5 -1 11])
axis off
print -depsc bounding_b.eps

%%%%% Tight bounding boxes
figure(3);
clf;
hold on;
axis image
h = rectangle('Position',[0 5 4 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[3 0 1 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[0 5 4 5]);
set(h,'linewidth',3);
h = rectangle('Position',[3 0 1 5]);
set(h,'linewidth',3);
h = text(3.3,-0.5,'1');
set(h,'fontsize',12);
h = text(2,-1.5,'4');
set(h,'fontsize',16);
h = rectangle('Position',[0 0 4 10]);
set(h,'linewidth',3);
set(h,'linestyle','--');
axis([-1 5 -1 11])
axis off
print -depsc bounding_c.eps

%%%%% Extend to minimal size
figure(4);
clf;
hold on;
axis image
h = rectangle('Position',[0 5 4 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[3 0 1 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[0 5 4 5]);
set(h,'linewidth',3);
h = rectangle('Position',[3 0 2 5]);
set(h,'linewidth',3);
h = text(3.3,-0.5,'1');
set(h,'fontsize',12);
h = text(2,-1.5,'4');
set(h,'fontsize',16);
h = rectangle('Position',[0 0 4 10]);
set(h,'linewidth',3);
set(h,'linestyle','--');
axis([-1 5 -1 11])
axis off
print -depsc bounding_d.eps

% Extend to minimal size
figure(5);
clf;
hold on;
axis image
h = rectangle('Position',[0 5 4 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[3 0 1 5]);
set(h,'facecolor','red');
set(h,'edgecolor','red');
h = rectangle('Position',[0 5 4 5]);
set(h,'linewidth',3);
h = rectangle('Position',[2 0 2 5]);
set(h,'linewidth',3);
h = text(3.3,-0.5,'1');
set(h,'fontsize',12);
h = text(2,-1.5,'4');
set(h,'fontsize',16);
h = rectangle('Position',[0 0 4 10]);
set(h,'linewidth',3);
set(h,'linestyle','--');
axis([-1 5 -1 11])
axis off
print -depsc bounding_e.eps
