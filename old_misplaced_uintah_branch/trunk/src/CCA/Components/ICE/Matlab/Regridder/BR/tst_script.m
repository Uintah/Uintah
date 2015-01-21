load ../testcases/generic4
g = generic4';
[rect,s] =  create_cluster(g);

figure(1);
clf;
[i,j]   = find(g);
h       = plot(i,j,'.');
set(h,'color','red');
set(h,'MarkerSize',15);
%set(h,'linewidth',0.001);
%plot_points(g,'red');
hold on;
offset = 0.3;
for i = 1:size(rect,1)
    h = rectangle('Position',[rect(i,1:2)-offset,[rect(i,3:4)-rect(i,1:2)]+2*offset]);
    set(h,'EdgeColor','black');
    set(h,'LineWidth',3);
%    axis off;
    set(gcf,'Position',[520 520 300 300]);
end
axis equal;
axis off;
print -dtiff -r40 br.tif
