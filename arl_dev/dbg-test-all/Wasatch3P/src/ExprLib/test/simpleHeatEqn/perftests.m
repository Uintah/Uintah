function pertests

% timings obtained on Macbook pro 2 core laptop

% serial
n    = [ 10^3 15^3 20^3 30^3 40^3 50^3];
tseg = [ .15  .38  .80  2.43 5.37 10.5];
tcom = [ .12  .33  .66  2.05 4.66 9.2 ];
tmon = [ .10  .27  .61  2.03 4.76 9.2 ];
plotit(n,tseg,tcom,tmon,'Serial')

% 2 threads
n    = [ 10^3 15^3 20^3 30^3 40^3 50^3];
tseg = [ .38  .52  .74  1.76 3.47 6.7 ];
tcom = [ .31  .44  .63  1.49 3.07 6.1 ];
tmon = [ .11  .21  .39  1.28 2.88 6.2 ];
plotit(n,tseg,tcom,tmon,'2 Threads')

% 4 threads
n    = [ 10^3 15^3 20^3 30^3 40^3 50^3 60^3];
tseg = [ 1.18 1.34 1.39 2.08 3.55 6.12 9.85];
tcom = [ 0.97 1.06 1.12 1.73 3.07 5.70 9.67];
tmon = [ 0.12 0.19 0.34 0.95 2.14 4.58 7.94];
plotit(n,tseg,tcom,tmon,'4 Threads')
end


function plotit( n, ts, tc, tm, titlestring )
fontsize   = 16;
linewidth  = 1.5;
markersize = 10;
figure;
loglog( n, ts, '-o', n, tc, '-x', n, tm, '-s', 'LineWidth', linewidth, 'MarkerSize', markersize );
set(gca,'FontSize',16)
xlabel('# pts');
ylabel('Time');
axis tight; grid;
title(titlestring);
legend('segregated','combined','monolithic','Location','best');
axis( [1000,216000,0.1,10.5] )
end