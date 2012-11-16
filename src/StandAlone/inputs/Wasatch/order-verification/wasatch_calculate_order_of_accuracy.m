function [] = wasatch_calculate_order_of_accuracy(varname,fieldtype,figure_title,outpuf_file_name)

format long
plot_style;
%
xfilename = strcat('X',fieldtype,'_512.txt');
yfilename = strcat('Y',fieldtype,'_512.txt');

x = load(xfilename);
y = load(yfilename);

x = x(:,4);
y = y(:,4);
%
nLevels = 5;
nRefinements = 4;
%
filename{1} = strcat(varname,'_dt_0_01_timestep_1.txt');
filename{2} = strcat(varname,'_dt_0_005_timestep_2.txt');
filename{3} = strcat(varname,'_dt_0_0025_timestep_4.txt');
filename{4} = strcat(varname,'_dt_0_00125_timestep_8.txt');
filename{5} = strcat(varname,'_dt_0_000625_timestep_16.txt');
%
for i=1:nLevels
    num{i} = load(filename{i});
    num{i} = num{i}(:,4);
end
%
for i=1:nRefinements
    err(i) = norm(num{i+1} - num{i},inf);
end
%
for i=1:3
    log( err(i+1) / err(i) ) / log(1/2)
end

%--------------------------------------------
dt =[0.005;0.0025;0.00125;0.000625];

err_n = err/err(1);

% create a new figure
figure

% plot first order error line
dt_1n = dt/dt(1);
loglog(dt,dt_1n,'k-.','linewidth',1.1)

hold on
% plot secondorder error line
dt_2n = dt.^2/dt(1)^2;
loglog(dt,dt_2n,'b--','linewidth',1.1)

% plot third order error line
dt_3n = dt.^3/dt(1)^3;
loglog(dt,dt_3n,'r-','linewidth',1.1)

% plot solution error
loglog(dt,err_n,'bo','markerfacecolor','b')

% format figure...
set(gca,'XDir','Reverse')
set(gca, 'FontName', Font_Name);

title(figure_title);


legend('1^{st}','2^{nd}','3^{rd}','Normalized error','Location','NorthEast');

set(gca,'FontSize',12.0);
xlabel('\Delta {\it t}','FontSize',12.0);
ylabel('normalized error','FontSize',12.0);

grid on; 
% print to pdf
set(gcf,'Visible',Figure_Visibility);
set(gcf,'PaperUnits',Paper_Units);
set(gcf,'PaperSize',[Paper_Width (Paper_Height)]);
set(gcf,'PaperPosition',[0.0 0.0 Paper_Width (Paper_Height)]);
print(gcf,'-dpdf',['',outpuf_file_name]);

hold off
end