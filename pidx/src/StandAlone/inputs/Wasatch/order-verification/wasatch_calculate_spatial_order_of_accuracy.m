function [] = wasatch_calculate_spatial_order_of_accuracy(varname,xfieldtype,yfieldtype,figure_title,outpuf_file_name,use_richardson)

format long
nLevels = 5;
nRefinements = 4;
if (nargin < 6)  ||  isempty(use_richardson)
    use_richardson = 0;
end
plot_style;
xfilename{1} = strcat('X',xfieldtype,'_dx_32x32_timestep_1.txt');
xfilename{2} = strcat('X',xfieldtype,'_dx_64x64_timestep_1.txt');
xfilename{3} = strcat('X',xfieldtype,'_dx_128x128_timestep_1.txt');
xfilename{4} = strcat('X',xfieldtype,'_dx_256x256_timestep_1.txt');
xfilename{5} = strcat('X',xfieldtype,'_dx_512x512_timestep_1.txt');

yfilename{1} = strcat('Y',yfieldtype,'_dx_32x32_timestep_1.txt');
yfilename{2} = strcat('Y',yfieldtype,'_dx_64x64_timestep_1.txt');
yfilename{3} = strcat('Y',yfieldtype,'_dx_128x128_timestep_1.txt');
yfilename{4} = strcat('Y',yfieldtype,'_dx_256x256_timestep_1.txt');
yfilename{5} = strcat('Y',yfieldtype,'_dx_512x512_timestep_1.txt');

for i=1:nLevels
    x{i} = load(xfilename{i});
    x{i} = x{i}(:,4);
    x{i} = 2*pi.*x{i};
    y{i} = load(yfilename{i});
    y{i} = y{i}(:,4); 
    y{i} = 2*pi.*y{i};
end

%
%
%

filename{1} = strcat(varname,'_dx_32x32_timestep_1.txt');
filename{2} = strcat(varname,'_dx_64x64_timestep_1.txt');
filename{3} = strcat(varname,'_dx_128x128_timestep_1.txt');
filename{4} = strcat(varname,'_dx_256x256_timestep_1.txt');
filename{5} = strcat(varname,'_dx_512x512_timestep_1.txt');
%
for i=1:nLevels
    num{i} = load(filename{i});
    num{i} = num{i}(:,4);
end
%
A = 1.0;
nu = 0.001;
%t = 0.000625;
dt=[0.001;0.0005;0.00025;0.000125;0.0000625]
t=0;
denom = [1;4;16;64;256];
N     = [1024;4096;16384;65536;262144];

for i=1:nLevels
    %analytical = 1 - A*exp(-2*nu*t(i)).*cos(x{i} - t(i)).*sin(y{i} - t(i));
    u = 1 - A*exp(-2*nu*t).*cos(x{i} - t).*sin(y{i} - t);
    v = 1 + A*exp(-2*nu*t).*cos(y{i} - t).*sin(x{i} - t);    
    if strcmp(varname,'xmom')
      analytical = u;
    elseif strcmp(varname,'ymom')
      analytical = v;    
    elseif strcmp(varname,'tauxx')
      analytical = - 2*pi*2*nu*A*exp(-2*nu*t).*sin(t - x{i}).*sin(t- y{i});
    elseif strcmp(varname,'tauyy')
      analytical =   2*pi*2*nu*A*exp(-2*nu*t).*sin(t-x{i}).*sin(t-y{i});            
    elseif strcmp(varname,'tauyx')
      analytical = zeros(size(num{i}));                  
    elseif strcmp(varname,'tauxy')
      analytical = zeros(size(num{i}));
    elseif strcmp(varname,'xconvx')
      analytical = u.*u;            
    elseif strcmp(varname,'xconvy')
      analytical = u.*v;    
    elseif strcmp(varname,'yconvy')
      analytical = v.*v;            
    elseif strcmp(varname,'yconvx')
      analytical = u.*v;                        
    elseif strcmp(varname,'xmomrhspart')     
      analytical = A*pi*exp(-4*nu*t).*( - A.*sin(2*t - 2.*x{i}) + 2*exp(2*nu*t).*( cos(2*t - x{i} - y{i}) - 4*pi*nu.*cos(t-x{i}).*sin(t-y{i}) ) );
    elseif strcmp(varname,'ymomrhspart')
      analytical = A*pi*exp(-4*nu*t).*( - A.*sin(2*t - 2.*y{i}) - 2*exp(2*nu*t).*( cos(2*t - x{i} - y{i}) - 4*pi*nu.*cos(t-y{i}).*sin(t-x{i}) ) );      
    elseif strcmp(varname,'xmomrhsfull')
      analytical =   A*exp(-2*nu*t).*( cos(2*t - x{i} - y{i}) - nu.*sin(x{i} - y{i}) - nu.*sin(2*t - x{i} - y{i}) );
    elseif strcmp(varname,'ymomrhsfull')
      analytical = - A*exp(-2*nu*t).*( cos(2*t - x{i} - y{i}) + nu.*sin(x{i} - y{i}) - nu.*sin(2*t - x{i} - y{i}) );       
    end
    
    err(i) = norm(num{i} - analytical,2)*sqrt(1/N(i));
    
    if strcmp(varname,'xmom') || strcmp(varname,'ymom')
        err(i) = err(i)/dt(i);
    end
end
%
if use_richardson
    for i=1:(nRefinements-1)
        order = log ( (err(i+2) - err(i+1)) /(err(i+1) - err(i) ) ) / log(1/2)    
    end
else
    for i=1:nRefinements
        order = log ( err(i+1)/err(i) ) / log(1/2)
    end    
end
%
if use_richardson
    for i=1:4
        err_n(i) = err(i+1) - err(i);
    end
else
    err_n= err;
end

%--------------------------------------------
dx =[1/32;1/64;1/128;1/256;1/512];

if use_richardson
    dx =[1/64;1/128;1/256;1/512];
end

err_n = err_n/err_n(1);

% create a new figure
figure

% plot first order error line
dx_1n = dx/dx(1);
loglog(dx,dx_1n,'k-.','linewidth',1.1)

hold on
% plot secondorder error line
dx_2n = dx.^2/dx(1)^2;
loglog(dx,dx_2n,'b--','linewidth',1.1)

% plot third order error line
dx_3n = dx.^3/dx(1)^3;
loglog(dx,dx_3n,'r-','linewidth',1.1)

% plot solution error
loglog(dx,err_n,'bo','markerfacecolor','b')

% format figure...
%set(gca,'XDir','Reverse')
set(gca, 'FontName', Font_Name);

title(figure_title);


legend('1^{st}','2^{nd}','3^{rd}','Normalized error','Location','SouthEast');

set(gca,'FontSize',12.0);
xlabel('\Delta {\it x}','FontSize',12.0);
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