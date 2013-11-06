%% MMS case
clc
clear all
close all
a = dlmread('MMS_u_n.txt',' ', 0, 0); 
for i=1:257 
    b(:,i)=a((a(:,1)==i-1),4);
end
u_mms_n=b';

clear ans a i b
a = dlmread('MMS_xxvol.txt',' ', 0, 0); 
xxvol=a(:,1);
clear a

tvec=[1000,2000,3000,4000,5000];

% for i=1:length(tvec)
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%      tim = tvec(i)*0.001;
%      S=-(5*tim/((tim^2)+1)) * sin(2*pi*xxvol/(3*tim+30));
%     plot(xxvol,S,xxvol,u_mms_n(:,i),':','LineWidth',2);
%     axis([-15 15 -2.5 2.5]);
%     ylabel('x-velocity','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('manufactured','numerical');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,2.1,timestr,'FontSize',40);
% 
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     plot(xxvol,abs(S-u_mms_n(:,i)),'LineWidth',2);
%     axis([-15 15 0 0.01]);
%     ylabel('absolute error in x-velocity ','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('absolute error');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,0.008,timestr,'FontSize',40);
% end;
% 
%figure('OuterPosition',[300,300,800,600],'Color','w');
%axes('FontSize',40,'FontName','Times New Roman');    
uNorm2_n = zeros(1,5);
for i=1:5
    tim = tvec(i)*0.001;
    S=-(5*tim/((tim^2)+1)) * sin(2*pi*xxvol/(3*tim+30));
    uNorm2_n(i) = norm(S-u_mms_n(:,i),2);    
end
uNorm2_n
% t=1:1:5;
% plot(t,uNorm2_n,'LineWidth',2);
% ylabel('norm of the abs-err in x-vel','FontSize',40,'FontName','Times New Roman');
% xlabel('time','FontSize',40,'FontName','Times New Roman');
% legend('old','new');
% axis tight

%  
a = dlmread('MMS_f_n.txt',' ', 0, 0); 
for i=1:255
    b(:,i)=a((a(:,1)==i-1),4);
end
f_mms_n=b';
clear ans a i b
a = dlmread('MMS_xsvol.txt',' ', 2, 0); 
xsvol=a(:,1);
clear a
% 
% for i=1:length(tvec)
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     tim = tvec(i)*0.001;
%     S=(5/(2*tim+5)) * exp(-5*xsvol.^2/(10+tim));    
%     plot(xsvol,S,xsvol,f_mms(:,tvec(i)),'--',xsvol,f_mms_n(:,tvec(i)),'--','LineWidth',2);
%     axis([-15 15 0 1.0]);
%     ylabel('mixture fraction','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('manufactured','old','new');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,0.9,timestr,'FontSize',40);
% 
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     plot(xsvol,abs(S-f_mms(:,tvec(i))),xsvol,abs(S-f_mms_n(:,tvec(i))),'--','LineWidth',2);
%     axis([-15 15 0 0.01]);
%     ylabel('absolute error in f','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('old','new');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,0.008,timestr,'FontSize',40);
% end;
% 
% figure('OuterPosition',[300,300,800,600],'Color','w');
% axes('FontSize',40,'FontName','Times New Roman');    
fNorm2_n = zeros(1,5);
for i=1:5
    tim = tvec(i)*0.001;
    S=(5/(2*tim+5)) * exp(-5*xsvol.^2/(10+tim));    
    fNorm2_n(i) = norm(S-f_mms_n(:,i),2);    
end
fNorm2_n
clear all
%close all
% t=0.001:0.001:5;
% plot(t,fNorm2,t,fNorm2_n,'--','LineWidth',2);
% ylabel('norm of the abs-err in f','FontSize',40,'FontName','Times New Roman');
% xlabel('time','FontSize',40,'FontName','Times New Roman');
% legend('old','new');
% axis tight
% 
% %% Convective case
% 
% a = dlmread('outputs/Conv_u_n.txt',' ', 2, 0); 
% for i=1:257 
%     b(:,i)=a((a(:,1)==i-1),4);
% end
% u_conv_n=b';
% clear ans a i b
% a = dlmread('outputs/Conv_u.txt',' ', 2, 0); 
% for i=1:257 
%     b(:,i)=a((a(:,1)==i-1),4);
% end
% u_conv=b';
% clear ans a i b
% a = dlmread('outputs/MMS_xxvol.txt',' ', 2, 0); 
% xxvol=a(:,1);
% clear a
% 
% tvec=[1,1700,3400,5000];
% 
% for i=1:length(tvec)
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     tim = tvec(i)*0.001;
%     S=ones(257,1);
%     plot(xxvol,S,xxvol,u_conv(:,tvec(i)),'--',xxvol,u_conv_n(:,tvec(i)),'--','LineWidth',2);
%     axis([-15 15 0.9 1.1]);
%     ylabel('x-velocity','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('manufactured','old','new');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,1.08,timestr,'FontSize',40);
% 
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     plot(xxvol,abs(S-u_conv(:,tvec(i))),xxvol,abs(S-u_conv_n(:,tvec(i))),'--','LineWidth',2);
%     axis([-15 15 0 0.01]);
%     ylabel('absolute error in x-velocity ','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('old','new');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,0.008,timestr,'FontSize',40);
% end;
% 
% figure('OuterPosition',[300,300,800,600],'Color','w');
% axes('FontSize',40,'FontName','Times New Roman');    
% uNorm2_n = zeros(1,5000);
% uNorm2 = zeros(1,5000);
% for i=1:5000
%     tim = i*0.001;
%     S=ones(257,1);
%     uNorm2_n(i) = norm(S-u_conv_n(:,i),inf);
%     uNorm2(i) = norm(S-u_conv(:,i),inf);
% end
% t=0.001:0.001:5;
% plot(t,uNorm2,t,uNorm2_n,'--','LineWidth',2);
% ylabel('norm of the abs-err in x-vel','FontSize',40,'FontName','Times New Roman');
% xlabel('time','FontSize',40,'FontName','Times New Roman');
% legend('old','new');
% axis tight
% 
% 
% a = dlmread('outputs/Conv_f_n.txt',' ', 2, 0); 
% for i=1:257 
%     b(:,i)=a((a(:,1)==i-1),4);
% end
% f_conv_n=b';
% clear ans a i b
% a = dlmread('outputs/Conv_f.txt',' ', 2, 0); 
% for i=1:257 
%     b(:,i)=a((a(:,1)==i-1),4);
% end
% f_conv=b';
% clear ans a i b
% a = dlmread('outputs/MMS_xsvol.txt',' ', 2, 0); 
% xsvol=a(:,1);
% clear a
% 
% for i=1:length(tvec)
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     tim = tvec(i)*0.001;
%     S=exp(-(xsvol-tim).^2/2);
%     plot(xsvol,S,xsvol,f_conv(:,tvec(i)),'--',xsvol,f_conv_n(:,tvec(i)),'--','LineWidth',2);
%     axis([-15 15 0 1.0]);
%     ylabel('mixture fraction','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend1=legend('manufactured','old','new');
%     set(legend1,'Location','SouthWest')
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,0.9,timestr,'FontSize',40);
% 
%     figure('OuterPosition',[300,300,800,600],'Color','w');
%     axes('FontSize',40,'FontName','Times New Roman');    
%     plot(xsvol,abs(S-f_conv(:,tvec(i))),xsvol,abs(S-f_conv_n(:,tvec(i))),'--','LineWidth',2);
%     axis([-15 15 0 0.01]);
%     ylabel('absolute error in f','FontSize',40,'FontName','Times New Roman');
%     xlabel('x','FontSize',40,'FontName','Times New Roman');
%     legend('old','new');
%     t=num2str(tvec(i)*0.001);
%     timestr=['t = ',t];
%     text(-14,0.008,timestr,'FontSize',40);
% end;
% 
% figure('OuterPosition',[300,300,800,600],'Color','w');
% axes('FontSize',40,'FontName','Times New Roman');    
% fNorm2_n = zeros(1,5000);
% fNorm2 = zeros(1,5000);
% for i=1:5000
%     tim = i*0.001;
%     S=exp(-(xsvol-tim).^2/2);
%     fNorm2_n(i) = norm(S-f_conv_n(:,i),2);
%     fNorm2(i) = norm(S-f_conv(:,i),2);
% end
% t=0.001:0.001:5;
% plot(t,fNorm2,t,fNorm2_n,'--','LineWidth',2);
% ylabel('norm of the abs-err in f','FontSize',40,'FontName','Times New Roman');
% xlabel('time','FontSize',40,'FontName','Times New Roman');
% legend1=legend('old','new');
% set(legend1,'Location','NorthWest')
% axis tight
