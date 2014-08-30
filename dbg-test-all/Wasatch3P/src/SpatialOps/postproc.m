clear; clc; close all;
doX = 1;
doY = 1;
doZ = 1;

test2der = 0;

if( doX )
   % x-data
   f=mmread('fx.mm'); x=mmread('x.mm');
   fx=mmread('fintx.mm'); xx=mmread('xx.mm');
   fx2=mmread('fintx2.mm');
   df=mmread('dfdx.mm');
   d2f=mmread('d2fdx2.mm');
   subplot(3,1,1); plot(x,f,'k.',xx,fx,'ro',x,fx2,'bs'); legend('orig','interp','interp2');
   subplot(3,1,2); plot(xx,cos(xx),'kx',xx,df,'rs'); legend('dx','num');  axis([0,pi,-1,1]);
   subplot(3,1,3); plot(x,-sin(x),'kx',x,d2f,'rs'); legend('d2x','num');  axis([0,pi,-1,1]);
   title('x results');
end

if ( doY )
   % y-data
   figure;
   y=mmread('y.mm'); f=mmread('fy.mm');
   yy=mmread('yy.mm'); fy=mmread('finty.mm');
   fy2=mmread('finty2.mm');
   df=mmread('dfdy.mm');
   d2f=mmread('d2fdy2.mm');
   subplot(3,1,1);  plot(y,f,'k.',yy,fy,'ro',y,fy2,'bs');     legend('orig','interp','interp2');
   subplot(3,1,2);  plot(yy,cos(yy),'kx',yy,df,'rs'); legend('exact grad','num grad');  axis([0,pi,-1,1]);
   subplot(3,1,3);  plot(y,-sin(y),'kx',y,d2f,'bs');   legend('exact div','num div');   axis([0,pi,-1,1]);
   title('y results');
end

if( doZ )
   % z-data
   figure;
   z=mmread('z.mm'); f=mmread('fz.mm');
   zz=mmread('zz.mm'); fz=mmread('fintz.mm');
   fz2=mmread('fintz2.mm');
   df=mmread('dfdz.mm');
   d2f=mmread('d2fdz2.mm');
   subplot(3,1,1);  plot(z,f,'k.',zz,fz,'ro',z,fz2,'bs');     legend('orig','interp','interp2');
   subplot(3,1,2);  plot(zz,cos(zz),'kx',zz,df,'rs'); legend('exact grad','num grad');  axis([0,pi,-1,1]);
   subplot(3,1,3);  plot(z,-sin(z),'kx',z,d2f,'bs');   legend('exact div','num div');   axis([0,pi,-1,1]);
   title('z results');
end