hname='127.0.0.1:5505';

a=transport([2 1],hname);
a

 a=a*2;
% a=sparse([0.1 0 1.5; 0 -0.2 0; 0 4.7 0.3]');

transport([2 2],hname,a);

