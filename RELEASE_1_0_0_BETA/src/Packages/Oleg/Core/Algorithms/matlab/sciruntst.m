hname='gauss:5505';

a=transport([2 1],hname);
a
a=a*2;
transport([2 2],hname,a);

save -ascii a.dat a;
