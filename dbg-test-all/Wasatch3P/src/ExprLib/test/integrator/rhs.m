function dcdt = rhs( t, c )
k = 1.0;
dcdt = -k*c;