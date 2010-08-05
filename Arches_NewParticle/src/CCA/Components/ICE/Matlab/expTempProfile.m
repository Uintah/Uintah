clear all;
close all;

x = 0:0.01:2;
dir = [1,0,0]
coeff = 10;
xmin = 0.5
xmax = 1.0


for ( i = 1:length(x))
  temp_CC(i) = 300.0;
  if( x(i) > xmin & x(i) < xmax )
      
    stuff = coeff - (xmax - xmin)/( (x(i) - xmin) * ( xmax - x(i) ) );
    
    temp_CC(i)  = temp_CC(i) +  dir(1) *  exp(  stuff + 1e-100  );
  end
end

plot(x, temp_CC)
