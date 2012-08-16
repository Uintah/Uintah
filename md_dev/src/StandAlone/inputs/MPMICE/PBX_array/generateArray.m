#! /usr/bin/octave -qf
%_________________________________
% This octave file generates the location for center of an array of cylinders
% and prints out the ups specifications that can be copied into a ups file
% There are two different configurations tha a user can select, loose and
% tight packings

clear all;
close all;
format short e;


%__________________________________
%  user defined variables 
nRows    = 8
nCols    = 8
radius   = 0.0254
zLo      = 0.0
zHi      = 0.01
loosePacking = false;
tightPacking = true;

%__________________________________
x = zeros(nRows, nCols);
y = zeros(nRows, nCols);


%__________________________________
% loose packing
if(loosePacking)
  printf("loosepacking\n");
  % coordinates for the first row and column
  for(c = 2:nCols)
    x(1,c) = x(1,c-1) + 2*radius;
  end
  for(r = 2:nRows)
    y(r,1) = y(r-1,1) + 2*radius;
  end

  % coordinates for the remaining rows and columns
  for (r = 2:nRows)
    for (c = 2:nCols)
     x(r,c) =  x(r,c-1) + 2*radius;
     y(r,c) =  y(r-1,c) + 2*radius;
    end
  end
endif

%__________________________________
%   tight packing
% On all even rows there is a shift in the x direction
if(tightPacking)
  printf( "tightpacking\n");
  
  % x coordinates vertically along 1st column
  for(r = 1:nRows)
    if(rem(r,2) == 0)  %even rows
      x(r,1) = radius;
    endif
  end  
  
  % y coordinate horizontally along 1st row
  for(c = 1:nCols)
    if(rem(c,2) == 0)
      y(1,c) = 0;
    endif
  end  
  
  % x coordinate along 1 row
  for(c = 2:nCols)
    x(1,c) = x(1,c-1) + 2*radius;
  end
  
  % y coordinate along 1 column
  for(r = 2:nRows)
    y(r,1) = y(r-1,1) + sin(60*pi/180) * 2*radius;
  end
  
  % coordinates for the remaining rows and columns
  for (r = 2:nRows)
    for (c = 2:nCols)
      x(r,c) =  x(r,c-1) +  2 * radius;
      y(r,c) =  y(r-1,c) +  sin(60*pi/180) * 2 * radius;
    end
  end
endif

%__________________________________
% output to the screen so

printf("--------------------------------------\n")
printf("  cut below and add to MPM section \n")
counter = 0;
for (r = 1:nRows)
  for (c = 1:nCols)
    printf("<cylinder label = \"%d\">\n",counter)
    printf("    <bottom>  [%f, %f, %f]  </bottom>\n",x(r,c),y(r,c),zLo) 
    printf("    <top>     [%f, %f, %f]  </top>\n",x(r,c),y(r,c),zHi)
    printf("    <radius>       %f       </radius>\n",radius)
    printf("</cylinder>\n") 
    counter +=1;
  end
end


printf("--------------------------------------\n")
printf("  cut below and add to ICE section \n")
counter = 0;
for (r = 1:nRows)
  for (c = 1:nCols)
    printf("<cylinder label = \"%d\">     </cylinder>\n",counter)
    counter +=1;
  end
end

exit
