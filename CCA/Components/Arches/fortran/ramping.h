c common inlcude file with ramping function
      factor = 1.0d0
c      if (time .lt. 2.0d0) then
c         factor = time*0.5d0
c         if (time.lt.0.02d0) then
c            factor = 0.000001d0*factor
c         elseif (time.lt.0.1d0) then
c            factor = 0.001d0*factor
c         elseif (time.lt.0.2d0) then
c            factor = 0.01d0*factor
c         elseif (time.lt.0.3d0) then
c            factor = 0.1d0*factor
c         elseif (time.lt.0.4d0) then
c            factor = 0.5d0*factor
c         elseif (time.lt.1.0d0) then
c            factor = 0.8d0*factor
c         endif
c      endif
