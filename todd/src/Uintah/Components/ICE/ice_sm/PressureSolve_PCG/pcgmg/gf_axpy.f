      subroutine gf_axpy( y, ydlo, ydhi, yvlo, yvhi,
     &                    alpha,
     &                    x, xdlo, xdhi, xvlo, xvhi )

c  $Id$ 

      implicit none

c  Purpose-  BLAS _axpy operation (gridfunction data format):
c
c               y <- y + alpha*x

c  Arguments-
c    Input:   y     - double precision array
c                     gridfunction
c             ydlo  - integer array of length 2
c                     coordinates of lower left corner of y.
c             ydhi  - integer array of length 2
c                     coordinates of upper right corner of y.
c             yvlo  - integer array of length 2
c                     coordinates of lower left corner of valid part of y.
c             yvhi  - integer array of length 2
c                     coordinates of upper right corner of valid part of y.
c             alpha - double precision
c                     scalar used in linear combination
c             x     - double precision array
c                     gridfunction
c             xdlo  - integer array of length 2
c                     coordinates of lower left corner of x.
c             xdhi  - integer array of length 2
c                     coordinates of upper right corner of x.
c             xvlo  - integer array of length 2
c                     coordinates of lower left corner of valid part of x.
c             xvhi  - integer array of length 2
c                     coordinates of upper right corner of valid part of x.

c    Output:  y     - double precision array
c                     overwritten by result of linear combination.

c  Dummy arguments-

      integer xdlo(2), xdhi(2), xvlo(2), xvhi(2)
      integer ydlo(2), ydhi(2), yvlo(2), yvhi(2)

      double precision alpha

      double precision x(xdlo(1):xdhi(1),xdlo(2):xdhi(2))
      double precision y(ydlo(1):ydhi(1),ydlo(2):ydhi(2))

c  Local variables-

      integer ix, iy
      integer jx, jy

c  Start of executable code-

      jx = xvlo(2)
      do jy = yvlo(2), yvhi(2)
         ix = xvlo(1)
         do iy = yvlo(1), yvhi(1)
            y(iy,jy) = y(iy,jy) + alpha*x(ix,jx)
            ix = ix + 1
         end do
         jx = jx + 1
      end do

      return
      end

