      subroutine setbc( u, dlo, dhi, vlo, vhi )

c $Id$

      implicit none

c  Purpose - set boundary conditions for zero boundary flux

c  Arguments-
c    Input:   vlo - index pair for lower left corner of
c                      valid domain
c             vhi - index pair for upper right corner of
c                      valid domain
c             dlo - index pair for lower left corner of
c                      declared domain
c             dhi - index pair for upper right corner of
c                      declared domain
c    Output:  u     - ghost cells updated to reflect boundary conditions

c  Dummy arguments - 

      integer dhi(2), vhi(2)
      integer dlo(2), vlo(2)

      double precision u(dlo(1):dhi(1),dlo(2):dhi(2))

c  Local variables -

      integer i
      integer ihi
      integer ilo
      integer j
      integer jhi
      integer jlo
      integer nx
      integer ny

      double precision h
      double precision xi
      double precision yj

      double precision zero,       one,       two
      parameter      ( zero=0.0d0, one=1.0d0, two=2.0d0 )

c  Start of executable code-

c  Left and right boundaries.

      ilo = vlo(1)-1
      ihi = vhi(1)+1
      jlo = vlo(2)-1
      jhi = vhi(2)+1
      h = one/float(jhi-jlo+1)
      yj = h/two
      do j = vlo(2), vhi(2)
         u(ilo,j) = zero
         u(ihi,j) = zero
         yj = yj + h
      end do

c  Top and bottom boundaries.

      jlo = vlo(2)-1
      jhi = vhi(2)+1
      ilo = vlo(1)-1
      ihi = vhi(1)+1
      h = one/float(ihi-ilo+1)
      xi = h/two
      do i = vlo(1), vhi(1)
         u(i,jlo) = zero
         u(i,jhi) = zero
         xi = xi + h
      end do

      return 
      end
