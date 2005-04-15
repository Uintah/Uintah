      subroutine relax( v, v_dlo, v_dhi, v_vlo, v_vhi, 
     &                  f, f_dlo, f_dhi,
     &                  ap, ae, aw, an, as, op_lo, op_hi, nu )

c  $Id$

      implicit none

c  Purpose - apply nu applications of Gauss-Seidel relaxation
c            to a 2-dimensional problem.

c  Arguments -
c    Input:   v      - current approximate solution
c             f      - rhs
c             [fv]_dlo - index pairs for lower left corner of
c                         declared domain
c             [fv]_dhi - index pairs for upper right corner of
c                         declared domain
c             v_vlo     - index pair for lower left corner of
c                         valid domain
c             v_vhi     - index pair for upper right corner of
c                         valid domain
c             a[pewns]  - stencil coefficients
c             op_lo     - index pair for lower left corner of
c                         valid domain
c             op_hi     - index pair for upper right corner of
c                         valid domain
c             nu     - number of times to apply gauss-seidel relaxation

c    Output:  v      - updated coarse grid correction

      integer nu

      integer f_dhi(2)
      integer f_dlo(2)
      integer op_hi(2)
      integer op_lo(2)
      integer v_dhi(2), v_vhi(2)
      integer v_dlo(2), v_vlo(2)

      double precision f(f_dlo(1):f_dhi(1),f_dlo(2):f_dhi(2))
      double precision v(v_dlo(1):v_dhi(1),v_dlo(2):v_dhi(2))

      double precision ap(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision ae(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision aw(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision an(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision as(op_lo(1):op_hi(1),op_lo(2):op_hi(2))

c  Local variables -

      integer i
      integer j
      integer k

c  Start of executable code -

      do k = 1, nu

c  Forward sweep.

         do j = v_vlo(2), v_vhi(2)
            do i = v_vlo(1), v_vhi(1)
               v(i,j) = ( f(i,j) - ( as(i,j)*v(i,j-1) +
     &                               aw(i,j)*v(i-1,j) +
     &                               an(i,j)*v(i,j+1) +
     &                               ae(i,j)*v(i+1,j) )  )/ap(i,j)
            end do
         end do

c  Normalize.

         call normalize( v, v_dlo, v_dhi, v_vlo, v_vhi )

c Backward sweep.

         do j = v_vhi(2), v_vlo(2), -1
            do i = v_vhi(1), v_vlo(1), -1
               v(i,j) = ( f(i,j) - ( as(i,j)*v(i,j-1) +
     &                               aw(i,j)*v(i-1,j) +
     &                               an(i,j)*v(i,j+1) +
     &                               ae(i,j)*v(i+1,j) ) )/ap(i,j)
            end do
         end do

c  Normalize.

         call normalize( v, v_dlo, v_dhi, v_vlo, v_vhi )

      end do

      return
      end
