      subroutine gemv( f, f_dlo, f_dhi, v, v_dlo, v_dhi,
     &                 r, r_dlo, r_dhi, r_vlo, r_vhi,
     &                 ap, ae, aw, an, as, op_lo, op_hi, alpha, beta )

c  $Id$ 

      implicit none

c  Purpose - general matrix-vector product for a 5 point stencil acting
c            on 2-dimensional gridfunctions.

c  r = alpha*f + beta*Av

c  Arguments -
c    Input:   f       - rhs
c             v       - current approximation
c             [fvr]_dlo - index pairs for lower left corner of
c                         declared domain
c             [fvr]_dhi - index pairs for upper right corner of
c                         declared domain
c             r_vlo     - index pair for lower left corner of
c                         valid domain
c             r_vhi     - index pair for upper right corner of
c                         valid domain
c             a[pewns]  - stencil coefficients
c             op_lo     - index pair for lower left corner of
c                         valid domain
c             op_hi     - index pair for upper right corner of
c                         valid domain

c    Output:  r         - alpha*f + beta*Av

c  Dummy arguments - 

      integer f_dhi(2)
      integer f_dlo(2)
      integer op_hi(2)
      integer op_lo(2)
      integer r_dhi(2), r_vhi(2)
      integer r_dlo(2), r_vlo(2)
      integer v_dhi(2)
      integer v_dlo(2)

      double precision alpha
      double precision beta

      double precision f(f_dlo(1):f_dhi(1),f_dlo(2):f_dhi(2))
      double precision r(r_dlo(1):r_dhi(1),r_dlo(2):r_dhi(2))
      double precision v(v_dlo(1):v_dhi(1),v_dlo(2):v_dhi(2))

      double precision ap(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision ae(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision aw(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision an(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision as(op_lo(1):op_hi(1),op_lo(2):op_hi(2))

c  Local variables -

      integer i
      integer j

c  Start of executable code-

      do j = r_vlo(2), r_vhi(2)
         do i = r_vlo(1), r_vhi(1)
            r(i,j) = alpha*f(i,j) + beta*( as(i,j)*v(i,j-1) +
     &                                     aw(i,j)*v(i-1,j) +
     &                                     an(i,j)*v(i,j+1) +
     &                                     ae(i,j)*v(i+1,j) +
     &                                     ap(i,j)*v(i,j)   )
         end do
      end do

      return

c  End of gemv.

      end
