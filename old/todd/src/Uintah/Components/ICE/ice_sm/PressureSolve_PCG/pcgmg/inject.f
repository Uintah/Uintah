      subroutine inject( pf, f_dlo, f_dhi, f_vlo, f_vhi,
     &                   pc, c_dlo, c_dhi, c_vlo, c_vhi )

c  $Id$

      implicit none

c  Purpose - aggregate fine grid data to a coarse grid

c  Arguments -
c    Input:   pf  - fine grid vector
c             f_dlo - index pair for lower left corner of declared 
c                     fine grid data
c             f_dhi - index pair for upper right corner of declared 
c                     fine grid data
c             f_vlo - index pair for lower left corner of valid 
c                     fine grid data
c             f_vhi - index pair for upper right corner of valid 
c                     fine grid data
c             c_dlo - index pair for lower left corner of declared 
c                     coarse grid data
c             c_dhi - index pair for upper right corner of declared 
c                     coarse grid data
c             c_vlo - index pair for lower left corner of valid 
c                     coarse grid data
c             c_vhi - index pair for upper right corner of valid 
c                     coarse grid data
c
c    Output:  pc - coarse grid vector

c  Arguments-

      integer c_dhi(2), c_vhi(2)
      integer c_dlo(2), c_vlo(2)
      integer f_dhi(2), f_vhi(2)
      integer f_dlo(2), f_vlo(2)

      double precision pc(c_dlo(1):c_dhi(1),c_dlo(2):c_dhi(2))
      double precision pf(f_dlo(1):f_dhi(1),f_dlo(2):f_dhi(2))

c  Local variables -

      integer ic
      integer if
      integer jc
      integer jf

c  Start of executable code -

      jf = f_vlo(2)
      do jc = c_vlo(2), c_vhi(2)
         if = f_vlo(1)
         do ic = c_vlo(1), c_vhi(2)
            pc(ic,jc) = pf(if,jf)   + pf(if+1,jf) +
     &                  pf(if,jf+1) + pf(if+1,jf+1)
            if = if + 2
         end do
         jf = jf + 2
      end do

      return
      end
