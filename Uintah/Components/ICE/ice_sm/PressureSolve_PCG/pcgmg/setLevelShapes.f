      subroutine setLevelShapes( nx, ny, levels, 
     &                           p_dlo, p_dhi, p_vlo, p_vhi )

c  $Id$

      implicit none

c  Purpose-  Set gridfunction shapes on all grid levels.

c  Arguments-
c    Input:   nx     - integer
c                      number of gridpoints in the x-direction
c             ny     - integer
c                      number of gridpoints in the y-direction
c             levels - integer
c                      number of levels in the grid hierarchy

c    Output:  p_dlo  - integer array declared 2 x levels+1
c                      same as u_dlo, but for cell-centered locations
c             p_dhi  - integer array declared 2 x levels+1
c                      same as u_dhi, but for cell-centered locations
c             p_vlo  - integer array declared 2 x levels+1
c                      same as u_vlo, but for cell-centered locations
c             p_vhi  - integer array declared 2 x levels+1
c                      same as u_vhi, but for cell-centered locations
c  Dummy arguments-

      integer nx
      integer ny
      integer levels

      integer p_dlo(2,0:levels), p_dhi(2,0:levels)
      integer p_vlo(2,0:levels), p_vhi(2,0:levels)

c  Local variables-

      integer l
      integer m
      integer n

c  Start of executable code-

      m = nx
      n = ny
      do l = 0, levels
         call setshape( m, n,
     &                  p_dlo(1,l), p_dhi(1,l), p_vlo(1,l), p_vhi(1,l) )
         m = m/2
         n = n/2
      end do

      return
      end
