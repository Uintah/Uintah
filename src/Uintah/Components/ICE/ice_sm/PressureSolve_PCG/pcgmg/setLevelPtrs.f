      subroutine setLevelPtrs( nx, ny, levels, g )

c  $Id$

      implicit none

c  Purpose-  Set pointers to grid levels.

c  Arguments-
c    Input:   nx     - integer
c                      number of gridpoints in the x-direction
c             ny     - integer
c                      number of gridpoints in the y-direction
c             levels - integer
c                      number of levels in the grid hierarchy

c    Output:  g      - integer array of length levels+1
c                      pointers to grid levels

c  Dummy arguments-

      integer nx
      integer ny
      integer levels

      integer g(0:levels)

c  Local variables-

      integer l
      integer m
      integer n

c  Start of executable code-

      m = nx
      n = ny
      g(0) = 1
      do l = 1, levels
         g(l) = g(l-1) + (m+2)*(n+2)
         m = m/2
         n = n/2
      end do

      return
      end
