      subroutine setshape( nx, ny,
     &                     p_dlo, p_dhi, p_vlo, p_vhi )

c  $Id$

      implicit none

c  Purpose:  Initialize arrays that decribe shape of 
c            gridfunctions on a staggered grid.  Both a
c            "declared" size (maximum problem size) and a
c            "valid" size (subgrid on which solution resides)
c            are defined.

c  Arguments-
c    Input:   nx    - integer
c                     number of gridpoints in x-direction
c             ny    - integer

c    Output:  p_dlo - integer array of length 2
c                     coordinates of lower left corner of
c                     declared region
c             p_dhi - integer array of length 2
c                     coordinates of upper right corner of
c                     declared region
c             p_vlo - integer array of length 2
c                     coordinates of lower left corner of
c                     valid region
c             p_vhi - integer array of length 2
c                     coordinates of upper right corner of
c                     valid region

c  Dummy arguments-

      integer nx
      integer ny

      integer p_dlo(2), p_dhi(2)
      integer p_vlo(2), p_vhi(2)

c  Start of executable code-

      p_dlo(1) = 0
      p_dlo(2) = 0
      p_dhi(1) = nx+1
      p_dhi(2) = ny+1
      p_vlo(1) = 1
      p_vlo(2) = 1
      p_vhi(1) = nx
      p_vhi(2) = ny

      return
      end
