       subroutine setval( a, dlo, dhi, val )

c  $Id$

      implicit none

c  Purpose- Initialize a gridfunction to a constant value.

c  Arguments-
c    Input:   val   - double precision
c                     initialization value
c             dlo   - integer array of length 2
c                     coordinates of lower left corner of u.
c             dhi   - integer array of length 2
c                     coordinates of upper right corner of u.

c   Output:   a - double precision array
c                 gridfunction initialized to val

c  Dummy arguments-

       integer dlo(2), dhi(2)

       double precision val

       double precision a(dlo(1):dhi(1),dlo(2):dhi(2))

c  Local variables-

       integer i, j

c  Start of executable code-

       do j = dlo(2), dhi(2)
          do i = dlo(1), dhi(1)
             a(i,j) = val
          end do
       end do

       return
       end
