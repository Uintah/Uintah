      subroutine copy_to_gridfcn( vec, len, 
     &                            gridfcn, dlo, dhi, vlo, vhi )

c	$Id$	

      implicit none

c  Purpose- Copy data from vector data structure to gridfunction 
c           data structure.

c  Arguments-
c    Input:   vec   - double precision array
c                     vector data
c             len   - integer
c                     length of vector
c             dlo   - integer array of length 2
c                     coordinates of lower left corner of u.
c             dhi   - integer array of length 2
c                     coordinates of upper right corner of u.
c             vlo   - integer array of length 2
c                     coordinates of lower left corner of valid part of u.
c             vhi   - integer array of length 2
c                     coordinates of upper right corner of valid part of u.

c    Output:  u     - double precision array
c                     gridfunction

c  Dummy arguments-

      integer len

      integer dlo(2), dhi(2)
      integer vlo(2), vhi(2)

      double precision vec(len)
      double precision gridfcn(dlo(1):dhi(1),dlo(2):dhi(2))

c  Local variables-

      integer i, j, k

c  Start of executable code-

      k = 0
      do j = vlo(2), vhi(2)
         do i = vlo(1), vhi(1)
            k = k + 1
            gridfcn(i,j) = vec(k)
         end do
      end do

      if ( k .ne. len ) then
         print*, 'ERROR in copying to gridfunction!'
         print*, 'Asked to copy ', len, ' items.'
         print*, 'Copied ', k, ' items'
      endif

      return
      end
