      subroutine copy_from_gridfcn( gridfcn, dlo, dhi, vlo, vhi, 
     &                              vec, len )

c	$Id$	

      implicit none

c  Purpose- Copy data from gridfunction data structure to vector 
c           data structure.

c  Arguments-
c    Input:   u     - double precision array
c                     gridfunction
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

c    Output:  vec   - double precision array
c                     vector data

c  Dummy arguments-

      integer len

      integer dlo(2), dhi(2)
      integer vlo(2), vhi(2)

      double precision vec(len)
      double precision gridfcn(dlo(1):dhi(1),dlo(2):dhi(2))

c  Local variables-

      integer i, j, k

c  Local variables-

      k = 0
      do j = vlo(2), vhi(2)
         do i = vlo(1), vhi(1)
            k = k + 1
            vec(k) = gridfcn(i,j)
         end do
      end do

      if ( k .ne. len )  then
         print*, "ERROR in copying from gridfunction!"
         print*, 'Asked to copy ', len, ' items.'
         print*, 'Copied ', k, ' items'
      endif

      return
      end

