      subroutine normalize( u, dlo, dhi, vlo, vhi )

c  $Id$

      implicit none

      integer dhi(2), vhi(2)
      integer dlo(2), vlo(2)

      double precision u(dlo(1):dhi(1),dlo(2):dhi(2))

c  Local variables-

      integer i
      integer j

      double precision avgVal

c  Start of executable code-

       avgVal = 0.0d0
       do j = vlo(2), vhi(2)
          do i = vlo(1), vhi(1)
             avgVal = avgVal + u(i,j)
         end do
      end do
      avgVal = avgVal/float(vhi(2)-vlo(2)+1)
      avgVal = avgVal/float(vhi(1)-vlo(1)+1)

      do j = vlo(2), vhi(2)
         do i = vlo(1), vhi(1)
            u(i,j) = u(i,j) - avgVal
         end do
      end do

      return 
      end
