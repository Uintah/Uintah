c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/dynamic_3loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision uep, uwp, unp, usp, utp, ubp
      double precision vnp, vsp, vep, vwp, vtp, vbp
      double precision wnp, wsp, wep, wwp, wtp, wbp
      double precision uvelcur, vvelcur, wvelcur
      double precision uvelxp1, vvelyp1, wvelzp1
      double precision sewcur, snscur, stbcur

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO K = KST, KEND
         DO J = JST, JEND
            DO I = IST, IEND   

               uvelcur = uVel(i,j,k)
               vvelcur = vVel(i,j,k)
               wvelcur = wVel(i,j,k)
               uvelxp1 = uVel(i+1,j,k)
               vvelyp1 = vVel(i,j+1,k)
               wvelzp1 = wVel(i,j,k+1)

               uep = uvelxp1
               uwp = uvelcur
               unp = 0.25D0*(uvelxp1 + uvelcur +
     +                       uVel(i+1,j+1,k) + uVel(i,j+1,k))
               usp = 0.25D0*(uvelxp1 + uvelcur +
     +                       uVel(i+1,j-1,k) + uVel(i,j-1,k))
               utp = 0.25D0*(uvelxp1 + uvelcur +
     +                       uVel(i+1,j,k+1) + uVel(i,j,k+1))
               ubp = 0.25D0*(uvelxp1 + uvelcur + 
     +                       uVel(i+1,j,k-1) + uVel(i,j,k-1))

               vnp = vvelyp1
               vsp = vvelcur
               vep = 0.25D0*(vvelyp1 + vvelcur +
     +                       vVel(i+1,j+1,k) + vVel(i+1,j,k))
               vwp = 0.25D0*(vvelyp1 + vvelcur +
     +                       vVel(i-1,j+1,k) + vVel(i-1,j,k))
               vtp = 0.25D0*(vvelyp1 + vvelcur + 
     +                       vVel(i,j+1,k+1) + vVel(i,j,k+1))
               vbp = 0.25D0*(vvelyp1 + vvelcur +
     +                       vVel(i,j+1,k-1) + vVel(i,j,k-1))

               wtp = wvelzp1
               wbp = wvelcur
               wep = 0.25D0*(wvelzp1 + wvelcur + 
     +                       wVel(i+1,j,k+1) + wVel(i+1,j,k))
               wwp = 0.25D0*(wvelzp1 + wvelcur +
     +                       wVel(i-1,j,k+1) + wVel(i-1,j,k))
               wnp = 0.25D0*(wvelzp1 + wvelcur + 
     +                       wVel(i,j+1,k+1) + wVel(i,j+1,k))
               wsp = 0.25D0*(wvelzp1 + wvelcur +
     +                       wVel(i,j-1,k+1) + wVel(i,j-1,k))

C     calculate the grid strain rate tensor

               sewcur = sew(i)
               snscur = sns(j)
               stbcur = stb(k)

               SIJ1(i,j,k) = (uep-uwp)/sewcur
               SIJ2(i,j,k) = (vnp-vsp)/snscur
               SIJ3(i,j,k) = (wtp-wbp)/stbcur
               SIJ4(i,j,k) = 0.5D0*((unp-usp)/snscur + (vep-vwp)/sewcur)
               SIJ5(i,j,k) = 0.5D0*((utp-ubp)/stbcur + (wep-wwp)/sewcur)
               SIJ6(i,j,k) = 0.5D0*((vtp-vbp)/stbcur + (wnp-wsp)/snscur)

            END DO
         END DO
      END DO

      RETURN
      END
