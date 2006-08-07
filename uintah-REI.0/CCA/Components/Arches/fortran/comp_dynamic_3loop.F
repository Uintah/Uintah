c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_3loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision sewcur, snscur, stbcur
      double precision uep, uwp, unp, usp, utp, ubp
      double precision vnp, vsp, vep, vwp, vtp, vbp
      double precision wnp, wsp, wep, wwp, wtp, wbp
      double precision uvelcur, vvelcur, wvelcur
      double precision uvelxp1, vvelyp1, wvelzp1
      double precision fuep, fuwp, funp, fusp, futp, fubp
      double precision fvnp, fvsp, fvep, fvwp, fvtp, fvbp
      double precision fwtp, fwbp, fwep, fwwp, fwnp, fwsp
      double precision fuvelcur, fvvelcur, fwvelcur
      double precision fuvelxp1, fvvelyp1, fwvelzp1

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
               sewcur = sew(i)
               snscur = sns(j)
               stbcur = stb(k)

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

C              calculate the grid strain rate tensor
               SIJ1(i,j,k) = (uep-uwp)/sewcur
               SIJ2(i,j,k) = (vnp-vsp)/snscur
               SIJ3(i,j,k) = (wtp-wbp)/stbcur
               SIJ4(i,j,k) = 0.5D0*((unp-usp)/snscur + (vep-vwp)/sewcur)
               SIJ5(i,j,k) = 0.5D0*((utp-ubp)/stbcur + (wep-wwp)/sewcur)
               SIJ6(i,j,k) = 0.5D0*((vtp-vbp)/stbcur + (wnp-wsp)/snscur)

               fuvelcur = filterRhoU(i,j,k)/
     /                    (0.5*(filterRho(i,j,k) + filterRho(i-1,j,k)))
               fvvelcur = filterRhoV(i,j,k)/
     /                    (0.5*(filterRho(i,j,k) + filterRho(i,j-1,k)))
               fwvelcur = filterRhoW(i,j,k)/
     /                    (0.5*(filterRho(i,j,k) + filterRho(i,j,k-1)))
               fuvelxp1 = filterRhoU(i+1,j,k)/
     /                    (0.5*(filterRho(i,j,k) + filterRho(i+1,j,k)))
               fvvelyp1 = filterRhoV(i,j+1,k)/
     /                    (0.5*(filterRho(i,j,k) + filterRho(i,j+1,k)))
               fwvelzp1 = filterRhoW(i,j,k+1)/
     /                    (0.5*(filterRho(i,j,k) + filterRho(i,j,k+1)))

               fuep = fuvelxp1
               fuwp = fuvelcur
               funp = 0.25*(fuvelxp1 + fuvelcur + filterRhoU(i+1,j+1,k)/
     /               (0.5*(filterRho(i,j+1,k) + filterRho(i+1,j+1,k))) +
     +                filterRhoU(i,j+1,k)/
     /               (0.5*(filterRho(i,j+1,k) + filterRho(i-1,j+1,k))))
               fusp = 0.25*(fuvelxp1 + fuvelcur + filterRhoU(i+1,j-1,k)/
     /               (0.5*(filterRho(i,j-1,k) + filterRho(i+1,j-1,k))) +
     +                filterRhoU(i,j-1,k)/
     /               (0.5*(filterRho(i,j-1,k) + filterRho(i-1,j-1,k))))
               futp = 0.25*(fuvelxp1 + fuvelcur + filterRhoU(i+1,j,k+1)/
     /               (0.5*(filterRho(i,j,k+1) + filterRho(i+1,j,k+1))) +
     +                filterRhoU(i,j,k+1)/
     /               (0.5*(filterRho(i,j,k+1) + filterRho(i-1,j,k+1))))
               fubp = 0.25*(fuvelxp1 + fuvelcur + filterRhoU(i+1,j,k-1)/
     /               (0.5*(filterRho(i,j,k-1) + filterRho(i+1,j,k-1))) +
     +                filterRhoU(i,j,k-1)/
     /               (0.5*(filterRho(i,j,k-1) + filterRho(i-1,j,k-1))))

               fvnp = fvvelyp1
               fvsp = fvvelcur
               fvep = 0.25*(fvvelyp1 + fvvelcur + filterRhoV(i+1,j+1,k)/
     /               (0.5*(filterRho(i+1,j,k) + filterRho(i+1,j+1,k))) +
     +                filterRhoV(i+1,j,k)/
     /               (0.5*(filterRho(i+1,j,k) + filterRho(i+1,j-1,k))))
               fvwp = 0.25*(fvvelyp1 + fvvelcur + filterRhoV(i-1,j+1,k)/
     /               (0.5*(filterRho(i-1,j,k) + filterRho(i-1,j+1,k))) +
     +                filterRhoV(i-1,j,k)/
     /               (0.5*(filterRho(i-1,j,k) + filterRho(i-1,j-1,k))))
               fvtp = 0.25*(fvvelyp1 + fvvelcur + filterRhoV(i,j+1,k+1)/
     /               (0.5*(filterRho(i,j,k+1) + filterRho(i,j+1,k+1))) +
     +                filterRhoV(i,j,k+1)/
     /               (0.5*(filterRho(i,j,k+1) + filterRho(i,j-1,k+1))))
               fvbp = 0.25*(fvvelyp1 + fvvelcur + filterRhoV(i,j+1,k-1)/
     /               (0.5*(filterRho(i,j,k-1) + filterRho(i,j+1,k-1))) +
     +                filterRhoV(i,j,k-1)/
     /               (0.5*(filterRho(i,j,k-1) + filterRho(i,j-1,k-1))))

               fwtp = fwvelzp1
               fwbp = fwvelcur
               fwep = 0.25*(fwvelzp1 + fwvelcur + filterRhoW(i+1,j,k+1)/
     /               (0.5*(filterRho(i+1,j,k) + filterRho(i+1,j,k+1))) +
     +                filterRhoW(i+1,j,k)/
     /               (0.5*(filterRho(i+1,j,k) + filterRho(i+1,j,k-1))))
               fwwp = 0.25*(fwvelzp1 + fwvelcur + filterRhoW(i-1,j,k+1)/
     /               (0.5*(filterRho(i-1,j,k) + filterRho(i-1,j,k+1))) +
     +                filterRhoW(i-1,j,k)/
     /               (0.5*(filterRho(i-1,j,k) + filterRho(i-1,j,k-1))))
               fwnp = 0.25*(fwvelzp1 + fwvelcur + filterRhoW(i,j+1,k+1)/
     /               (0.5*(filterRho(i,j+1,k) + filterRho(i,j+1,k+1))) +
     +                filterRhoW(i,j+1,k)/
     /               (0.5*(filterRho(i,j+1,k) + filterRho(i,j+1,k-1))))
               fwsp = 0.25*(fwvelzp1 + fwvelcur + filterRhoW(i,j-1,k+1)/
     /               (0.5*(filterRho(i,j-1,k) + filterRho(i,j-1,k+1))) +
     +                filterRhoW(i,j-1,k)/
     /               (0.5*(filterRho(i,j-1,k) + filterRho(i,j-1,k-1))))

C              calculate the filtered strain rate tensor
               filterSIJ1(i,j,k) = (fuep-fuwp)/sewcur
               filterSIJ2(i,j,k) = (fvnp-fvsp)/snscur
               filterSIJ3(i,j,k) = (fwtp-fwbp)/stbcur
               filterSIJ4(i,j,k) = 0.5*((funp-fusp)/snscur +
     +                                  (fvep-fvwp)/sewcur)
               filterSIJ5(i,j,k) = 0.5*((futp-fubp)/stbcur +
     +                                  (fwep-fwwp)/sewcur)
               filterSIJ6(i,j,k) = 0.5*((fvtp-fvbp)/stbcur + 
     +                                  (fwnp-fwsp)/snscur)


            END DO
         END DO
      END DO

      RETURN
      END
