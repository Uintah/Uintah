c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_4loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision isic, trace, denc
      double precision scalarc,enthalpyc, reactscalarc
      double precision uvelc, vvelc, wvelc

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
               uvelc = ccuvel(i,j,k)
               vvelc = ccvvel(i,j,k)
               wvelc = ccwvel(i,j,k)
               denc = den(i,j,k)

               isic = isi(i,j,k)

                 if (d_reactingFlow) then
                   scalarc = scalar(i,j,k)
                   scalarBeta1(i,j,k) = denc*isic*scalarGrad1(i,j,k)
                   scalarBeta2(i,j,k) = denc*isic*scalarGrad2(i,j,k)
                   scalarBeta3(i,j,k) = denc*isic*scalarGrad3(i,j,k)
                   rhofu(i,j,k) = denc*scalarc*uvelc
                   rhofv(i,j,k) = denc*scalarc*vvelc
                   rhofw(i,j,k) = denc*scalarc*wvelc
                 endif
                 if (d_calcEnthalpy) then
                   enthalpyc = enthalpy(i,j,k)
                   enthalpyBeta1(i,j,k) = denc*isic*
     *                                    enthalpyGrad1(i,j,k)
                   enthalpyBeta2(i,j,k) = denc*isic*
     *                                    enthalpyGrad2(i,j,k)
                   enthalpyBeta3(i,j,k) = denc*isic*
     *                                    enthalpyGrad3(i,j,k)
                   rhoEU(i,j,k) = denc*enthalpyc*uvelc
                   rhoEV(i,j,k) = denc*enthalpyc*vvelc
                   rhoEW(i,j,k) = denc*enthalpyc*wvelc
                 end if
                 if (d_calcReactingScalar) then
                   reactScalarc = reactScalar(i,j,k)
                   reactScalarBeta1(i,j,k) = denc*isic*
     *                                       reactScalarGrad1(i,j,k)
                   reactScalarBeta2(i,j,k) = denc*isic*
     *                                       reactScalarGrad2(i,j,k)
                   reactScalarBeta3(i,j,k) = denc*isic*
     *                                       reactScalarGrad3(i,j,k)
                   rhoRFU(i,j,k) = denc*reactScalarc*uvelc
                   rhoRFV(i,j,k) = denc*reactScalarc*vvelc
                   rhoRFW(i,j,k) = denc*reactScalarc*wvelc
                 end if
            END DO
         END DO
      END DO

      RETURN
      END
