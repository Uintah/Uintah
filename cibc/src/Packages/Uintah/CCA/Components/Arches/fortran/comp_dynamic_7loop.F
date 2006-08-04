c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_7loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision shatij1c, shatij2c, shatij3c
      double precision shatij4c, shatij5c, shatij6c
      double precision ishaticur
      double precision delta, filter, filterdencur
      double precision filterRhoUcur, filterRhoVcur
      double precision filterRhoWcur
      double precision enthalpylx, enthalpyly, enthalpylz
      double precision enthalpymx, enthalpymy, enthalpymz, filterrhoecur

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
               delta = sew(i)*sns(j)*stb(k)
               filter = delta**(1.0D0/3.0D0)

C              test filter width is assumed to be twice that of the basic filter
C              needs following modifications:
Ca)            make the test filter work for anisotropic grid
Cb)            generalize the filter operation
               shatij1c = SHATIJ1(i,j,k)
               shatij2c = SHATIJ2(i,j,k)
               shatij3c = SHATIJ3(i,j,k)
               shatij4c = SHATIJ4(i,j,k)
               shatij5c = SHATIJ5(i,j,k)
               shatij6c = SHATIJ6(i,j,k)
               IshatIcur=sqrt(
     *                      2.0D0*(shatij1c*shatij1c+shatij2c*shatij2c+
     +                      shatij3c*shatij3c+2.0D0*(shatij4c*shatij4c+ 
     +                      shatij5c*shatij5c+shatij6c*shatij6c)))
               filterdencur = filterrho(i,j,k)

                   filterRhoEcur = filterRhoE(i,j,k)
                   enthalpyLX =  filter*filter*
     *                           (enthalpyBetaHat1(i,j,k)-
     -                            2.0*2.0*filterDencur*IshatIcur*
     *                            filterEnthalpyGrad1(i,j,k))
                   enthalpyLY =  filter*filter*
     *                           (enthalpyBetaHat2(i,j,k)-
     -                            2.0*2.0*filterDencur*IshatIcur*
     *                            filterEnthalpyGrad2(i,j,k))
                   enthalpyLZ =  filter*filter*
     *                           (enthalpyBetaHat3(i,j,k)-
     -                            2.0*2.0*filterDencur*IshatIcur*
     *                            filterEnthalpyGrad3(i,j,k))
                   enthalpyMX = filterRhoEU(i,j,k) -
     -                          filterRhoEcur*filterRhoUcur/filterDencur
                   enthalpyMY = filterRhoEV(i,j,k) -
     -                          filterRhoEcur*filterRhoVcur/filterDencur
                   enthalpyMZ = filterRhoEW(i,j,k) -
     -                          filterRhoEcur*filterRhoWcur/filterDencur
                   enthalpyNum(i,j,k) = enthalpyLX*enthalpyLX +
     +                                  enthalpyLY*enthalpyLY +
     +                                  enthalpyLZ*enthalpyLZ
                   enthalpyDenom(i,j,k) = enthalpyMX*enthalpyLX +
     +                                    enthalpyMY*enthalpyLY +
     +                                    enthalpyMZ*enthalpyLZ
            END DO
         END DO
      END DO

      RETURN
      END
