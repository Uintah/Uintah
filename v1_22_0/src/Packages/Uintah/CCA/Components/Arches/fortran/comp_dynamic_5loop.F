c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_5loop_fort.h>

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
      double precision scalarlx, scalarly, scalarlz
      double precision scalarmx, scalarmy, scalarmz, filterrhofcur

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

                   filterRhoFcur = filterRhoF(i,j,k)
                   scalarLX =  filter*filter*
     *                         (scalarBetaHat1(i,j,k)-
     -                          2.0*2.0*filterDencur*IshatIcur*
     *                          filterScalarGrad1(i,j,k))
                   scalarLY =  filter*filter*
     *                         (scalarBetaHat2(i,j,k)-
     -                          2.0*2.0*filterDencur*IshatIcur*
     *                          filterScalarGrad2(i,j,k))
                   scalarLZ =  filter*filter*
     *                         (scalarBetaHat3(i,j,k)-
     -                          2.0*2.0*filterDencur*IshatIcur*
     *                          filterScalarGrad3(i,j,k))
                   scalarMX = filterRhoFU(i,j,k) -
     -                 filterRhoFcur*filterRhoUcur/filterDencur
                   scalarMY = filterRhoFV(i,j,k) -
     -                 filterRhoFcur*filterRhoVcur/filterDencur
                   scalarMZ = filterRhoFW(i,j,k) -
     -                 filterRhoFcur*filterRhoWcur/filterDencur
                   scalarNum(i,j,k) = scalarLX*scalarLX +
     +                                scalarLY*scalarLY +
     +                                scalarLZ*scalarLZ
                   scalarDenom(i,j,k) = scalarMX*scalarLX +
     +                                  scalarMY*scalarLY +
     +                                  scalarMZ*scalarLZ
            END DO
         END DO
      END DO

      RETURN
      END
