c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_2loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision shatij1c, shatij2c, shatij3c
      double precision shatij4c, shatij5c, shatij6c
      double precision ishaticur
      double precision mij1cur, mij2cur, mij3cur
      double precision mij4cur, mij5cur, mij6cur
      double precision delta, filter, filterdencur, trace, lijtrace
      double precision filterRhoUcur, filterRhoVcur
      double precision filterRhoWcur
      double precision lij1cur, lij2cur, lij3cur
      double precision lij4cur, lij5cur, lij6cur

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
C              ignoring the trace
C              trace = (shatij1c+shatij2c+shatij3c)/3.0D0
               trace = 0.0D0

               IsImag(i,j,k) = IsI(i,j,k) 

               MIJ1cur = 2.0D0*filter*filter*(betaHATIJ1(i,j,k)-
     -                          2.0D0*2.0D0*filterdencur*
     *                          IshatIcur*(shatij1c-trace))
               MIJ2cur = 2.0D0*filter*filter*(betaHATIJ2(i,j,k)-
     -                          2.0D0*2.0D0*filterdencur*
     *                          IshatIcur*(shatij2c-trace))
               MIJ3cur = 2.0D0*filter*filter*(betaHATIJ3(i,j,k)-
     -                          2.0D0*2.0D0*filterdencur*
     *                          IshatIcur*(shatij3c-trace))
               MIJ4cur = 2.0D0*filter*filter*(betaHATIJ4(i,j,k)-
     -                          2.0D0*2.0D0*filterdencur*
     *                          IshatIcur*shatij4c)
               MIJ5cur = 2.0D0*filter*filter*(betaHATIJ5(i,j,k)-
     -                          2.0D0*2.0D0*filterdencur*
     *                          IshatIcur*shatij5c)
               MIJ6cur = 2.0D0*filter*filter*(betaHATIJ6(i,j,k)-
     -                          2.0D0*2.0D0*filterdencur*
     *                          IshatIcur*shatij6c)


C compute Leonard stress tensor
C index 0: L11, 1:L22, 2:L33, 3:L12, 4:L13, 5:L23
               filterRhoUcur = filterRhoU(i,j,k)
               filterRhoVcur = filterRhoV(i,j,k)
               filterRhoWcur = filterRhoW(i,j,k)
               LIJ1cur = filterRhoUU(i,j,k)-
     -                   filterRhoUcur*filterRhoUcur/filterdencur
               LIJ2cur = filterRhoVV(i,j,k)-
     -                   filterRhoVcur*filterRhoVcur/filterdencur
               LIJ3cur = filterRhoWW(i,j,k)-
     -                   filterRhoWcur*filterRhoWcur/filterdencur
               LIJ4cur = filterRhoUV(i,j,k)-
     -                   filterRhoUcur*filterRhoVcur/filterdencur
               LIJ5cur = filterRhoUW(i,j,k)-
     -                   filterRhoUcur*filterRhoWcur/filterdencur
               LIJ6cur = filterRhoVW(i,j,k)-
     -                   filterRhoVcur*filterRhoWcur/filterdencur

C Explicitly making LIJ traceless here
C Actually, trace has been ignored	  
C               LIJtrace = (LIJ1cur + LIJ2cur + LIJ3cur)/3.0D0
               LIJtrace = 0.0D0
               LIJ1cur = LIJ1cur - LIJtrace
               LIJ2cur = LIJ2cur - LIJtrace
               LIJ3cur = LIJ3cur - LIJtrace

C compute the magnitude of ML and MM
               MLI(i,j,k) = MIJ1cur*LIJ1cur+MIJ2cur*LIJ2cur+
     +                      MIJ3cur*LIJ3cur+2.0D0*(MIJ4cur*LIJ4cur+
     +                      MIJ5cur*LIJ5cur+MIJ6cur*LIJ6cur)
C calculate absolute value of the grid strain rate
               MMI(i,j,k) = MIJ1cur*MIJ1cur+MIJ2cur*MIJ2cur+
     +                      MIJ3cur*MIJ3cur+2.0D0*(MIJ4cur*MIJ4cur+
     +                      MIJ5cur*MIJ5cur+MIJ6cur*MIJ6cur)

            END DO
         END DO
      END DO

      RETURN
      END
