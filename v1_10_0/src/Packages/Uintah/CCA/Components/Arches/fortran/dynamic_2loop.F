c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/dynamic_2loop_fort.h>

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
      double precision delta, filter
      double precision filterUVelcur, filterVVelcur, filterWVelcur
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


               IsImag(i,j,k) = IsI(i,j,k) 


               MIJ1cur = 2.0D0*(filter*filter)*(betaHATIJ1(i,j,k)-
     -                          2.0D0*2.0D0*IshatIcur*shatij1c)
               MIJ2cur = 2.0D0*(filter*filter)*(betaHATIJ2(i,j,k)-
     -                          2.0D0*2.0D0*IshatIcur*shatij2c)
               MIJ3cur = 2.0D0*(filter*filter)*(betaHATIJ3(i,j,k)-
     -                          2.0D0*2.0D0*IshatIcur*shatij3c)
               MIJ4cur = 2.0D0*(filter*filter)*(betaHATIJ4(i,j,k)-
     -                          2.0D0*2.0D0*IshatIcur*shatij4c)
               MIJ5cur = 2.0D0*(filter*filter)*(betaHATIJ5(i,j,k)-
     -                          2.0D0*2.0D0*IshatIcur*shatij5c)
               MIJ6cur = 2.0D0*(filter*filter)*(betaHATIJ6(i,j,k)-
     -                          2.0D0*2.0D0*IshatIcur*shatij6c)


C compute Leonard stress tensor
C index 0: L11, 1:L22, 2:L33, 3:L12, 4:L13, 5:L23
               filterUVelcur = filterUVel(i,j,k)
               filterVVelcur = filterVVel(i,j,k)
               filterWVelcur = filterWVel(i,j,k)
               LIJ1cur = (filterUU(i,j,k)-filterUVelcur*filterUVelcur)
               LIJ2cur = (filterVV(i,j,k)-filterVVelcur*filterVVelcur)
               LIJ3cur = (filterWW(i,j,k)-filterWVelcur*filterWVelcur)
               LIJ4cur = (filterUV(i,j,k)-filterUVelcur*filterVVelcur)
               LIJ5cur = (filterUW(i,j,k)-filterUVelcur*filterWVelcur)
               LIJ6cur = (filterVW(i,j,k)-filterVVelcur*filterWVelcur)

C compute the magnitude of ML and MM
               MLI(i,j,k) = MIJ1cur*LIJ1cur+MIJ2cur*LIJ2cur+
     +                      MIJ3cur*LIJ3cur+2.0D0*(MIJ4cur*LIJ4cur+
     +                      MIJ5cur*LIJ5cur+MIJ6cur*LIJ6cur)
               MMI(i,j,k) = MIJ1cur*MIJ1cur+MIJ2cur*MIJ2cur+
     +                      MIJ3cur*MIJ3cur+2.0D0*(MIJ4cur*MIJ4cur+
     +                      MIJ5cur*MIJ5cur+MIJ6cur*MIJ6cur)
C calculate absolute value of the grid strain rate
            END DO
         END DO
      END DO

      RETURN
      END
