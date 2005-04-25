c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/comp_dynamic_6loop_fort.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      double precision sewcur, snscur, stbcur
      double precision scalarxp, scalarxm, scalaryp
      double precision scalarym, scalarzp, scalarzm
      double precision scalarcurr
      double precision fscalarxp, fscalarxm, fscalaryp
      double precision fscalarym, fscalarzp, fscalarzm
      double precision fscalarcurr
      double precision enthalpyxp, enthalpyxm, enthalpyyp
      double precision enthalpyym, enthalpyzp, enthalpyzm
      double precision enthalpycurr
      double precision fenthalpyxp, fenthalpyxm, fenthalpyyp
      double precision fenthalpyym, fenthalpyzp, fenthalpyzm
      double precision fenthalpycurr
      double precision reactScalarxp, reactScalarxm, reactScalaryp
      double precision reactScalarym, reactScalarzp, reactScalarzm
      double precision reactScalarcurr
      double precision freactScalarxp, freactScalarxm, freactScalaryp
      double precision freactScalarym, freactScalarzp, freactScalarzm
      double precision freactScalarcurr

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

                 if (d_reactingFlow) then
                   scalarcurr = scalar(i,j,k)

                   scalarxm = 0.5*(scalarcurr+scalar(i-1,j,k))
                   scalarxp = 0.5*(scalarcurr+scalar(i+1,j,k))
                   scalarym = 0.5*(scalarcurr+scalar(i,j-1,k))
                   scalaryp = 0.5*(scalarcurr+scalar(i,j+1,k))
                   scalarzm = 0.5*(scalarcurr+scalar(i,j,k-1))
                   scalarzp = 0.5*(scalarcurr+scalar(i,j,k+1))

                   scalarGrad1(i,j,k) = (scalarxp-scalarxm)/sewcur
                   scalarGrad2(i,j,k) = (scalaryp-scalarym)/snscur
                   scalarGrad3(i,j,k) = (scalarzp-scalarzm)/stbcur

                   fscalarcurr = filterRhoF(i,j,k)/filterRho(i,j,k)

                   fscalarxm = 0.5*(fscalarcurr+
     +                         filterRhoF(i-1,j,k)/filterRho(i-1,j,k))
                   fscalarxp = 0.5*(fscalarcurr+
     +                         filterRhoF(i+1,j,k)/filterRho(i+1,j,k))
                   fscalarym = 0.5*(fscalarcurr+
     +                         filterRhoF(i,j-1,k)/filterRho(i,j-1,k))
                   fscalaryp = 0.5*(fscalarcurr+
     +                         filterRhoF(i,j+1,k)/filterRho(i,j+1,k))
                   fscalarzm = 0.5*(fscalarcurr+
     +                         filterRhoF(i,j,k-1)/filterRho(i,j,k-1))
                   fscalarzp = 0.5*(fscalarcurr+
     +                         filterRhoF(i,j,k+1)/filterRho(i,j,k+1))

                   filterScalarGrad1(i,j,k) = (fscalarxp-fscalarxm)/
     /                                        sewcur
                   filterScalarGrad2(i,j,k) = (fscalaryp-fscalarym)/
     /                                        snscur
                   filterScalarGrad3(i,j,k) = (fscalarzp-fscalarzm)/
     /                                        stbcur
                 end if
                 if (d_calcEnthalpy) then
                   enthalpycurr = enthalpy(i,j,k)

                   enthalpyxm = 0.5*(enthalpycurr+enthalpy(i-1,j,k))
                   enthalpyxp = 0.5*(enthalpycurr+enthalpy(i+1,j,k))
                   enthalpyym = 0.5*(enthalpycurr+enthalpy(i,j-1,k))
                   enthalpyyp = 0.5*(enthalpycurr+enthalpy(i,j+1,k))
                   enthalpyzm = 0.5*(enthalpycurr+enthalpy(i,j,k-1))
                   enthalpyzp = 0.5*(enthalpycurr+enthalpy(i,j,k+1))

                   enthalpyGrad1(i,j,k) = (enthalpyxp-enthalpyxm)/sewcur
                   enthalpyGrad2(i,j,k) = (enthalpyyp-enthalpyym)/snscur
                   enthalpyGrad3(i,j,k) = (enthalpyzp-enthalpyzm)/stbcur

                   fenthalpycurr = filterRhoE(i,j,k)/filterRho(i,j,k)

                   fenthalpyxm = 0.5*(fenthalpycurr+
     +                           filterRhoE(i-1,j,k)/filterRho(i-1,j,k))
                   fenthalpyxp = 0.5*(fenthalpycurr+
     +                           filterRhoE(i+1,j,k)/filterRho(i+1,j,k))
                   fenthalpyym = 0.5*(fenthalpycurr+
     +                           filterRhoE(i,j-1,k)/filterRho(i,j-1,k))
                   fenthalpyyp = 0.5*(fenthalpycurr+
     +                           filterRhoE(i,j+1,k)/filterRho(i,j+1,k))
                   fenthalpyzm = 0.5*(fenthalpycurr+
     +                           filterRhoE(i,j,k-1)/filterRho(i,j,k-1))
                   fenthalpyzp = 0.5*(fenthalpycurr+
     +                           filterRhoE(i,j,k+1)/filterRho(i,j,k+1))

                   filterEnthalpyGrad1(i,j,k) = 
     =                      (fenthalpyxp-fenthalpyxm)/sewcur
                   filterEnthalpyGrad2(i,j,k) =
     =                      (fenthalpyyp-fenthalpyym)/snscur
                   filterEnthalpyGrad3(i,j,k) =
     =                      (fenthalpyzp-fenthalpyzm)/stbcur
                 end if
                 if (d_calcReactingScalar) then
                   reactScalarcurr = reactScalar(i,j,k)

                   reactScalarxm = 0.5*(reactScalarcurr+
     +                                  reactScalar(i-1,j,k))
                   reactScalarxp = 0.5*(reactScalarcurr+
     +                                  reactScalar(i+1,j,k))
                   reactScalarym = 0.5*(reactScalarcurr+
     +                                  reactScalar(i,j-1,k))
                   reactScalaryp = 0.5*(reactScalarcurr+
     +                                  reactScalar(i,j+1,k))
                   reactScalarzm = 0.5*(reactScalarcurr+
     +                                  reactScalar(i,j,k-1))
                   reactScalarzp = 0.5*(reactScalarcurr+
     +                                  reactScalar(i,j,k+1))

                   reactScalarGrad1(i,j,k) =
     =                      (reactScalarxp-reactScalarxm)/sewcur
                   reactScalarGrad2(i,j,k) =
     =                      (reactScalaryp-reactScalarym)/snscur
                   reactScalarGrad3(i,j,k) =
     =                      (reactScalarzp-reactScalarzm)/stbcur

                   freactScalarcurr = filterRhoRF(i,j,k)/
     /                                filterRho(i,j,k)

                   freactScalarxm = 0.5*(freactScalarcurr+
     +                  filterRhoRF(i-1,j,k)/filterRho(i-1,j,k))
                   freactScalarxp = 0.5*(freactScalarcurr+
     +                  filterRhoRF(i+1,j,k)/filterRho(i+1,j,k))
                   freactScalarym = 0.5*(freactScalarcurr+
     +                  filterRhoRF(i,j-1,k)/filterRho(i,j-1,k))
                   freactScalaryp = 0.5*(freactScalarcurr+
     +                  filterRhoRF(i,j+1,k)/filterRho(i,j+1,k))
                   freactScalarzm = 0.5*(freactScalarcurr+
     +                  filterRhoRF(i,j,k-1)/filterRho(i,j,k-1))
                   freactScalarzp = 0.5*(freactScalarcurr+
     +                  filterRhoRF(i,j,k+1)/filterRho(i,j,k+1))

                   filterReactScalarGrad1(i,j,k) =
     =                  (freactScalarxp-freactScalarxm)/sewcur
                   filterReactScalarGrad2(i,j,k) = 
     =                  (freactScalaryp-freactScalarym)/snscur
                   filterReactScalarGrad3(i,j,k) = 
     =                  (freactScalarzp-freactScalarzm)/stbcur
                 end if
            END DO
         END DO
      END DO

      RETURN
      END
