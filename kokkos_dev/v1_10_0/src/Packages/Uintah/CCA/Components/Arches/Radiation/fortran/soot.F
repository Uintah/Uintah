      SUBROUTINE SOOT(I,J,K,OMEGA,FV,OPL,TEMP,ABSKP)

      IMPLICIT NONE

      double precision omega,fv,opl,temp,abskp,ambda,ff
      integer i,j,k

      AMBDA=10000./OMEGA

C     ABSORPTION COEF. IS BASED UPON MEASUREMENTS OF DALZELL AND
C     SAROFIM,J.H.T, vol.91, #1, p.100-104,1969,(spectral)
C     Soot model is based on a correlation given by Sarofim and Hottel     
C     (for details of citation see radcoef.F) (global)

      FF=7./AMBDA

C      ABSKP=FF*FV*1.E06*OPL

      ABSKP = MIN(25.0d0,((4.0d0/OPL) 
     &* DLOG(1.0d0 + 350.0d0*FV*TEMP*OPL)))
 
      RETURN
      END
