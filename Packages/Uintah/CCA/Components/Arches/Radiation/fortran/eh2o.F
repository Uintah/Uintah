      SUBROUTINE EH2O(OMEGA,TEMP,SD,SDWEAK)

      DOUBLE PRECISION OMEGA,SDWEAK,TEMP,TTEMP,
     &W1,WW,TT,T1,TW

      DOUBLE PRECISION SD(6,376)

      INTEGER I,J

      IF (OMEGA.GE.9300..OR.OMEGA.LT.50.)GO TO 200

C     Read value from Ludwig's table and interpolate

      J=(OMEGA-25.)/25.
      TTEMP=TEMP
      IF(TEMP.GE.2500.) TEMP=2499.99
      IF(TEMP.LT.300.) TEMP=300.
      I=TEMP/500. +1
      IF(I.EQ.2.AND.TEMP.LT.600.) I=1
      W1=25.+25.*FLOAT(J)
      WW=(OMEGA-W1)/25.
      IF(I.GT.2) GO TO 75
      IF(I.EQ.1) TT=(TEMP-300.)/300.
      IF(I.EQ.2) TT=(TEMP-600.)/400.
      GO TO 100
75    T1=FLOAT(I-1)*500.
      TT=(TEMP-T1)/500.
100   TW=TT*WW
      SDWEAK=SD(I,J)*(1.-TT-WW+TW)+SD(I+1,J)*(TT-TW)+SD(I,J+1)*(WW-TW)
     & +SD(I+1,J+1)*TW

      TEMP=TTEMP
      RETURN
200   CONTINUE
      SDWEAK=0.

      RETURN
      END
