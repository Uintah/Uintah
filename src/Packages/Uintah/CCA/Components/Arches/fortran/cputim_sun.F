      SUBROUTINE cputim(TIM)
C--------------------------------------------------------------------
C     THIS SUBROUTINE RETURNED THE ELAPSED CPU TIME SINCE THE 
C          LAST LOGIN BY THE USER.  TIM IS THE CPU TIME IN SECONDS 
C--------------------------------------------------------------------
      REAL*4 DUMMY(2)
      TIM=ETIME(DUMMY)
      RETURN
      END
