	SUBROUTINE eih2o(TK, PLA, EH2O)

        IMPLICIT NONE

	double precision TEMP(5)
	double precision AA(5)
	double precision BB(5)
	double precision CC(5)

	double precision TR,TK,PLA,PL,ALOGPL,EH2O,TMP,A,B,C

	integer ITEMP

        SAVE

	DATA TEMP / 500.0d0, 1000.0d0, 1500.0d0, 2000.0d0, 5000.0d0 /
	DATA AA /-1.03668d0,-1.26274d0,-1.3823d0,-1.5577d0,-2.7500d0 /
	DATA BB / 0.33619d0, 0.37853d0,0.4274d0,0.4708d0,0.6896d0 /
	DATA CC /-0.04208d0,-0.04386d0,-0.04796d0,-0.053d0,-0.09528d0 /

C	TR = DEG R, TK = DEG K; PLA = FT-ATM

        ITEMP = 1
	TR = TK * 1.8d0

C	Set bounds, interpolate Hottel chart to find H2O emissivity

	IF (PLA .GE. 0.005d0) THEN
	  PL = MIN(PLA, 20.0d0)
	  TMP = MAX(TR, 500.0d0)
	  TMP = MIN(TR, 5000.0d0)
	  CALL FIND(TMP, A, TEMP, AA, 5, ITEMP)
	  CALL FIND(TMP, B, TEMP, BB, 5, ITEMP)
	  CALL FIND(TMP, C, TEMP, CC, 5, ITEMP)
	  ALOGPL = DLOG(PL)
	  EH2O = EXP(A + ALOGPL * (B + ALOGPL * C))
	ELSE
	  EH2O = 0.0d0
	END IF

	RETURN
	END
