C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
	SUBROUTINE eico2(TK, PLA, ECO2)

        IMPLICIT NONE

	double precision TEMP(7)
	double precision AA(7)
	double precision BB(7)
	double precision CC(7)

	double precision TR,TK,PLA,PL,ALOGPL,ECO2,TMP,A,B,C

	integer ITEMP

        SAVE

	DATA TEMP /  500.0d0,   1000.0d0,   1500.0d0,   2000.0d0, 
     X                     2500.0d0,   3000.0d0,   5000.0d0 /
	DATA AA / -1.86189d0, -2.04189d0, -1.89565d0, -1.83814d0,
     X                   -1.94996d0, -2.14107d0, -3.01083d0 /
	DATA BB /  0.18990d0,  0.22385d0,  0.18790d0,  0.21910d0,
     X                    0.25793d0,  0.30845d0,  0.38896d0 /
	DATA CC / -0.02613d0, -0.02004d0, -0.03814d0, -0.04054d0,
     X                   -0.03876d0, -0.02882d0, -0.02949d0 /

C	TR = DEG R, TK = DEG K; PLA = FT-ATM

        ITEMP = 1
	TR = TK * 1.8d0

C	Set up bounds, interpolate Hottel chart to find CO2 emissivity

	IF (PLA .GE. 0.001d0) THEN
	  PL = MIN (PLA, 5.0d0)
	  TMP = MAX (TR, 500.0d0)
	  TMP = MIN (TR, 5000.0d0)
	  CALL FIND( TMP, A, TEMP, AA, 7, ITEMP)
	  CALL FIND( TMP, B, TEMP, BB, 7, ITEMP)
	  CALL FIND( TMP, C, TEMP, CC, 7, ITEMP)
	  ALOGPL = DLOG(PL)
	  ECO2 = EXP(A + ALOGPL * (B + ALOGPL * C))
	ELSE
	  ECO2 = 0.0d0
	END IF

	RETURN
	END






