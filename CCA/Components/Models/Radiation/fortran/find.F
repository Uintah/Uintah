      SUBROUTINE find(ARG,ANS,X,Y,NPTS,I)

      IMPLICIT NONE

      double precision X(50) 
      double precision Y(50)

      double precision ARG,ANS

      integer NPTS,I

101    IF (ARG .LE. X(1)) THEN
         ANS = Y(1)
       ELSE IF (ARG .GE. X(NPTS)) THEN
         ANS = Y(NPTS)
        ELSE IF (ARG .LE. X(I)) THEN
         ANS = Y(I) - (Y(I) - Y(I-1)) * (X(I) - ARG) / (X(I) - X(I-1))
        ELSE
        I = I + 1
        GO TO 101
       END IF

       RETURN
       END
