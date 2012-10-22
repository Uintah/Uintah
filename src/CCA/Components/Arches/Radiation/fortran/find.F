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
