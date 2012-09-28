C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
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
