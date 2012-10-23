C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
      SUBROUTINE KAYENTA_CHK ( UI, GC, DC)

      CHARACTER*6 IAM
      PARAMETER( IAM = 'KAYENTA_CHK' )

      CALL LOGMES('This model is only available to licensees')
      CALL FATERR(IAM,'See Jim Guilkey for more information')

      END

      SUBROUTINE KAYENTA_RXV(UI,GC,DC,
     &  NX, NAMEA, KEYA, RINIT, RDIM, IADVCT, ITYPE)

      CHARACTER*6 IAM
      PARAMETER( IAM = 'KAYENTA_RXV' ) 
      CALL LOGMES('This model is only available to licensees')
      CALL FATERR(IAM,'See Jim Guilkey for more information')

      END

      SUBROUTINE KAYENTA_CALC( NBLK, NINSV, DT, UI, GC, DC,
     $                                   SIGARG, D, SVARG, USM)

      CHARACTER*12 IAM
      PARAMETER (IAM='KAYENTA_CALC')

      CALL LOGMES('This model is only available to licensees')
      CALL FATERR(IAM,'See Jim Guilkey for more information')

      END
