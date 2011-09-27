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
