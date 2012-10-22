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
C 
C***********************************************************************
      SUBROUTINE TOKENS(N,SA,CA)
C    This routine converts the array of strings SA to a single character
C    stream CA with a pipe (|) separating entries.  For example, suppose
C
C              SA(  1) = 'first string             '
C              SA(  2) = 'a witty saying           '
C              SA(  3) = '                         '
C              SA(  4) = 'last                     '
C
C     Then the output of this routine is
C
C             CA = first string|a witty saying||last|
C
C input
C -----
C   N: number of strings in SA (i.e., the dimension of SA)
C   SA: array of strings
C
C output
C ------
C   CA: single character stream of the strings in SA separated by pipes.
C         BEWARE: it is the responsibility of the calling routine to
C         dimension CA at least as large as N*(1+LEN(SA)).
C
C     written: 04/20/95
C     author:  Rebecca Brannon
C
C calling arguments:
      INTEGER N
      CHARACTER*(*) SA(N)
      CHARACTER*1   CA(*)
C local:
      CHARACTER*1 PIPE,BLANK
      PARAMETER (PIPE='|',BLANK=' ')
      INTEGER I,KNT,NCHR,ICHR
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      KNT=0
      DO 502 I=1,N
         DO 500 NCHR=LEN(SA(I)),1,-1
  500    IF(SA(I)(NCHR:NCHR).NE.BLANK) GO TO 7
    7    DO 501 ICHR=1,NCHR
            KNT=KNT+1
            CA(KNT)=SA(I)(ICHR:ICHR)
  501    CONTINUE
         KNT=KNT+1
         CA(KNT)=PIPE
  502 CONTINUE
      RETURN
      END !SUBROUTINE TOKENS

C*****************************************************************************
      SUBROUTINE PARTOK(N,CA,SA)
C    This routine reverses the operation of subroutine TOKENS.
C    This routine converts character stream CA with a pipe (|)
C    separating entries to an array of strings SA.
C    For example, suppose
C
C             CA = first string|a witty saying||last|
C
C     Then the output of this routine is
C
C              SA(  1) = 'first string             '
C              SA(  2) = 'a witty saying           '
C              SA(  3) = '                         '
C              SA(  4) = 'last                     '
C
C input
C ------
C   N: number of strings in to be extracted from CA
C   CA: single character stream separating strings in SA by pipes.
C
C output
C -----
C   SA: array of strings
C
C BEWARE: it is the responsibility of the calling routine to dimension
C         CA at least as large as N*(1+LEN(SA)).
C
C     written: 04/10/95
C     author:  Rebecca Brannon
C
C calling arguments:
      INTEGER N
      CHARACTER*(*) SA(*)
      CHARACTER*1   CA(*)
C local:
      CHARACTER*1 PIPE
      INTEGER IS,KNT,M,I,MLS
      PARAMETER (PIPE='|')
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     set upper bound on last needed character in CA
C     worst case: each SA is packed.
      MLS=LEN(SA(1))
      M=(MLS+1)*N
C
      IS=1
      IF(IS.GT.N)RETURN
      KNT=0
      SA(IS)=' '
C
      DO 100 I=1,M
         IF(CA(I).EQ.PIPE)THEN
           IS=IS+1
           IF(IS.GT.N)RETURN
           KNT=0
           SA(IS) = ' '
         ELSE IF(KNT+1.GT.MLS)THEN
           GO TO 100
         ELSE
           KNT=KNT+1
           SA(IS)(KNT:KNT)=CA(I)
         END IF
  100 CONTINUE
      RETURN
      END !SUBROUTINE PARTOK
C
