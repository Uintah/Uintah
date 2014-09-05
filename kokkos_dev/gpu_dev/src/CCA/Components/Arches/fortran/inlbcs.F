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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/inlbcs_fort.h>
#include <CCA/Components/Arches/fortran/param4.h>
      
c*********************************************************************
c     Locals :
c*********************************************************************
      integer IST, IEND, JST, JEND, KST, KEND

      integer i, j, k
      double precision factor, y,z
      double precision denom, ave_u

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

#include <CCA/Components/Arches/fortran/ramping.h>


C-----------------------------------------------------------------------
C     EAST AND WEST INLETS
C-----------------------------------------------------------------------

C************ SWIRL ONLY ON X- FACES! ************
      if (xminus.or.xplus) then
         DO 120 K = kst,kend
            DO 100 J = jst,jend
               IF (xminus) then
                  if (PCELL(ist-1,J,K).EQ.PFIELD) THEN
                     UU(ist,J,K) = factor*UU(ist-1,J,K) *
     &                    DEN(ist-1,J,K)
     &                    / (PT5*(DEN(ist-1,J,K) + DEN(IST,J,K)))
                    if ( do_swirl ) then 

                      y = J * dy - cent_y
                      z = K * dz + dz/2.0 - cent_z

                      denom = y*y + z*z
                      denom = denom**0.5

                      ave_u = PT5 * ( UU(ist,J,K)+UU(ist-1,J,K) )

                     VV(ist-1,J,K) = -1.0 * z * swirl_no * ave_u / denom

                      y = J * dy + dy/2.0 - cent_y
                      z = K * dz - cent_z

                      denom = y*y + z*z
                      denom = denom**0.5

                      WW(ist-1,J,K) = y * swirl_no * ave_u / denom

                    endif 
                  ENDIF
               endif
               if (xplus) then
                  IF (PCELL(iend+1,J,K).EQ.PFIELD) THEN
                     UU(iend+1,J,K) = factor*UU(iend+2,J,K) *
     &                    DEN(iend+1,J,K)
     &                    / (PT5*(DEN(iend+1,J,K) + DEN(IEND,J,K)))
                  ENDIF
               endif
 100        CONTINUE
 120     CONTINUE
      endif
C--------------------------------------------------------------------
C     NORTH AND SOUTH INLETS
C--------------------------------------------------------------------
      if (yminus.or.yplus) then
         DO 220 K = kst,kend
            DO 200 I = ist,iend
               if (yminus) then
                  IF (PCELL(I,jst-1,K).EQ.PFIELD) THEN
                     VV(I,jst,K) = factor*VV(I,jst-1,K) * 
     &                    DEN(I,jst-1,K)
     &                    / (PT5*(DEN(I,jst-1,K) + DEN(I,JST,K)))
                  ENDIF
               endif
               if (yplus) then
                  IF (PCELL(I,jend+1,K).EQ.PFIELD) THEN
                     VV(I,jend+1,K) = factor*VV(I,jend+2,K) * 
     &                    DEN(I,jend+1,K)
     &                    / (PT5*(DEN(I,jend+1,K) + DEN(I,JEND,K)))
                  ENDIF
               endif
 200        CONTINUE
 220     CONTINUE
      endif
C--------------------------------------------------------------------
C     TOP AND BOTTOM INLETS
C--------------------------------------------------------------------
      if (zminus.or.zplus) then
         DO 320 J = jst,jend
            DO 300 I = ist,iend
               if (zminus) then
                  IF (PCELL(I,J,kst-1).EQ.PFIELD) THEN
                     WW(I,J,kst) = factor*WW(I,J,kst-1)
     &                    * DEN(I,J,kst-1)
     &                    / (PT5*(DEN(I,J,kst-1) + DEN(I,J,KST)))
                  ENDIF
               endif
               if (zplus) then
                  IF (PCELL(I,J,kend+1).EQ.PFIELD) THEN
                     WW(I,J,kend+1) = factor*WW(I,J,kend+2)
     &                    * DEN(I,J,kend+1)
     &                    / (PT5*(DEN(I,J,kend+1) + DEN(I,J,KEND)))
                  endif
               ENDIF
 300        CONTINUE
 320     CONTINUE
      endif
      
C--------------------------------------------------------------------
      
      RETURN
      END

c*********************************************************************
c
c Revision 1.6  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.5  2000/10/05 16:39:49  rawat
c modified bcs for multi-patch
c
c Revision 1.4  2000/07/07 23:07:47  rawat
c added inlet bc's
c
c Revision 1.3  2000/07/03 05:30:22  bbanerje
c Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
c
c Revision 1.2  2000/07/02 05:47:33  bbanerje
c Uncommented all PerPatch and CellInformation stuff.
c Updated array sizes in inlbcs.F
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
