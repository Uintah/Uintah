c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/grdgrf_fort.h>
#else
      SUBROUTINE grdgrf(indexLow, indexHigh,
     &     JPST, JPED, JSST, JSED,
     &     LRDGD, LCORD, LPRIN,
     &     ierr)
C------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     SUBROUTINE DISPLAYS HELPFUL INFORMATION IN LOG FILE
C------------------------------------------------------------------
      
      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer JPST, JPED, JSST, JSED
      logical LRDGD, LCORD, LPRIN
      integer ierr

c*********************************************************************
c     Locals :
c*********************************************************************
      integer NI, NJ, NK

c*********************************************************************
c     Start :
c*********************************************************************
      NI = indexHigh(1)
      NJ = indexHigh(2)
      NK = indexHigh(3)

      IF (LRDGD) THEN
        IF (LCORD) THEN
          IF (LPRIN) WRITE(3,3333) NI,NJ,NK
          WRITE(6,3333) NI,NJ,NK
        ELSE
          IF (LPRIN) WRITE(3,4444) NK,NI,NJ
          WRITE(6,4444) NK,NI,NJ
        END IF
      ELSE
        IF (LCORD) THEN
          IF (LPRIN) WRITE (3,1111) NK,NI,JSST,JPED,JSED,NJ
          WRITE (6,1111) NK,NI,JSST,JPED,JSED,NJ
        ELSE
          IF (LPRIN) WRITE (3,2222) NI,NK,JSST,JSED,JPST,JPED,NJ
          WRITE (6,2222) NI,NK,JSST,JSED,JPST,JPED,NJ
        END IF
      END IF
      RETURN
 1111    FORMAT(' '/20X,' NK = ',I3/46X,'NI =',I3/
     &        15X,'WWWWWWW---------------------------OO'/ 
     &        13X,'WW',7X,'WW',27X,'OO'/
     &        12X,'W',11X,'W',28X,'O'/
     &        11X,'W',5X,'SSS',5X,'W',28X,'O'/
     &        10X,'W',4X,'SS   SS',4X,'W',28X,'O'/
     &        10X,'W   S',7X,'S   W',28X,'O'/
     &        9X,'W   S   PPP   S   W',28X,'O'/
     &        9X,'W   S  P   P  S   W',28X,'O'/
     &        9X,'W   S  P   P  S   W',28X,'O'/
     &        9X,'W   S   PPP   S   W',28X,'O'/
     &        10X,'W   S',7X,'S   W',28X,'O'/
     &        10X,'W',4X,'SS   SS',4X,'W',28X,'O'/
     &        11X,'W',5X,'SSS',5X,'W',28X,'O'/
     &        12X,'W',11X,'W',28X,'O'/
     &        13X,'WW',7X,'WW',27X,'OO'/
     &        15X,'WWWWWWW---------------------------OO'/ 
     &        4X,'J = ',10X,I1,1X,I1,2X,I2,2X,I2/)
 2222    FORMAT(' ',/,15X,I3,/,18X,45('_')/
     &        17X,'/',44X,'/I',/,16X,'/',44X,'/ I'/
     &        15X,'/',44X,'/  I',/,14X,'/',44X,'/   I'/
     &        13X,'/',44X,'/    I',/,12X,'/',44X,'/',5X,'I'/
     &        8X,I3,'/',44('_'),'/',6X,'I'/
     &        10X,'I',44X,'I',7X,'I'/
     &        10X,'I',8X,I3,20X,I3,10X,'I',7X,'I'/
     &        10X,'I',10X,'.',22('_'),'.',10X,'I',7X,'I'/
     &        10X,'I',10X,'I',4X,I3,6X,I3,6X,'I',10X,'I',7X,'I'/
     &        10X,'I',10X,'I',6X,'.________.',6X,'I',10X,'I',7X,'I'/
     &        10X,'I',10X,'I',6X,'I',8X,'I',6X,'I',10X,'I',6X,'/'/
     &        10X,'I',10X,'I',6X,'I________I',6X,'I',10X,'I',5X,'/'/
     &        10X,'I',10X,'I',22X,'I',10X,'I',4X,'/'/
     &        10X,'I',10X,'I',22('_'),'I',10X,'I   /'/
     &        10X,'I',44X,'I  /',/,10X,'I',44X,'I /'/ 
     &        10X,'I',44('_'),'I/',/,52X,I3,/)
 3333    FORMAT(' ',/,16X,'_______________________________'/
     &        15X,'/     \ ',23X,' \ '/
     &        14X,'/       \ ',5X,'NI = ',I3,10X,' \ '/
     &        13X,'|         |',25X,'|',/,
     &        13X,'|         |',5X,'NJ = ',I3,12X,'|'/
     &        13X,'|         |',25X,'|'/
     &        14X,'\       /',6X,'NK = ',I3,11X,'/'/
     &        15X,'\_____/_________________________/'/)
 4444    FORMAT(' '/26X,25('_')/
     &        25X,'/',24X,'/|'/24X,'/',24X,'/ |'/
     &        23X,'/',24X,'/  |'/
     &        22X,'/',15X,'NK = ',I3,' /',3X,'|'/
     &        21X,'/',24('_'),'/',4X,'|'/
     &        20X,'|',24X,'|',5X,'|'/
     &        20X,'|',24X,'|',5X,'|'/
     &        20X,'|',24X,'|',5X,'|'/
     &        20X,'|',24X,'|',4X,'/'/
     &        20X,'|',24X,'|   / NI = ',I3/
     &        20X,'|',24X,'|  /'/
     &        20X,'|',24X,'|',1X,'/'/
     &        20X,'|',24('_'),'|/'/13X,'NJ = ',I3/)
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
