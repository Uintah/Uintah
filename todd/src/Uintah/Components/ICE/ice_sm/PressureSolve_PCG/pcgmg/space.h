
c  $Id$

c  Space for MG preconditioner.	 

c  Maximum problem size on finest grid.

      integer     MAX_NX,    MAX_NY,        MAX_NXNY
      parameter ( MAX_NX=512, MAX_NY=MAX_NX, MAX_NXNY=MAX_NX*MAX_NY )

c  Maximum number of levels.

      integer     MAX_LEVEL
      parameter ( MAX_LEVEL=9 )

c  Total storage needed for all grid levels.

      integer    MG_SIZE
      parameter (MG_SIZE=4*MAX_NXNY/3+4*(MAX_NX+MAX_NY)+4*(MAX_LEVEL+1))
