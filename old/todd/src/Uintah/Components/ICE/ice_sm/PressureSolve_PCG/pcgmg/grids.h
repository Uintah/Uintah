
c  $Id$ 

c  Grid descriptors for MG preconditioner.

c  Pointers to grid levels.

      integer levels

      integer g(0:MAX_LEVEL)

c  Arrays for grid function shapes on grid levels.

      integer mg_dlo(2,0:MAX_LEVEL), mg_dhi(2,0:MAX_LEVEL),
     &        mg_vlo(2,0:MAX_LEVEL), mg_vhi(2,0:MAX_LEVEL)


      common /grids/ levels, 
     &               g, mg_dlo, mg_dhi, mg_vlo, mg_vhi

      save /grids/

