
c  $Id$

c  Operators for MG preconditioner.

      double precision as(MG_SIZE)
      double precision aw(MG_SIZE)
      double precision ap(MG_SIZE)
      double precision ae(MG_SIZE)
      double precision an(MG_SIZE)

c  All in common and saved, data persists across subroutine calls.

      common /operators/ as, aw, ap, ae, an

      save /operators/

