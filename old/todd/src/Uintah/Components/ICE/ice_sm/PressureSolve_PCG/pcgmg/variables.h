
c  $Id$

c  Variables for MG preconditioner.

      double precision u(MG_SIZE)
      double precision r(MG_SIZE)
      double precision f(MG_SIZE)

c  All in common and saved, data persists across subroutine calls.

      common /variables/ u, r, f

      save /variables/

