      subroutine setCoarserOperator( 
     &                 apFine, aeFine, awFine, anFine, asFine, 
     &                 flo, fhi,
     &                 apCoarse, aeCoarse, awCoarse, anCoarse, asCoarse,
     &                 clo, chi )

c $Id$

      implicit none

c  Purpose- Define coarse grid operator through aggregation.

c  Arguments-
c     Input:  a[pewns]Fine   - stencil weights corresponding to fine grid
c                              operator 
c             flo            - index of lower left corner of fine grid
c             fhi            - index of upper right corner of fine grid
c             clo            - index of lower left corner of coarse grid
c             chi            - index of upper right corner of coarse grid

c    Output:  a[pewns]Coarse - stencil weights corresponding to coarse grid
c                              operator 

      integer chi(2)
      integer clo(2)
      integer fhi(2)
      integer flo(2)

      double precision apFine(flo(1):fhi(1),flo(2):fhi(2))
      double precision aeFine(flo(1):fhi(1),flo(2):fhi(2))
      double precision awFine(flo(1):fhi(1),flo(2):fhi(2))
      double precision anFine(flo(1):fhi(1),flo(2):fhi(2))
      double precision asFine(flo(1):fhi(1),flo(2):fhi(2))

      double precision apCoarse(clo(1):chi(1),clo(2):chi(2))
      double precision aeCoarse(clo(1):chi(1),clo(2):chi(2))
      double precision awCoarse(clo(1):chi(1),clo(2):chi(2))
      double precision anCoarse(clo(1):chi(1),clo(2):chi(2))
      double precision asCoarse(clo(1):chi(1),clo(2):chi(2))

c  Local variables-

      integer iCoarse
      integer iFine
      integer jCoarse
      integer jFine

      double precision fctr

c  Parameters-

      double precision half
      parameter      ( half=0.5d0 )

c  Start of executable code-

c  This is the "fudge factor" used by Koobus and Lallemand.

      fctr = half

c  Aggregate fine operator.  This corresponds to a Galerkin coarse grid
c  operator with a volume sum restriction operator and a piecewise
c  constant prolongation operator.  Careful around the boundaries!

      jFine = flo(2)
      do jCoarse = clo(2), chi(2)
         iFine = flo(1)
         do iCoarse = clo(1), chi(1)
            aeCoarse(iCoarse,jCoarse) = (aeFine(iFine+1,jFine  ) +
     &                                   aeFine(iFine+1,jFine+1))*fctr
            awCoarse(iCoarse,jCoarse) = (awFine(iFine,  jFine  ) +
     &                                   awFine(iFine,  jFine+1))*fctr
            anCoarse(iCoarse,jCoarse) = (anFine(iFine,  jFine+1) +
     &                                   anFine(iFine+1,jFine+1))*fctr
            asCoarse(iCoarse,jCoarse) = (asFine(iFine,  jFine  ) +
     &                                   asFine(iFine+1,jFine  ))*fctr
            apCoarse(iCoarse,jCoarse) = ( apFine(iFine,  jFine  ) +
     &          aeFine(iFine,  jFine)   + anFine(iFine,  jFine  ) +
     &                                    apFine(iFine+1,jFine  ) +
     &          awFine(iFine+1,jFine)   + anFine(iFine+1,jFine  ) +
     &                                    apFine(iFine  ,jFine+1) +
     &          aeFine(iFine  ,jFine+1) + asFine(iFine,  jFine+1) +
     &                                    apFine(iFine+1,jFine+1) +
     &          awFine(iFine+1,jFine+1) + asFine(iFine+1,jFine+1))*fctr
            iFine = iFine + 2
         end do
         jFine = jFine + 2
      end do

c  Fix the boundaries.

      jCoarse = clo(2)
      iFine = flo(1)
      jFine = flo(2)
      do iCoarse = clo(1), chi(1)
         asCoarse(iCoarse,jCoarse) = asFine(iFine,  jFine) + 
     &                               asFine(iFine+1,jFine)
         iFine = iFine + 2
      end do

      jCoarse = chi(2)
      iFine = flo(1)
      jFine = fhi(2)
      do iCoarse = clo(1), chi(1)
         anCoarse(iCoarse,jCoarse) = anFine(iFine,  jFine) + 
     &                               anFine(iFine+1,jFine)
         iFine = iFine + 2
      end do

      iCoarse = clo(1)
      iFine = flo(1)
      jFine = flo(2)
      do jCoarse = clo(2), chi(2)
         awCoarse(iCoarse,jCoarse) = awFine(iFine,jFine) +
     &                               awFine(iFine,jFine+1)
         jFine = jFine+2
      end do

      iCoarse = chi(1)
      iFine = fhi(1)
      jFine = flo(2)
      do jCoarse = clo(2), chi(2)
         aeCoarse(iCoarse,jCoarse) = aeFine(iFine,jFine) +
     &                               aeFine(iFine,jFine+1)
         jFine = jFine+2
      end do

      return
      end

