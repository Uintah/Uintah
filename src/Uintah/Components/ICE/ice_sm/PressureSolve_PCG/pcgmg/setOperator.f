      subroutine setOperator( ppap, ppae, ppaw, ppan, ppas,
     &                        ppop_lo, ppop_hi,
     &                        ap, ae, aw, an, as, op_lo, op_hi )

c $Id$

      implicit none

c  Purpose- Initialize Laplacian

      integer op_hi(2)
      integer op_lo(2)
      integer ppop_hi(2)
      integer ppop_lo(2)

      double precision ap(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision ae(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision aw(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision an(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision as(op_lo(1):op_hi(1),op_lo(2):op_hi(2))
      double precision ppap(ppop_lo(1):ppop_hi(1),ppop_lo(2):ppop_hi(2))
      double precision ppae(ppop_lo(1):ppop_hi(1),ppop_lo(2):ppop_hi(2))
      double precision ppaw(ppop_lo(1):ppop_hi(1),ppop_lo(2):ppop_hi(2))
      double precision ppan(ppop_lo(1):ppop_hi(1),ppop_lo(2):ppop_hi(2))
      double precision ppas(ppop_lo(1):ppop_hi(1),ppop_lo(2):ppop_hi(2))

      integer i
      integer j

      call gf_copy( ap, op_lo, op_hi, op_lo, op_hi,
     &              ppap, ppop_lo, ppop_hi, ppop_lo, ppop_hi )
      call gf_copy( ae, op_lo, op_hi, op_lo, op_hi,
     &              ppae, ppop_lo, ppop_hi, ppop_lo, ppop_hi )
      call gf_copy( aw, op_lo, op_hi, op_lo, op_hi,
     &              ppaw, ppop_lo, ppop_hi, ppop_lo, ppop_hi )
      call gf_copy( an, op_lo, op_hi, op_lo, op_hi,
     &              ppan, ppop_lo, ppop_hi, ppop_lo, ppop_hi )
      call gf_copy( as, op_lo, op_hi, op_lo, op_hi,
     &              ppas, ppop_lo, ppop_hi, ppop_lo, ppop_hi )

      return 
      end

