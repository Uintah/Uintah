      subroutine gf_copy( u, u_dlo, u_dhi, u_vlo, u_vhi,
     &                    v, v_dlo, v_dhi, v_vlo, v_vhi )	

c  $Id$ 

      implicit none

c  Purpose-  BLAS _copy operation (gridfunction data format).

c  Arguments-
c    Input:   v     - double precision array
c                     gridfunction
c             v_dlo - integer array of length 2
c                     coordinates of lower left corner of v.
c             v_dhi - integer array of length 2
c                     coordinates of upper right corner of v.
c             v_vlo - integer array of length 2
c                     coordinates of lower left corner of valid part of v.
c             v_vhi - integer array of length 2
c                     coordinates of upper right corner of valid part of v.
c             u_dlo - integer array of length 2
c                     coordinates of lower left corner of u.
c             u_dhi - integer array of length 2
c                     coordinates of upper right corner of u.
c             u_vlo - integer array of length 2
c                     coordinates of lower left corner of valid part of u.
c             u_vhi - integer array of length 2
c                     coordinates of upper right corner of valid part of u.

c    Output:  u     - double precision array
c                     contents of v copied to u

c  Dummy arguments-

      integer u_dlo(2), u_dhi(2), v_dlo(2), v_dhi(2)
      integer u_vlo(2), u_vhi(2), v_vlo(2), v_vhi(2)

      double precision u(u_dlo(1):u_dhi(1),u_dlo(2):u_dhi(2))
      double precision v(v_dlo(1):v_dhi(1),v_dlo(2):v_dhi(2))

c  Local variables-

      integer iu, ju
      integer iv, jv

c  Start of executable code-

      jv = v_vlo(2)
      do ju = u_vlo(2), u_vhi(2)
         iv = v_vlo(1)
         do iu = u_vlo(1), u_vhi(1)
            u(iu,ju) = v(iv,jv)
            iv = iv + 1
         end do
         jv = jv + 1
      end do

      return
      end

