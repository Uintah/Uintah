#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rshresults_fort.h>
      integer i,j,k

      double precision PI,SIG,aterm,bterm,lhs,rhs,error

      pi = dacos(-1.0D0)
      DATA SIG/5.67D-08/

c*********************************************************************
c     SK: I am commenting out the i.ne.idxhi(1) stuff
c     because I am not sure where it comes from.  It seems
c     as though this might prevent proper parallelization.
c     I will revisit this later.  For now, one problem
c     seems to be the calculation of the fluxes; they 
c     depend on cenint, which seems to demand ghosts, but
c     since cenint is a temporary variable, it has no ghosts.
c

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)

c               volq(i,j,k) = volq(i,j,k) + cenint(i,j,k)

               src(i,j,k) = src(i,j,k) +
     &              (abskg(i,j,k) * cenint(i,j,k) -
     &              4.0D0*PI * esrcg(i,j,k))

10          continue
20       continue
30    continue

      do 60 k=idxlo(3),idxhi(3)
         do 50 j=idxlo(2),idxhi(2)
            do 40 i=idxlo(1),idxhi(1)

               if (pcell(i+1,j,k).eq.ffield) then

c            if (i.ne.idxhi(1)) then
                  qfluxe(i,j,k) = qfluxe(i,j,k)+ 
     &                 (2.0d0*shgamma(i+1,j,k)*shgamma(i,j,k)/
     &                 (shgamma(i+1,j,k)+shgamma(i,j,k)))*
     &                 abs((cenint(i+1,j,k)-cenint(i,j,k))/(dxep(i)))
c            else
c      qfluxe(i,j,k) = 0.0d0
c            end if

               else

                  aterm = abskg(i+1,j,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i+1,j,k)))

                  qfluxe(i,j,k) = qfluxe(i,j,k) +
     &                 (aterm)* 
     &                 (cenint(i,j,k) - 4.0d0*sig*
     &                 abs(fraction(bands))*(tg(i+1,j,k)**4))

               end if

               if (pcell(i-1,j,k).eq.ffield) then

c            if (i.ne.idxlo(1)) then
                  qfluxw(i,j,k) = qfluxw(i,j,k)+
     &                 (2.0d0*shgamma(i-1,j,k)*shgamma(i,j,k)/
     &                 (shgamma(i-1,j,k)+shgamma(i,j,k)))*
     &                 abs((cenint(i,j,k)-cenint(i-1,j,k))/(dxpw(i)))
c            else
c      qfluxw(i,j,k) = 0.0d0
c            end if

               else

                  aterm = abskg(i-1,j,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i-1,j,k)))

                  qfluxw(i,j,k) = qfluxw(i,j,k) + 
     &                 (aterm)* 
     &                 (cenint(i,j,k) - 4.0d0*sig*
     &                 abs(fraction(bands))*(tg(i-1,j,k)**4))

               end if

               if (pcell(i,j+1,k).eq.ffield) then

c            if (j.ne.idxhi(2)) then
                  qfluxn(i,j,k) = qfluxn(i,j,k)+
     &                 (2.0d0*shgamma(i,j+1,k)*shgamma(i,j,k)/
     &                 (shgamma(i,j+1,k)+shgamma(i,j,k)))*
     &                 abs((cenint(i,j+1,k)-cenint(i,j,k))/(dynp(j)))
c            else
c      qfluxn(i,j,k) = 0.0d0
c            end if

               else

                  aterm = abskg(i,j+1,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j+1,k)))

                  qfluxn(i,j,k) = qfluxn(i,j,k) + 
     &                 (aterm)* 
     &                 (cenint(i,j,k) - 4.0d0*sig*
     &                 abs(fraction(bands))*(tg(i,j+1,k)**4))

               end if

               if (pcell(i,j-1,k).eq.ffield) then

c            if (j.ne.idxlo(2)) then
                  qfluxs(i,j,k) = qfluxs(i,j,k) + 
     &                 (2.0d0*shgamma(i,j-1,k)*shgamma(i,j,k)/
     &                 (shgamma(i,j-1,k)+shgamma(i,j,k)))*
     &                 abs((cenint(i,j,k)-cenint(i,j-1,k))/(dyps(j)))
c            else
c      qfluxs(i,j,k) = 0.0d0
c            end if

               else

                  aterm = abskg(i,j-1,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j-1,k)))

                  qfluxs(i,j,k) = qfluxs(i,j,k) + 
     &                 (aterm)* 
     &                 (cenint(i,j,k) - 4.0d0*sig*
     &                 abs(fraction(bands))*(tg(i,j-1,k)**4))
 
               end if

               if (pcell(i,j,k+1).eq.ffield) then

c            if (k.ne.idxhi(3)) then
                  qfluxt(i,j,k) = qfluxt(i,j,k)+
     &                 (2.0d0*shgamma(i,j,k+1)*shgamma(i,j,k)/
     &                 (shgamma(i,j,k+1)+shgamma(i,j,k)))*
     &                 abs((cenint(i,j,k+1)-cenint(i,j,k))/(dztp(k)))
c            else
c      qfluxt(i,j,k) = 0.0d0
c            end if

               else

                  aterm = abskg(i,j,k+1)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j,k+1)))

                  qfluxt(i,j,k) = qfluxt(i,j,k) + 
     &                 (aterm)* 
     &                 (cenint(i,j,k) - 4.0d0*sig*
     &                 abs(fraction(bands))*(tg(i,j,k+1)**4))

               end if

               if (pcell(i,j,k-1).eq.ffield) then

c            if (k.ne.idxlo(3)) then
                  qfluxb(i,j,k) = qfluxb(i,j,k)+
     &                 (2.0d0*shgamma(i,j,k-1)*shgamma(i,j,k)/
     &                 (shgamma(i,j,k-1)+shgamma(i,j,k)))*
     &                 abs((cenint(i,j,k)-cenint(i,j,k-1))/(dzpb(k)))
c            else
c      qfluxb(i,j,k) = 0.0d0
c            end if

               else

                  aterm = abskg(i,j,k-1)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j,k-1)))

                  qfluxb(i,j,k) = qfluxb(i,j,k) + 
     &                 (aterm)* 
     &                 (cenint(i,j,k) - 4.0d0*sig*
     &                 abs(fraction(bands))*(tg(i,j,k-1)**4))

               end if

c     Individual cell energy balance for debugging

c      if((i.eq.0).and.(j.eq.0).and.(k.eq.0))then

c       write(6,*)cenint(i-1,j,k),cenint(i+1,j,k)
c       write(6,*)cenint(i,j-1,k),cenint(i,j+1,k)
c       write(6,*)cenint(i,j,k-1),cenint(i,j,k+1)
c       write(6,*)cenint(i,j,k)

c      rhs = src(i,j,k)*(1.0d-3)
c      lhs = (qfluxe(i,j,k) - qfluxw(i,j,k) + 
c     &       qfluxn(i,j,k) - qfluxs(i,j,k) + 
c     &       qfluxt(i,j,k) - qfluxb(i,j,k))*
c     &      (1.0d-2)
c      error = (rhs - lhs)*100.0d0/rhs
c      write(6,*)"% rel.error in indiv.cell =", error 

c            end if
c     End of debugging

 40         continue
 50      continue
 60   continue

      RETURN
      end





