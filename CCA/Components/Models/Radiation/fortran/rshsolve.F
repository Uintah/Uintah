#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/rshsolve_fort.h>

      integer i,j,k

      double precision PI,PI1,SIG
      double precision val, factor, aterm, bterm, cterm
      double precision areaew
      double precision arean
      double precision areatb

      DATA SIG/5.67D-08/

      pi = dacos(-1.0D0)
      PI1 = 1.0d0 / PI

      aterm = 0.0d0
      bterm = 0.0d0
      cterm = 0.0d0

c     Initial calculation of the matrix elements

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)
         
               plusX = .true.
               plusY = .true.
               plusZ = .true.

               volume(i,j,k) = sew(i) * sns(j) * stb(k)
               su(i,j,k) = 4.0d0*pi*esrcg(i,j,k)*volume(i,j,k)

               areaew = sns(j)*stb(k)
c               arean(i) = sew(i)*stb(k)   
               arean = sew(i)*stb(k)                 
c               areatb(i) = sew(i) * sns(j)
               areatb = sew(i) * sns(j)              

               if (pcell(i,j,k).ne.ffield) then

c  call fixval assume abskg to be one but change it in props calculation
                  val = -4*abs(fraction(bands))*SIG*tg(i,j,k)**4
                  factor = 1.d0
                  call fixradval(val,
     $                 aw(i,j,k),as(i,j,k),ab(i,j,k),
     $                 ae(i,j,k),an(i,j,k),at(i,j,k),
     $                 ap(i,j,k),su(i,j,k),factor)

               else

c     Calculate coefficients for all the interior nodes

                  if (pcell(i+1,j,k).eq.ffield) then

                     ae(i,j,k) = (2.0d0*shgamma(i+1,j,k)*
     &                    shgamma(i,j,k)/(shgamma(i+1,j,k)+
     &                    shgamma(i,j,k)))*(1.0d0/(xx(i+1)-xx(i)))*
     $                    areaew

                  else

                     ae(i,j,k) = 0.0d0

                  end if

                  if (pcell(i-1,j,k).eq.ffield) then

                     aw(i,j,k) = (2.0d0*shgamma(i-1,j,k)*
     &                    shgamma(i,j,k)/(shgamma(i-1,j,k)+
     &                    shgamma(i,j,k)))*(1.0d0/(xx(i)-xx(i-1)))*
     $                    areaew

                  else

                     aw(i,j,k) = 0.0d0
            
                  end if

                  if (pcell(i,j+1,k).eq.ffield) then

                     an(i,j,k) = (2.0d0*shgamma(i,j+1,k)*
     &                    shgamma(i,j,k)/(shgamma(i,j+1,k)+
c     &                    shgamma(i,j,k)))*(1.0d0/(yy(j+1)-yy(j)))*
c     $                    arean(i)
     &                    shgamma(i,j,k)))*(1.0d0/(yy(j+1)-yy(j)))*
     $                    arean

                  else

                     an(i,j,k) = 0.0d0
                     
                  end if

                  if (pcell(i,j-1,k).eq.ffield) then

                     as(i,j,k) = (2.0d0*shgamma(i,j-1,k)*
     &                    shgamma(i,j,k)/(shgamma(i,j-1,k)+
c     &                    shgamma(i,j,k)))*(1.0d0/(yy(j)-yy(j-1)))*
c     $                    arean(i)
     &                    shgamma(i,j,k)))*(1.0d0/(yy(j)-yy(j-1)))*
     $                    arean

                  else

                     as(i,j,k) = 0.0d0

                  end if

                  if (pcell(i,j,k+1).eq.ffield) then

                     at(i,j,k) = (2.0d0*shgamma(i,j,k+1)*
     &                    shgamma(i,j,k)/(shgamma(i,j,k+1)+
c     &                    shgamma(i,j,k)))*(1.0d0/(zz(k+1)-zz(k)))*
c     $                    areatb(i)
     &                    shgamma(i,j,k)))*(1.0d0/(zz(k+1)-zz(k)))*
     $                    areatb

                  else

                     at(i,j,k) = 0.0d0

                  end if

                  if (pcell(i,j,k-1).eq.ffield) then

                     ab(i,j,k) = (2.0d0*shgamma(i,j,k-1)*
     &                    shgamma(i,j,k)/(shgamma(i,j,k-1)+
c     &                    shgamma(i,j,k)))*(1.0d0/(zz(k)-zz(k-1)))*
c     $                    areatb(i)
     &                    shgamma(i,j,k)))*(1.0d0/(zz(k)-zz(k-1)))*
     $                    areatb

                  else

                     ab(i,j,k) = 0.0d0

                  end if

                  ap(i,j,k) = ae(i,j,k) + aw(i,j,k) +
     &                 an(i,j,k) + as(i,j,k) +
     &                 at(i,j,k) + ab(i,j,k) +
     &                  abskg(i,j,k)*volume(i,j,k)

                  su(i,j,k) = 4.0d0*pi*esrcg(i,j,k)*
     &                 volume(i,j,k)

               endif
               
 10         continue
 20      continue
 30   continue

c     Final calculation of the matrix elements

      do 60 k=idxlo(3),idxhi(3)
         do 50 j=idxlo(2),idxhi(2)
            do 40 i=idxlo(1),idxhi(1)

               areaew = sns(j)*stb(k)
               arean = sew(i)*stb(k)                 
               areatb = sew(i) * sns(j)

               if (pcell(i-1,j,k).ne.ffield) then

                  aterm = abskg(i-1,j,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i-1,j,k)))

                  ap(i,j,k) = ap(i,j,k) + (areaew*aterm) 

                  su(i,j,k) = su(i,j,k) + (4*sig*
     &                 abs(fraction(bands))*(tg(i-1,j,k)**4)*
     $                 areaew*aterm)

               endif      

               if (pcell(i+1,j,k).ne.ffield) then

                  aterm = abskg(i+1,j,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i+1,j,k)))

                  ap(i,j,k) = ap(i,j,k) + (areaew*aterm)

                  su(i,j,k) = su(i,j,k) + (4*sig*
     &                 abs(fraction(bands))*(tg(i+1,j,k)**4)*
     $                 areaew*aterm)

               endif

               if (pcell(i,j-1,k).ne.ffield) then

                  aterm = abskg(i,j-1,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j-1,k)))

c                  ap(i,j,k) = ap(i,j,k) + (arean(i)*aterm)
                  ap(i,j,k) = ap(i,j,k) + (arean*aterm)

                  su(i,j,k) = su(i,j,k) + (4*sig*
     &                 abs(fraction(bands))*(tg(i,j-1,k)**4)*
c     $                 arean(i)*aterm)
     $                 arean*aterm)

               endif

               if (pcell(i,j+1,k).ne.ffield) then

                  aterm = abskg(i,j+1,k)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j+1,k)))

c                  ap(i,j,k) = ap(i,j,k) + (arean(i)*aterm)
                  ap(i,j,k) = ap(i,j,k) + (arean*aterm)

                  su(i,j,k) = su(i,j,k) + (4*sig*
     &                 abs(fraction(bands))*(tg(i,j+1,k)**4)*
c     $                 arean(i)*aterm)
     $                 arean*aterm)

               endif

               if (pcell(i,j,k-1).ne.ffield) then

                  aterm = abskg(i,j,k-1)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j,k-1)))

c                  ap(i,j,k) = ap(i,j,k) + (areatb(i)*aterm)
                  ap(i,j,k) = ap(i,j,k) + (areatb*aterm)

                  su(i,j,k) = su(i,j,k) + (4*sig*
     &                 abs(fraction(bands))*(tg(i,j,k-1)**4)*
c     $                 areatb(i)*aterm)
     $                 areatb*aterm)

               endif

               if (pcell(i,j,k+1).ne.ffield) then

                  aterm = abskg(i,j,k+1)/(2.0d0*(2.0d0 - 
     &                 abskg(i,j,k+1)))
 
c                  ap(i,j,k) = ap(i,j,k) + (areatb(i)*aterm)
                  ap(i,j,k) = ap(i,j,k) + (areatb*aterm)

                  su(i,j,k) = su(i,j,k) + (4*sig*
     &                 abs(fraction(bands))*(tg(i,j,k+1)**4)*
c     $                 areatb(i)*aterm) 
     $                 areatb*aterm) 

               endif
               
c     Debug statements
c     if((i.eq.0).and.(j.eq.0).and.(k.eq.0))then

c     write(6,*)ae(i,j,k),aw(i,j,k)
c     write(6,*)an(i,j,k),as(i,j,k)
c     write(6,*)at(i,j,k),ab(i,j,k)
c     write(6,*)ap(i,j,k),su(i,j,k)
               
c     endif

 40         continue
 50      continue
 60   continue

      RETURN
      end










