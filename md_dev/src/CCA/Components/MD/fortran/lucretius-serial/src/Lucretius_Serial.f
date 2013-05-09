      PROGRAM md_dampingP
C     the same as md_damping.f but includes the stress tensor calculation for Q-mu recip
C
C     v2.14
C
C     polarizable md with mu-mu damping function, scaled mu-mu interactions
C     The exclude() subroutine has been modified to handle lone pairs
C     read25 has been modified to include redfactor and lredonefour
C     read12 has been modified to include lonepairtype array
C     if deform bond constant is zero, constrain the position of the atom
C     en4cen() and improper() have been moved to the middle timestep
C     v1.4a
C     v1.2 dummy atoms have been added, array of ldummy is created in read26
C     read12 has been changed
C     dummy type is defined using the charge type in ff.dat
C     if the atom is a dummy then bonds connected to it are not constrained
C     bonds to dummy atoms must have zero force constant
C     v2.0 damping of mu-mu interactions using A(Thole) a_thole has been added
C     v2.1 1.0d-4*(12/b)**12/r^12 has been added to exp-6 potential
C     v2.1g In order to remove slight center of mass drift in rather anisotropic
C           systems (~0.0001 m/s after ~1 ns) due to trancation of dipole-dipole
C           interactions and nonexact solution of induced dipole-dipole eq.s
C           center of mass velocity of the system is set to zero every timestep
C     v2.2 output of DipoleTot and chargeflux has been added
C     v2.3 SAVE statement has been added in results()
C     v2.4 stress tensor contribution due to dummy atoms has been added
C     v2.5 instead of chargeflux, DipolePol is used
C     v2.6 shstress(6) in fort.78, all times in fort.7* are real (not double)
C     v2.7 carbonyl-based lone pairs are treated separetely
C     v2.7b in cyclic molecules all atoms from listex are excluded from list14
C         c modified Ewald
C     v2.8 includes stress tensor for Ewald (md_dampingP.f)
C     v2.8a improved reciprocal part of Ewald
C     v2.9 different treatment of electrostatics. Lp-O-X-Lp interactions are excluded
C     v2.10 a check for correct assigment of bonds for unwrap() has been added
C     v2.11 Outputs DipoleChain in addition to DipoleTot and DipolePol 
C     calculates dipoles relative to the Q*Rcm
C     v2.11b a bug associated with px,py,pz on the last atom has been fixed.
C     v2.12 a stricter definition of a lone pair using bend atom types has been added
C     v2.13 a bug with a strict definition for Lp has been fixed
C     v2.14 no prediction of the induced dipoles based upon previous history
C     v2.15 if lstess=.false. stress tensor is incorrectly calculations
C
C     one needs to specify a bend C-O(Lp)-C to define a lone pair
C
C     polarizable version and extended force centers are written by Oleg Borodin
C     last modified 04/30/2008 by Oleg Borodin
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****localvariables
C
      integer iat,kk
      real*8 vcum(3)
C
C
C     *****initializations
C
      data kount /0/
      data kb /1.38066d-23/
      data avogadro /6.02205d+23/
      data gask /8.3144036/
      data gaskinkc /1.98709d-3/
      data cline(1:42) /'*------------------------------------------'/
      data cline(43:72) /'------------------------------------------'/
      data clineo(1:42) /'------------------------------------------'/
      data clineo(43:72) /'-------------------------------------------'/
      data clinec(1:42) /'#------------------------------------------'/
      data clinec(43:72) /'------------------------------------------'/
      pi = dacos(-1.0d0)
      rootpi = dsqrt(pi)
      delzero = 1.0d-12  ! system zero in the code
C
C     *****read data files
C
      CALL read25()
      CALL read12()
      CALL read11()
      CALL CheckBondNumber()
      CALL read26()
      CALL initialize()
      CALL intsetup() 
      CALL checker()
C
      if (printout)write(6,220)kount,box,pres,temp,toten
C
C     ***************************************************************
C
C                  *******begin the dynamics loop*******
C
C     ***************************************************************
C
800   continue
C
C     *****change volume if scaling to new value
C
      if (chvol) then
        boxold = box
        box = box + deltbox
        do iat = 1,nat
          do kk = 1,3
            x(kk,iat) = x(kk,iat)*(box/boxold)
            xn(kk,iat) = xn(kk,iat)*(box/boxold)
            xn_1(kk,iat) = xn_1(kk,iat)*(box/boxold)
          end do
        end do
      end if
C
C     *****make new nblist every nlist steps
C
      if (update) then
        CALL boxes()
        update = .false.
      endif
C
      CALL integrator()
C
C     *****eliminate center of mass velocity ADDED in md_nodamping.f by Oleg
C
      do kk=1,3 
       vcum(kk)=0.0d0
      end do
      do iat = 1,nat
        do kk = 1,3
          vcum(kk) = vcum(kk) + v(kk,iat)*mass(iat)
        end do
      end do
      do iat = 1,nat
        if (.not.ldummy(iat)) then
          do kk = 1,3
            v(kk,iat) = v(kk,iat) - vcum(kk)/(dble(natreal)*mass(iat))  ! Oleg
          end do
        endif
      end do
C
C     end of addition
C
      CALL output1()
C
      if(kount.le.nsteps) go to 800
C
      CALL output2()
C
      stop ! THE END 
C
C     *****formats
C
 220  format('Kount =',i7,' Box = ',f8.4,' P =',f11.2,' T = ',f9.3,
     +              ' E = ',f11.3)
C
      end
C
C
C     ***** Includes
C
      SUBROUTINE boxes()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer*2 subbox(maxbox,maxdim3)
      integer*2 boxofatom(3,maxat)
      integer isub(3)
      real*8 rbig,rdimmax,rdimmin,rdim,rcheck,size,xij1,xij2,xij3,r2
      integer icheck,ncheck,idim2,ibox,iz,iy,ix,iat,kk,ich
      integer nayo,isubz,izz,izbox,isuby,iyy,iybox,isubx,ixx,j,jat
      integer jch,iex,indexex,nayos
C
C     *****initialize some values
C
      el2 =box/2.0d0
C
C     *****determine box size
C
      rshort = ros + driftmax
      rsbox = rshort + driftmax
      rshort2 = rshort*rshort
      ros2 = ros*ros
      rsbox2 = rsbox*rsbox
      rbig = rcut
      rdimmax = rbig/2.0d0
      rdimmin = box/nat**(1.0d0/3.0d0)
      if (rdimmax .lt. rdimmin) then
        rdim = rdimmax
       else
        icheck = 2
5       icheck = icheck + 1
        rcheck = rbig/icheck
        if (rcheck .gt. rdimmin) goto 5
        rdim = rbig/(icheck-1)
       end if
      idim = box/rdim
      if (idim .gt. maxdim) then
        write(6,*)' idim > maxdim , reassigning idim to maxdim '
        idim = maxdim
       end if
      rdim = box/idim
      ncheck = rbig/rdim + 1
      if (idim .lt. 2*ncheck+1) then
        if (mod(idim,2) .eq. 0) then
          idim = idim + 1
          rdim = box/idim
        end if
        ncheck = (idim-1)/2
      end if
      idim2 = idim*idim
C      write (6,*) 'Boxes : idim = ',idim
C
C     *****begin calculations
C
      size = box/idim 
      ibox = 0
      do ibox = 1,idim*idim*idim
        listsubbox(ibox) = 0
      end do
C
      do iat = 1,nat
        do kk = 1,3
          isub(kk) = x(kk,iat)/size + 1
          boxofatom(kk,iat) = isub(kk)
        end do
        ibox = (isub(3) - 1)*idim2 + (isub(2) - 1)*idim + isub(1)
        listsubbox(ibox) = listsubbox(ibox) + 1
        subbox(listsubbox(ibox),ibox) = iat 
      end do
C
      do iat = 1, nat - 1
        ich = chain(iat)
        nayo = 0  
        nayos = 0  
C
C     *****determine possible neighbors from surrounding boxes
C
        ix = boxofatom(1,iat)
        iy = boxofatom(2,iat)
        iz = boxofatom(3,iat)
        do isubz = iz-ncheck,iz+ncheck
          izz = isubz
          if (izz .gt. idim) izz = izz - idim
          if (izz .lt. 1)    izz = izz + idim
          izbox = (izz - 1)*idim2
          do isuby = iy-ncheck,iy+ncheck
            iyy = isuby
            if (iyy .gt. idim) iyy = iyy - idim
            if (iyy .lt. 1)    iyy = iyy + idim
            iybox = (iyy - 1)*idim
            do isubx = ix-ncheck,ix+ncheck
              ixx = isubx
              if (ixx .gt. idim) ixx = ixx - idim
              if (ixx .lt. 1)    ixx = ixx + idim
              ibox = izbox + iybox + ixx
              do j = 1, listsubbox(ibox)
                jat = subbox(j,ibox)
                if (jat .le. iat) goto 200
                jch = chain(jat)
C
C     *****check for exclusion
C
                if (ich .eq. jch) then
                  do iex = 1,listmex(iat)
                    indexex = listex(iex,iat)
                    if (indexex .eq. jat) goto 200
                  end do
                 else
                  if (.not. inter) goto 200
                end if
C
C     *****calculate distance
C
                xij1 = x(1,jat) - x(1,iat)
                xij2 = x(2,jat) - x(2,iat)
                xij3 = x(3,jat) - x(3,iat)
                if (abs(xij1) .gt. el2)xij1 = xij1 - dsign(box,xij1)
                if (abs(xij2) .gt. el2)xij2 = xij2 - dsign(box,xij2)
                if (abs(xij3) .gt. el2)xij3 = xij3 - dsign(box,xij3)
C
                r2 = xij1*xij1 + xij2*xij2 + xij3*xij3
C
C     *****check cutoff
C
                if (r2 .gt. rcut2) goto 200 
C
C     *****record neighbor here
C
                if (r2 .lt. rsbox2)then
                  nayos = nayos + 1
                  lists(nayos,iat) = jat
                end if
                nayo = nayo + 1
                list(nayo,iat) = jat
200           continue
              end do
            end do
          end do
        end do
C
C     *****record the number of entries for this atom
C
        if (nayo.ge.maxnay) then
          write(6,*)" INCREASE maxnay from ",maxnay," to more than",nayo
          stop
        end if
C
        listm(iat) = nayo
        listms(iat) = nayos
      end do
      return
      end
C
      SUBROUTINE checker()
      implicit none
      include "params.h"
      include "dimensions.h"
C
C     *****local variables
C
      integer maxn,maxnayrec,maxnayex,maxboxch,maxboxrec,maxboxex
      integer maxdimrec,maxdimex,iat,ix,iy,iz,ibox
      include "commons.h"
C
      write(6,*)'**************WARNINGS*********************'
      maxn = 0
      do iat = 1, nat-1
        if (listm(iat) .gt. maxn) maxn = listm(iat)
      end do
      if (maxn .gt. maxnay)then
        write(6,*)'maxnay = ',maxnay,'  maxn = ',maxn
        stop
      end if
      maxnayrec = int(maxn*1.2)
      maxnayex = int(maxn*1.25)
C
      maxdimrec = idim + 3
      maxdimex = idim + 5
C
      maxboxch = 0
      ibox=0
      do iz = 1,idim
        do iy = 1,idim
          do ix = 1,idim
            ibox = ibox + 1
        if(listsubbox(ibox) .gt. maxboxch) maxboxch=listsubbox(ibox)
          end do
        end do
      end do
      maxboxrec = int(maxboxch*1.2)
      maxboxex = int(maxboxch*1.25)
C
      if (maxnay .gt. maxnayex)then
       write(6,*)'current maxnay = ',maxnay,', recommended value = ',
     +            maxnayrec
      end if
C
      if (maxdim .gt. maxdimex)then
        write(6,*)'current maxdim = ',maxdim,', recommended value = ',
     +           maxdimrec
      end if
C
      if (maxbox .gt. maxboxex)then
        write(6,*)'current maxbox = ',maxbox,', recommended value = ',
     +           maxboxrec
      end if
C
      write(6,*)'********** END OF WARNINGS*****************'
C
      return
      end
      SUBROUTINE cload(cc,deltaspline)
C
      implicit none
C
C     *****local variables
C
      real*8 cc(6,6),deltaspline
C
      cc(1,1) = 1.0d0
      cc(1,2) = 0.0d0
      cc(1,3) = 0.0d0
      cc(1,4) = 0.0d0
      cc(1,5) = 0.0d0
      cc(1,6) = 0.0d0
      cc(2,1) = 1.0d0
      cc(2,2) = deltaspline
      cc(2,3) = deltaspline**2
      cc(2,4) = deltaspline**3
      cc(2,5) = deltaspline**4
      cc(2,6) = deltaspline**5
      cc(3,1) = 0.0d0
      cc(3,2) = 1.0d0
      cc(3,3) = 0.0d0 
      cc(3,4) = 0.0d0
      cc(3,5) = 0.0d0
      cc(3,6) = 0.0d0
      cc(4,1) = 0.0d0
      cc(4,2) = 1.0d0
      cc(4,3) = 2.0d0*deltaspline    
      cc(4,4) = 3.0d0*deltaspline**2
      cc(4,5) = 4.0d0*deltaspline**3
      cc(4,6) = 5.0d0*deltaspline**4
      cc(5,1) = 0.0d0
      cc(5,2) = 0.0d0
      cc(5,3) = 2.0d0          
      cc(5,4) = 0.0d0
      cc(5,5) = 0.0d0
      cc(5,6) = 0.0d0
      cc(6,1) = 0.0d0
      cc(6,2) = 0.0d0
      cc(6,3) = 2.0d0
      cc(6,4) = 6.0d0*deltaspline
      cc(6,5) = 12.0d0*deltaspline**2
      cc(6,6) = 20.0d0*deltaspline**3
C
      return
      end
      SUBROUTINE correct()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,iat,itype,neightot,k,jat,jtype,itypee
      real*8 twoalphapi,qq,xij1,xij2,xij3,z,r,alphar,zinv,rinv,dalphar
      real*8 ffcorrect,f1,f2,f3,derfc,r3inv,unbdcorrect
      real*8 ci,cj ! added in v1.1
C
      twoalphapi = 2.0d0*alpha/rootpi
      el2 = box*0.5d0
      unbdcorrect= 0.0d0
C     virialcorrect = 0.0d0
C
      do iat = 1, nat - 1
        itype = atomtype(iat)
        ci=q(itype)
        neightot = listmex(iat)
        do k  = 1, neightot
          jat = listex(k,iat)
          jtype = atomtype(jat)
          cj=q(jtype)
          itypee = typee(itype,jtype)
          qq = electrostatic(itypee,1)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if( abs(xij1) .gt. el2)xij1 = xij1 - dsign(box,xij1)
          if( abs(xij2) .gt. el2)xij2 = xij2 - dsign(box,xij2)
          if( abs(xij3) .gt. el2)xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          zinv = 1.0d0/z
          rinv = 1.0d0/r
          r3inv = zinv*rinv
          dalphar = 1.0d0 - derfc(alphar)
C
C     *****determine energy
C
          unbdcorrect = unbdcorrect - qq*dalphar*rinv
C
C     ****determine electrostatic field
C
          ffcorrect = (dalphar*r3inv
     +                    - twoalphapi*exp(-alphar*alphar)*zinv)
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          elf(1,iat) = elf(1,iat) + f1*cj
          elf(2,iat) = elf(2,iat) + f2*cj
          elf(3,iat) = elf(3,iat) + f3*cj
          elf(1,jat) = elf(1,jat) - f1*ci
          elf(2,jat) = elf(2,jat) - f2*ci
          elf(3,jat) = elf(3,jat) - f3*ci
C
C     *****determine force and virial
C
          ffcorrect = ffcorrect*qq
C
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          fewald(1,iat) = fewald(1,iat) + f1
          fewald(2,iat) = fewald(2,iat) + f2
          fewald(3,iat) = fewald(3,iat) + f3
          fewald(1,jat) = fewald(1,jat) - f1
          fewald(2,jat) = fewald(2,jat) - f2
          fewald(3,jat) = fewald(3,jat) - f3
C
          tvirpo(1,1) = tvirpo(1,1) - f1*xij1
          tvirpo(1,2) = tvirpo(1,2) - f1*xij2
          tvirpo(1,3) = tvirpo(1,3) - f1*xij3
          tvirpo(2,1) = tvirpo(2,1) - f2*xij1
          tvirpo(2,2) = tvirpo(2,2) - f2*xij2
          tvirpo(2,3) = tvirpo(2,3) - f2*xij3
          tvirpo(3,1) = tvirpo(3,1) - f3*xij1
          tvirpo(3,2) = tvirpo(3,2) - f3*xij2
          tvirpo(3,3) = tvirpo(3,3) - f3*xij3
C
        end do
      end do
C
      if (lredonefour) then
       do iat = 1, nat 
        itype = atomtype(iat)
        ci=q(itype)
        neightot = listm14(iat)
        do k  = 1, neightot
          jat = list14(k,iat)
          jtype = atomtype(jat)
          cj=q(jtype)
          itypee = typee(itype,jtype)
          qq = electrostatic(itypee,1)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if( abs(xij1) .gt. el2)xij1 = xij1 - dsign(box,xij1)
          if( abs(xij2) .gt. el2)xij2 = xij2 - dsign(box,xij2)
          if( abs(xij3) .gt. el2)xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          rinv = 1.0d0/r
          zinv = rinv*rinv
          r3inv = zinv*rinv
          dalphar = 1.0d0 - derfc(alphar)
C
C     *****determine energy
C
          unbdcorrect = unbdcorrect -redfactor*qq*dalphar*rinv
C
C     ****determine electrostatic field
C
          ffcorrect = redfactor*(dalphar*r3inv
     +               - twoalphapi*exp(-alphar*alphar)*zinv)
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          elf(1,iat) = elf(1,iat) + f1*cj
          elf(2,iat) = elf(2,iat) + f2*cj
          elf(3,iat) = elf(3,iat) + f3*cj
          elf(1,jat) = elf(1,jat) - f1*ci
          elf(2,jat) = elf(2,jat) - f2*ci
          elf(3,jat) = elf(3,jat) - f3*ci
C
C     *****determine force and virial
C
          ffcorrect = ffcorrect*qq
C
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          fewald(1,iat) = fewald(1,iat) + f1
          fewald(2,iat) = fewald(2,iat) + f2
          fewald(3,iat) = fewald(3,iat) + f3
          fewald(1,jat) = fewald(1,jat) - f1
          fewald(2,jat) = fewald(2,jat) - f2
          fewald(3,jat) = fewald(3,jat) - f3
C
          tvirpo(1,1) = tvirpo(1,1) - f1*xij1
          tvirpo(1,2) = tvirpo(1,2) - f1*xij2
          tvirpo(1,3) = tvirpo(1,3) - f1*xij3
          tvirpo(2,1) = tvirpo(2,1) - f2*xij1
          tvirpo(2,2) = tvirpo(2,2) - f2*xij2
          tvirpo(2,3) = tvirpo(2,3) - f2*xij3
          tvirpo(3,1) = tvirpo(3,1) - f3*xij1
          tvirpo(3,2) = tvirpo(3,2) - f3*xij2
          tvirpo(3,3) = tvirpo(3,3) - f3*xij3
C
        end do
       end do
      endif    ! (lredonefour)
C
      if (lredQ_mu14) then
       do iat = 1, nat-1
        itype = atomtype(iat)
        ci=q(itype)
        neightot = listm14(iat)
        do k  = 1, neightot
          jat = list14(k,iat)
          jtype = atomtype(jat)
          cj=q(jtype)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if( abs(xij1) .gt. el2)xij1 = xij1 - dsign(box,xij1)
          if( abs(xij2) .gt. el2)xij2 = xij2 - dsign(box,xij2)
          if( abs(xij3) .gt. el2)xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          zinv = 1.0d0/z
          rinv = 1.0d0/r
          r3inv = zinv*rinv
          dalphar = 1.0d0 - derfc(alphar)
C
          ffcorrect = redQmufactor*(dalphar*r3inv
     $               - twoalphapi*exp(-alphar*alphar)*zinv)
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          elf(1,iat) = elf(1,iat) + f1*cj
          elf(2,iat) = elf(2,iat) + f2*cj
          elf(3,iat) = elf(3,iat) + f3*cj
          elf(1,jat) = elf(1,jat) - f1*ci
          elf(2,jat) = elf(2,jat) - f2*ci
          elf(3,jat) = elf(3,jat) - f3*ci
C
        end do
       end do
      endif    ! (lredQ_mu14)
C
      vir = vir + unbdcorrect
      unbd = unbd + unbdcorrect
C
      do iat = 1,nat
        do kk = 1,3
          f(kk,iat) = f(kk,iat) + fewald(kk,iat)
        end do
      end do
C
      return
      end
C
      double precision FUNCTION det(a,b,c)
C
      implicit none
C
      real*8 a(3),b(3),c(3)
C
      det = a(1)*(b(2)*c(3)-b(3)*c(2)) -
     +         a(2)*(b(1)*c(3)-b(3)*c(1)) +
     +         a(3)*(b(1)*c(2)-b(2)*c(1))
      return
      end
C
      SUBROUTINE diffusion()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      real*8 delxup(3,maxat)
      integer iat,ich,kk
      real*8 disp,delxmax
      save delxup 
C
C
C     *****initializations
C
      delxmax = 0.0d0
C
C     *****calculate maximum movement, chain center of mass movement
C
      do iat = 1,nat
        disp = 0.0d0
        ich = chain(iat)
        do kk = 1,3
          sumdel(kk,iat) = sumdel(kk,iat) + delx(kk,iat)
          delxup(kk,iat) = delxup(kk,iat) + delx(kk,iat)
          disp = disp + delxup(kk,iat)*delxup(kk,iat)
        end do
        if (disp .gt. delxmax) delxmax = disp
      end do
      delxmax = sqrt(delxmax)
      if (delxmax .gt. driftmax) then
         update = .true.
         do iat = 1,nat
           do kk = 1,3
             delxup(kk,iat) = 0.0d0
           end do
         end do
       end if
C
C     *****calculate atom movement
C
      spol = 0.00d0
      do iat = 1,nat
        do kk = 1,3
          spol = spol + sumdel(kk,iat)*sumdel(kk,iat)
        end do
      end do
C
      return
      end
      double precision function dot(a,b)
C
      implicit none
C
C     *****local variables
C
      real*8 a(3),b(3)
C
      dot = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
      return
      end
C
      SUBROUTINE en2cen()
C
      implicit none
C
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,ibond,iat,jat,itype
      real*8 bondk,rij0,rbij2,dot,rbij,enbond,f1,fb
      real*8 dij(3)
C
C     *****initialization of energy
C
      ebond = 0.0d0
C
      el2 = box/2.0d0
      do ibond = 1, nbonds
C       if (.not.ldummybond(ibond)) then
        iat = bonds(1,ibond)
        jat = bonds(2,ibond)
        itype = bonds(3,ibond)
        bondk = stretch(1,itype)
        rij0 = stretch(2,itype)
C
C     *****compute bond lengths
C
        do kk = 1,3
          dij(kk) = x(kk,jat) - x(kk,iat) 
          if (abs(dij(kk)) .gt. el2) then
            dij(kk) = dij(kk) - dsign(box,dij(kk))
          endif
        end do
        rbij2 = dot(dij,dij)
        rbij = dsqrt(rbij2)
        do kk = 1,3
          stretches(kk,ibond) = dij(kk)
          stretches(kk,-ibond) = -dij(kk)
        end do
        stretches(4,ibond) = rbij2
        stretches(4,-ibond) = rbij2
        if (constrain) goto 50
C
C     *****compute the energy,virial,forces for the bond
C
        enbond = 0.5 * bondk * (rbij - rij0)**2  
        ebond = ebond + enbond 
C
        f1 =  bondk*(rbij - rij0)/rbij
        vir = vir - (f1*rbij*rbij)
        do kk=1,3
          fb = f1 * dij(kk)
          f(kk,iat) =  f(kk,iat) + fb
          f(kk,jat) =  f(kk,jat) - fb
          do jj = 1,3
            tvirpo(kk,jj) = tvirpo(kk,jj) - dij(jj)*fb
          end do
        end do
50      continue
C       endif
      end do
C
      return
      end
      SUBROUTINE en3cen()
C
      implicit none
C
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,ibend,iat,jat,kat,itype,ib1,ib2
      real*8 bendk,aijk0,diff,c11,c11inv,c12,dot,c22,c12inv,c22inv,cst
      real*8 theta,bondangle,snt,tnt,enbend,f1,q1,q2,q3
      real*8 raij(3),rajk(3)
      real*8 pos1(3),pos2(3),pos3(3)
C
C     *****initialization of force
C
      ebend = 0.0d0
      el2 = box/2.0d0
C
C     valence bending ......
C
C       i --- j --- k
C         ibb1   ibb2
C
      
      do ibend = 1,nbends
        iat    = bends(1,ibend)
        jat    = bends(2,ibend)
        kat    = bends(3,ibend)
        itype  = bends(4,ibend)
        ib1    = bends(5,ibend)
        ib2    = bends(6,ibend)
        bendk  = bend(1,itype)
        aijk0  = bend(2,itype) * pi/180.d0
C
C     *****for stress tensor
C
        do kk = 1,3
          pos1(kk) = x(kk,iat)
          diff = x(kk,jat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos2(kk) = pos1(kk) + diff
          diff = x(kk,kat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos3(kk) = pos1(kk) + diff
        end do
C
C     *****computation of the bond angle theta
C
        do kk  =  1,3
          raij(kk) =  stretches(kk,ib1)
          rajk(kk) =  stretches(kk,ib2)
        end do
C       c11 = dot(raij,raij)
        c11 = stretches(4,ib1)
        c11inv = 1.0d0/c11
        c12 = dot(raij,rajk)
C       c22 = dot(rajk,rajk)
        c22 = stretches(4,ib2)
C
        if (abs(c12) .lt. delzero)c12 = delzero
        c12inv = 1.0d0/c12
        c22inv = 1.0d0/c22
        cst = c12*sqrt(c11inv*c22inv) 
        if (cst .gt. 1.0d0)cst = 1.0d0
        if (cst .lt. -1.0d0)cst = -1.0d0
        theta = dacos(cst)
        bondangle = pi - theta
        snt   = dsin(theta)
        tnt   = dtan(theta)
C
C     *****energy contribution from theta
C
        enbend  = 0.5d0*bendk*(bondangle - aijk0)**2
        ebend    = ebend + enbend
        f1 = bendk*(bondangle - aijk0)/tnt
C
C     *****compute negative gradient ( =force)
C
        do kk = 1,3
          q1 = f1*((raij(kk)*c11inv) - (rajk(kk)*c12inv))
          q3 = f1*((raij(kk)*c12inv) - (rajk(kk)*c22inv))
          q2 = 0.0d0 - q1 - q3
          f(kk,iat) = -q1 + f(kk,iat)
          f(kk,jat) = -q2 + f(kk,jat)
          f(kk,kat) = -q3 + f(kk,kat)
          do jj = 1,3
            tvirpo(kk,jj) = tvirpo(kk,jj) - q1*pos1(jj) -
     +                             q2*pos2(jj) - q3*pos3(jj)
          end do
        end do
      end do
      return
      end
      SUBROUTINE en4cen()
C
      implicit none
C
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,itort,iat,jat,kat,lat,itype
      integer ibt1,ibt2,ibt3,nfolds,ifold
      real*8 tk0,diff,c11,dot,c12,c13,c22,c23,c33,t1,t2,t3,t4,t5,t6
      real*8 rat1,rat2,bee,cst,phi,sininv,rati,ratl,f1,entort
      real*8 rtij(3),rtjk(3),rtkl(3)
      real*8 fi(3),fj(3),fk(3),fl(3)
      real*8 tortk(10)
      real*8 pos1(3),pos2(3),pos3(3),pos4(3)
C
C     *****initialization of energy
C
      el2 = box/2.0d0
      etort = 0.0d0
C
C         ii --- jj --- kk --- ll
C
C            ibt1    ibt2    ibt3
C
C
      do itort = 1,ntorts
        iat = torts(1,itort)
        jat = torts(2,itort)
        kat = torts(3,itort)
        lat = torts(4,itort)
        itype=torts(5,itort)
        ibt1 = torts(6,itort)
        ibt2 = torts(7,itort)
        ibt3 = torts(8,itort)
        nfolds = nprms(itype) - 1
        tK0 = twist(0,itype)
        do ifold = 1,nfolds
          tortk(ifold) = twist(ifold,itype)
        end do
C
C     *****for stress tensor
C
        do kk = 1,3
          pos1(kk) = x(kk,iat)
          diff = x(kk,jat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos2(kk) = pos1(kk) + diff
          diff = x(kk,kat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos3(kk) = pos1(kk) + diff
          diff = x(kk,lat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos4(kk) = pos1(kk) + diff
        end do
C
C     *****torsional angle (phi) calculation
C
       do kk = 1,3
        rtij(kk) = stretches(kk,ibt1)
        rtjk(kk) = stretches(kk,ibt2)
        rtkl(kk) = stretches(kk,ibt3)
       end do
C       c11 = dot(rtij,rtij)
        c11 = stretches(4,ibt1)
        c12 = dot(rtij,rtjk)
        c13 = dot(rtij,rtkl)
C       c22 = dot(rtjk,rtjk)
        c22 = stretches(4,ibt2)
        c23 = dot(rtjk,rtkl)
C       c33 = dot(rtkl,rtkl)
        c33 = stretches(4,ibt3)
        t1 = c13*c22 - c12*c23
        t2 = c11*c23 - c12*c13
        t3 = c12*c12 - c11*c22
        t4 = c22*c33 - c23*c23
        t5 = c13*c23 - c12*c33
        t6 = -t1
C
        rat1 = c12/c22
        rat2 = c23/c22
        bee = dsqrt(-t3*t4)
        cst = t6/bee
        if (cst .lt. -1.0d0)cst = -1.0d0
        if (cst .gt. +1.0d0)cst = +1.0d0
        phi = dacos(cst)
        if (abs(phi) .lt. delzero)phi = delzero
        sininv = 1.0d0/dsin(phi)
        rati = cst*c22/(t1*t3)
        ratl = cst*c22/(t1*t4)
C       detval = det(rtij,rtjk,rtkl)
C       if (detval .lt. 0.0d0)phi = 2*pi - phi
C
C     *****energy
C
        f1 = 0.0d0
        entort = 0.0d0
        do ifold = 1,nfolds
          entort = entort - 0.5d0*tortk(ifold)*dcos(ifold*phi)
          f1 = f1 + 0.5d0*ifold*tortk(ifold)*dsin(ifold*phi)*sininv
        end do
        etort = etort + entort + 0.5*tK0
C
C     *****compute negative gradient
C
        do kk =1,3
          fi(kk) = -rati*(t1*(rtij(kk))+t2*(rtjk(kk))+t3*(rtkl(kk)))
          fl(kk) =  ratl*(t4*(rtij(kk))+t5*(rtjk(kk))+t6*(rtkl(kk)))
          fj(kk) = -(1+rat1)*fi(kk) + rat2*fl(kk)
          fk(kk) =  rat1*fi(kk) - (1+rat2)*fl(kk)
C
          f(kk,iat) = f1*fi(kk) + f(kk,iat)
          f(kk,jat) = f1*fj(kk) + f(kk,jat)
          f(kk,kat) = f1*fk(kk) + f(kk,kat)
          f(kk,lat) = f1*fl(kk) + f(kk,lat)
          do jj = 1,3
            tvirpo(kk,jj) = tvirpo(kk,jj) + f1*fi(kk)*pos1(jj) +
     +                   f1*fj(kk)*pos2(jj) + f1*fk(kk)*pos3(jj) +
     +                      f1*fl(kk)*pos4(jj)
          end do
        end do
      end do
      return 
      end
C
      double precision FUNCTION funct(z,innb,ifunct)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      integer ifunct,innb
      real*8 z
C
C     *****local variables
C
      integer imap
      real*8 derfc,tolz,r,fupper,flower,pforce,pintforce,a,b,c
      real*8 eforce,edforce,eintforce,aa
C
      funct = 0.0d0
      tolz = 1.0d-8
      r = dsqrt(z)
C
      if (ifunct .eq. 2)then
        if (.not. tbpol)return
        if (abs(rop - r) .lt. tolz)then
          funct = 0.0d0
          return 
        end if
        fupper = pforce(r)
        flower = pforce(rop)
      funct=pintforce(rop,flower,r,fupper)*electrostatic(innb,2)
      return
      end if
C
      if (ifunct .eq. 3)then
        imap = map(innb)
        a = nonbonded(1,imap)
        b = nonbonded(2,imap)
        c = nonbonded(3,imap)
        if (b .lt. 1.0d-6)then
          funct = a/z**6 - c/z**3
         else
          aa = 0.5d-4*(12.0d0/b)**12
          funct = a*exp(-b*sqrt(z)) - c/z**3 + aa/z**6
        end if
        return
      end if
C
C
      alpha = alphai/box
      if (ifunct .eq. 1)then
        funct = derfc(alpha*r)*electrostatic(innb,1)/r
        return
      end if
C
      end
      double precision FUNCTION functd(z,innb,ifunct)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      integer ifunct,innb
      real*8 z
C
C     *****local variables
C
      integer imap
      real*8 tolz,r,pforce,a,b,c,derfc
      real*8 eforce,edforce,eintforce,aa
C
      functd = 0.0d0
      tolz = 1.0d-8
      r = dsqrt(z)
C
      if (ifunct .eq. 2)then
        if (.not. tbpol)return
        functd = -pforce(r)*electrostatic(innb,2)/(2.0d0*r)
        return
      end if
C
      if (ifunct .eq. 3)then
        imap = map(innb)
        a = nonbonded(1,imap)
        b = nonbonded(2,imap)
        c = nonbonded(3,imap)
        if (b .lt. 1.0d-6)then
          functd = -6.0d0*a/z**7 + 3.0d0*c/z**4
         else
          aa = 0.5d-4*(12.0d0/b)**12
          functd = -a*b*exp(-b*sqrt(z))/(2.0d0*sqrt(z))     
     +          + 3.0d0*c/z**4 - 6.0d0*aa/z**7
        end if
        return
      end if
C
C
      if (ifunct .eq. 1)then
        functd = - electrostatic(innb,1)*((derfc(alpha*r) ! Ewald
     +    /(2.0d0*r**3)) + (alpha*dexp(-(alpha*r)**2)/(r*r*rootpi)))
        return
      end if
C
      end
      double precision FUNCTION functdd(z,innb,ifunct)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      integer ifunct,innb
      real*8 z
C
C     *****local variables
C
      integer imap
      real*8 tolz,r,endder,pdforce,pforce,a,b,c,eff,edf,derfc
      real*8 eforce,edforce,eintforce,aa
C
      functdd = 0.0d0
      tolz = 1.0d-8
      r = dsqrt(z)
C
      if (ifunct .eq. 2)then
        if (.not. tbpol)return
        endder = (1.0d0/(4.0d0*r))*(pforce(r)/(r*r)
     +                  - pdforce(r)/r)
        functdd = electrostatic(innb,2)*endder
        return
      end if
C
      if (ifunct .eq. 3)then
        imap = map(innb)
        a = nonbonded(1,imap)
        b = nonbonded(2,imap)
        c = nonbonded(3,imap)
        if (b .lt. 1.0d-6)then
          functdd = 42.0d0*a/z**8 -12.0d0*c/z**5
         else
          aa = 0.5d-4*(12.0d0/b)**12
          functdd = 0.25d0*a*b*exp(-b*sqrt(z))
     +              *(z**(-1.5d0) + b*z**(-1))
     +              -12.0d0*c/z**5 + 42.0d0*aa/z**8
        end if
      return
      end if
C
C
      if (ifunct .eq. 1)then
        eff = derfc(alpha*r)/(r*r)
     +              + (2.0d0*alpha/rootpi)*dexp(-(alpha*r)**2)/r
        edf = -2.0d0*derfc(alpha*r)/r**3
     +           -(4.0d0*alpha/rootpi)*exp(-(alpha*r)**2)*
     +              (1/r**2 + alpha**2)
        endder = (1.0d0/(4.0d0*r))*(eff/(r*r) - edf/r)
        functdd = electrostatic(innb,1)*endder
        return
      end if
C
      end
      SUBROUTINE gauss(matrix,defsiz,siz,pivot)
C
C     This subroutine performs gaussian elimination with scaled partial  
C     pivoting (rows).  the multipliers are stored in the eliminated entries.
C     row pivoting information is stored in a vector. 
C
      implicit none
C
C     *****shared variables
C
      integer siz,defsiz,pivot(defsiz)
      real*8 matrix(defsiz,defsiz)
C
C     *****local variables
C
      integer row,ind1,ind2,ix,brow,bstore
      real*8 big,bigent,mult,zzero,store
C
C     *****assign constant values
C
      zzero=1.0d-07
C
C     *****intitialize pivot vector
C
      do row=1,siz
        pivot(row)=row
      end do
C
C     *****begin interations
C
      do row=1, siz-1
        big=0.0
        brow=0
        do ind1=row,siz
          bigent=0.0d0
          do ind2=row+1,siz 
            if (abs(matrix(ind1,ind2)) .gt. bigent) then
              bigent=abs(matrix(ind1,ind2))
            end if
          end do
C
C     *****check for singular matrix
C
          if ((abs(matrix(ind1,row)) .le. zzero) .and.
     +	    ( bigent .le. zzero)) then
            write (6,*) 'matrix is singular'
          end if 
C
          if (bigent .lt. zzero)  bigent=zzero
          if (abs(matrix(ind1,row)/bigent) .gt. big) then
            big=abs(matrix(ind1,row)/bigent)
            brow=ind1
          end if
        end do
C
C     *****perform pivot, update pivot vector
C
        if (brow .ne. row) then
          do ind1=1,siz
            store=matrix(row,ind1)
            matrix(row,ind1)=matrix(brow,ind1)
            matrix(brow,ind1)=store
          end do
          bstore=pivot(row)
          pivot(row)=pivot(brow)
          pivot(brow)=bstore
        end if
C
        do ind1=row+1,siz
          mult=matrix(ind1,row)/matrix(row,row)
          do ind2=row+1,siz
            matrix(ind1,ind2)=matrix(ind1,ind2)-mult*matrix(row,ind2)
          end do
          matrix(ind1,row)=mult
        end do
      end do
C
      return
      end
      SUBROUTINE getkin()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer iat,kk,jj
      real*8 ratio,volinv,dndof,presold,pvir,mv2tot
      real*8 stresske(3,3),stresspo(3,3)
      real*8 stresstr(3,3),stressold(3,3)
C
      mv2tot = 0.0d0
      do iat = 1,nat
        mv2tot = mv2tot + (v(1,iat)*v(1,iat)+v(2,iat)*v(2,iat)+
     +      v(3,iat)*v(3,iat)) * mass(iat)
      end do
C
      kinen = 0.5d0 * mv2tot/(41.84d5)
C
      temp   = mv2tot/(dble(ndof) * gask * 1000.0d0)
C
C     *****for fixed temperature
C
      if (fixtemp) then 
        ratio = sqrt(tstart/temp)
        do iat = 1,nat
          v(1,iat) = v(1,iat)*ratio
          v(2,iat) = v(2,iat)*ratio
          v(3,iat) = v(3,iat)*ratio
        end do
        temp = tstart
      end if
C
C     *****Pressure and stress tensor calculations
C
      if (newpress) then
        volinv =1.0d0/(box*box*box)
        dndof = dble(ndof-3)/dble(ndof)
C       etrunc = esumninj*volinv
C
C     *****pressure terms
C
        ptrunc = psumninj*volinv*volinv
        pke = 45.42d0*dble(ndof-3)*temp*volinv
        pvir  = 2.2857d+04*vir*volinv
        pintr =  pvir + ptrunc 
        pres       =  pke + pintr
C
C     *****stress terms
C
        do kk = 1,3
          do jj = 1,3
            prtkine(kk,jj) = 0.0d0
            stress(kk,jj) = 0.0d0
          end do
        end do
        do iat = 1,nat
          do kk = 1,3
            do jj = 1,3
              prtkine(kk,jj) = prtkine(kk,jj) +
     +                            mass(iat)*v(kk,iat)*v(jj,iat)
            end do
          end do
        end do
C
        do kk = 1,3
          do jj = 1,3
            stresstr(kk,jj) = 0.0d0
            if (kk  .eq. jj) stresstr(kk,jj) = ptrunc
          end do
        end do
C
        do kk = 1,3
          do jj = 1,3
            stresske(kk,jj) = prtkine(kk,jj)*dndof*16.3884d-3*volinv
            stresspo(kk,jj) = tvirpo(kk,jj)*6.8571d4*volinv
            stress(kk,jj) = stresske(kk,jj) + stresspo(kk,jj) +
     +                       stresstr(kk,jj)
          end do
        end do
C
        presold = pres
        do kk = 1,3
          do jj = 1,3
            stressold(kk,jj) = stress(kk,jj)
          end do
        end do
C
       else 
        pres = presold
        do kk = 1,3
          do jj = 1,3
            stress(kk,jj) = stressold(kk,jj)
          end do
        end do
      end if
C
      return
      end 
      SUBROUTINE gsolve(matrix,defsiz,siz,b,xt,pivot)
C
C     This subroutine takes a gauss eliminated matrix with 
C       multipliers stored in eliminated entries, a b vector, and 
C       a pivot vector containing row pivot information associated 
C       with the matrix and performs backward substitution to yield the 
C       solution vector
C
      implicit none
C
C     *****shared variables
C
      integer defsiz,siz,pivot(defsiz)
      real*8 b(defsiz),matrix(defsiz,defsiz),xt(defsiz)
C
C     *****local variables
C
      integer ind1,ind2
      real*8 mult,sum
      real*8 btemp(20)
C
C     *****pivot b vector
C
      do ind1=1,siz
        btemp(ind1)=b(ind1)
      end do
C
      do ind1=1,siz
        b(ind1)=btemp(pivot(ind1))
      end do
C
C     *****perform elimination on b vector
C
      do ind1=1,siz-1
        do ind2=ind1+1,siz 
          mult=matrix(ind2,ind1)
          b(ind2)=b(ind2)-b(ind1)*mult
        end do
      end do
C
      xt(siz)=b(siz)/matrix(siz,siz)
      do ind1=siz-1,1,-1
        sum=0.0
        do ind2=ind1+1,siz
          sum=sum+matrix(ind1,ind2)*xt(ind2)
        end do
        xt(ind1)=(b(ind1)-sum)/matrix(ind1,ind1)
      end do
C
      return
      end
      SUBROUTINE hydriva(xref,dhdxa)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      real*8 dhdxa(3,3,3,*),xref(3,maxat)
C
C     *****local variables
C
      integer itype,im,icenter,icenter_m1,icenter_p1,kk,ic,icx
      real*8 avect(3),bvect(3),dpdx(3),dvdx(3),bxa(3),aminusb(3)
      real*8 vect1(3),vect2(3),vect3(3),vect4(3),vect5(3)
      real*8 p(3,3,3),vectderiv(3,2,3,3),ivdirect(3)
      real*8 dp,dv,dpi,dvi,dpi3,dvi3,totalp,totalv,fix,temph
C
C
C     *****vectderiv
C
      data itype/2/
      data vectderiv /-1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     +                 1.0, 0.0, 0.0,-1.0, 0.0, 0.0,
     +                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
     +                 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,
     +                 0.0, 1.0, 0.0, 0.0,-1.0, 0.0,
     +                 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
     +                 0.0, 0.0,-1.0, 0.0, 0.0, 0.0,
     +                 0.0, 0.0, 1.0, 0.0, 0.0,-1.0,
     +                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/
      data ivdirect / -1, 1, -1/
C
C     *****calculate vectors
C
      do im = 1,numaromatic
        ivdirect(2) = idpar(im)
        icenter = iaromatic(2,im)
        icenter_m1 = iaromatic(1,im)
        icenter_p1 = iaromatic(3,im)
        do kk = 1,3
          avect(kk) = xref(kk,icenter)    - xref(kk,icenter_m1)
          bvect(kk) = xref(kk,icenter_p1) - xref(kk,icenter)
        end do
C
C     *****calculate bxa,aminusb,dp and dv
C
        bxa(1)=bvect(2)*avect(3)-avect(2)*bvect(3)
        bxa(2)=-(bvect(1)*avect(3)-avect(1)*bvect(3))
        bxa(3)=bvect(1)*avect(2)-avect(1)*bvect(2)
        dp = 0.0d0
        dv = 0.0d0
        do kk = 1,3
          aminusb(kk) = avect(kk) - bvect(kk)
          dp = dp + bxa(kk)*bxa(kk)
          dv = dv + aminusb(kk)*aminusb(kk)
        end do
C
        dp = sqrt(dp)
        dpi = 1.0d0/dp
        dv = sqrt(dv)
        dvi = 1.0d0/dv
        dpi3 = dpi**3
        dvi3 = dvi**3
C
C     *****calculate p and v vectors for each derivative
C
        do ic = 1,3   ! each carbon
          do icx = 1,3 ! each coordinate
            vect1(1) = bvect(2)*vectderiv(3,1,ic,icx) -
     +                   bvect(3)*vectderiv(2,1,ic,icx)
            vect1(2) = -(bvect(1)*vectderiv(3,1,ic,icx) -
     +                     bvect(3)*vectderiv(1,1,ic,icx))
            vect1(3) = bvect(1)*vectderiv(2,1,ic,icx) -
     +                   bvect(2)*vectderiv(1,1,ic,icx)
            vect2(1) = vectderiv(2,2,ic,icx)*avect(3) -
     +                  vectderiv(3,2,ic,icx)*avect(2)
            vect2(2) = -(vectderiv(1,2,ic,icx)*avect(3) -
     +                     vectderiv(3,2,ic,icx)*avect(1))
            vect2(3) = vectderiv(1,2,ic,icx)*avect(2) -
     +                   vectderiv(2,2,ic,icx)*avect(1)
            totalp = 0.d0
            totalv = 0.d0
            vect3(1) = vect2(1) + vect1(1)
            vect5(1) = vectderiv(1,1,ic,icx) -
     +                     vectderiv(1,2,ic,icx)
            totalp = totalp + bxa(1)*vect3(1)
            totalv = totalv + aminusb(1)*vect5(1)
            vect3(2) = vect2(2) + vect1(2)
            vect5(2) = vectderiv(2,1,ic,icx) -
     +                     vectderiv(2,2,ic,icx)
            totalp = totalp + bxa(2)*vect3(2)
            totalv = totalv + aminusb(2)*vect5(2)
            vect3(3) = vect2(3) + vect1(3)
            vect5(3) = vectderiv(3,1,ic,icx) -
     +                     vectderiv(3,2,ic,icx)
            totalp = totalp + bxa(3)*vect3(3)
            totalv = totalv + aminusb(3)*vect5(3)
            fix = 0.d0
            if (itype .eq. ic .and.   1 .eq. icx) fix = 1.d0
            dpdx(  1) = - bxa(  1)*dpi3*totalp + vect3(  1)*dpi
            dvdx(  1) = - aminusb(  1)*dvi3*totalv + vect5(  1)*dvi
            temph = fix + ivdirect(itype)*dbond(im)*dvdx(  1)
            dhdxa(  1,ic,icx,im) = temph 
            fix = 0.d0
            if (itype .eq. ic .and.   2 .eq. icx) fix = 1.d0
            dpdx(  2) = - bxa(  2)*dpi3*totalp + vect3(  2)*dpi
            dvdx(  2) = - aminusb(  2)*dvi3*totalv + vect5(  2)*dvi
            temph = fix + ivdirect(itype)*dbond(im)*dvdx(  2)
            dhdxa(  2,ic,icx,im) = temph 
            fix = 0.d0
            if (itype .eq. ic .and.   3 .eq. icx) fix = 1.d0
            dpdx(  3) = - bxa(  3)*dpi3*totalp + vect3(  3)*dpi
            dvdx(  3) = - aminusb(  3)*dvi3*totalv + vect5(  3)*dvi
            temph = fix + ivdirect(itype)*dbond(im)*dvdx(  3)
            dhdxa(  3,ic,icx,im) = temph 
          end do
        end do
      end do
C
      return
      end
      SUBROUTINE improper()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,ideform,iat,jat,kat,lat,itype,ib1,ib2,ib3
      real*8 oopsk,diff,c11,dot,c12,c22,rat1,rat2,a1,snt,ophi,cst
      real*8 cstinv,enopbs,f1
      real*8 roij(3),rojk(3),rojl(3),p(3)
      real*8 fi(3),fj(3),fk(3),fl(3)
      real*8 pos1(3),pos2(3),pos3(3),pos4(3)
C
C
C     *****initialization of energy
C
      el2 = box/2.0d0
      eopbs = 0.0d0
C
C     *****4 center interactions (out of plane bends)
C
      do ideform = 1,ndeforms
       itype = deforms(5,ideform)
       oopsk = deform(itype)
       if (oopsk.gt.1.0d-4) then
        iat   = deforms(1,ideform)
        jat   = deforms(2,ideform)
        kat   = deforms(3,ideform)
        lat   = deforms(4,ideform)
        ib1   = deforms(6,ideform)
        ib2   = deforms(7,ideform)
        ib3   = deforms(8,ideform)
C
C     *****for stress tensor
C
        do kk = 1,3
          pos1(kk) = x(kk,iat)
          diff = x(kk,jat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos2(kk) = pos1(kk) + diff
          diff = x(kk,kat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos3(kk) = pos1(kk) + diff
          diff = x(kk,lat) - x(kk,iat)
          if (abs(diff) .gt. el2)diff = diff - dsign(box,diff)
          pos4(kk) = pos1(kk) + diff
        end do
C
C     *****torsional angle (phi) calculation
C
        do kk = 1,3
          roij(kk) = stretches(kk,ib1)
          rojk(kk) = stretches(kk,ib2)
          rojl(kk) = stretches(kk,ib3)
        end do

C       P=-(R_ij X R_jk) = |R_ij||R_jk|sin(theta_ijk)
        p(1) = roij(3)*rojk(2) - roij(2)*rojk(3)
        p(2) = roij(1)*rojk(3) - roij(3)*rojk(1)
        p(3) = roij(2)*rojk(1) - roij(1)*rojk(2)
C
        c11 = dot(p,p)
        c12 = dot(p,rojl)
C       c22 = dot(rojl,rojl)
        c22 = stretches(4,ib3)
        rat1 = c12/c11
        rat2 = c12/c22
C
        a1 = 1.0d0/dsqrt(c11*c22)
        snt= -a1*c12
        if (snt .lt. -1.0d0)snt = -1.0d0
        if (snt .gt.  1.0d0)snt =  1.0d0
        ophi = dasin(snt)
        cst = dcos(ophi)
        if (abs(cst) .lt. delzero)cst = delzero
        cstinv = 1.0d0/cst
C
        enopbs = 0.5d0 * oopsk * ophi * ophi 
        eopbs = eopbs + enopbs
        f1 = -a1*oopsk*ophi*cstinv
C
C     *****compute negative gradient
C
        do kk = 1,3
          fl(kk) = (rojl(kk)*rat2) - p(kk)
        end do
        fi(1) = rat1*(p(3)*rojk(2) - p(2)*rojk(3)) -
     +               (rojl(3)*rojk(2) - rojl(2)*rojk(3))
        fi(2) = rat1*(p(1)*rojk(3) - p(3)*rojk(1)) -
     +               (rojl(1)*rojk(3) - rojl(3)*rojk(1))
        fi(3) = rat1*(p(2)*rojk(1) - p(1)*rojk(2)) -
     +               (rojl(2)*rojk(1) - rojl(1)*rojk(2))

        fk(1) = rat1*(p(3)*roij(2) - p(2)*roij(3)) -
     +               (rojl(3)*roij(2) - rojl(2)*roij(3))
        fk(2) = rat1*(p(1)*roij(3) - p(3)*roij(1)) -
     +               (rojl(1)*roij(3) - rojl(3)*roij(1))
        fk(3) = rat1*(p(2)*roij(1) - p(1)*roij(2)) -
     +               (rojl(2)*roij(1) - rojl(1)*roij(2))
C
        do kk = 1,3
          fj(kk) = 0.0d0 - fi(kk) - fk(kk) - fl(kk)
        end do
C
        do kk = 1,3
          f(kk,iat) = f1*fi(kk) + f(kk,iat)
          f(kk,jat) = f1*fj(kk) + f(kk,jat)
          f(kk,kat) = f1*fk(kk) + f(kk,kat)
          f(kk,lat) = f1*fl(kk) + f(kk,lat)
          do jj = 1,3
            tvirpo(kk,jj) = tvirpo(kk,jj) + f1*fi(kk)*pos1(jj) +
     +                   f1*fj(kk)*pos2(jj) + f1*fk(kk)*pos3(jj) +
     +                      f1*fl(kk)*pos4(jj)
          end do
        end do
       endif
      end do
C
      return
      end
      SUBROUTINE initialize()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer ibond,iat,jat,idtype,itype,idef,ideftype,idbnum,kk
      integer iatdef,jatdef
      real*8 defk
      real*8 x1(3,maxat),diff,ci,rmass,pvir,volinv,dndof,mv2tot
      double precision etrunc
      logical arombond
C
C
      if (lboxdip)then
        open (76,file='fort.76',form='unformatted',status='new')
        close (76)
      end if
      if (lcoords)then
        open (77,file='fort.77',form='unformatted',status='new')
        close (77)
      end if
      if (lstress)then
        open (78,file='fort.78',form='unformatted',status='new')
        close (78)
      end if
      if (lvelocs)then
        open (79,file='fort.79',form='unformatted',status='new')
        close (79)
      end if
      open (70,file='fort.70',status='unknown')
      close(70,status='delete')
      open (65,file='fort.65',status='unknown')
C
      write(65,'(a72)')clinec
      write(65,'(a42)')'#                         Running Averages'
      write(65,'(a1)')'#'
      write(65,'(a39,a30)')'#     Time    Temp     Pressure     Box',
     +                '        Total E    Hamiltonian'
      write(65,'(a39,a30)')'#      fs       K         Atm       Ang',
     +                '        Kcal/mol    Kcal/mol  '
      write(65,'(a72)')clinec
      close (65)
C
C
      el2 = box/2.0d0
      boxini = box
C
      CALL setup()
      CALL exclude()
      CALL DoDummyInit()
      CALL boxes()
      CALL spline()
      intzs0 = int(zs(0)) - 1
C
C     *** is bond a dummy bond?
C
      do ibond = 1,nbonds
        iat = bonds(1,ibond)
        jat = bonds(2,ibond)
        ldummybond(ibond)=.false.
        if (ldummy(iat).or.ldummy(jat)) then
          ldummybond(ibond)=.true.
        endif
      end do
C
      if (constrain)then
        nbcon = 0
        do ibond = 1,nbonds
          iat = bonds(1,ibond)
          jat = bonds(2,ibond)
          itype = bonds(3,ibond)
C
C     *****if hydrogen positions are constrained, don't include
C            bond constraint
C
          arombond = .false.
          do idef = 1,ndeforms
            ideftype = deforms(5,idef)
            defk = deform(ideftype)
            if ( defk .lt. 1.0d-5) then
              iatdef = deforms(2,idef)
              jatdef = deforms(4,idef)
              if (iat.eq. iatdef .and. jat .eq. jatdef) arombond=.true.
              if (iat.eq. jatdef .and. jat .eq. iatdef) arombond=.true.
            end if
          end do
C
          if (ldummybond(ibond)) arombond=.true.! do not constrain dummy bonds
C
          if (.not. arombond) then
            nbcon = nbcon + 1
            bondcon(1,nbcon) = iat
            bondcon(2,nbcon) = jat
            bondcon(3,nbcon) = itype
            massred(nbcon) = 0.5d0*mass(iat)*mass(jat)/
     +                (mass(iat)+mass(jat))
            d2(nbcon) = stretch(2,itype)*stretch(2,itype)
          end if
        end do   ! bonds
C
        numaromatic = 0
        do idef = 1,ndeforms
          ideftype = deforms(5,idef)
          defk = deform(ideftype)
          if ( defk .lt. 1.0d-5) then

C
C     *****type 1 = ca-ca-ca-*ha
C
            numaromatic = numaromatic + 1
            iaromatic(1,numaromatic) = deforms(1,idef)
            iaromatic(2,numaromatic) = deforms(2,idef)
            iaromatic(3,numaromatic) = deforms(3,idef)
            iaromatic(4,numaromatic) = deforms(4,idef)
            idbnum = abs(deforms(8,idef))
            idtype = bonds(3,idbnum)
            dbond(numaromatic) = stretch(3,idtype)
          end if
        end do
C
        nconst = nbcon + 3*numaromatic
        write (6,*) 'nbcon = ',nbcon
        write (6,*) 'numaromatic = ',numaromatic
        write (6,*) 'nconst = ',nconst
C
        CALL shake(x,x,x)
        do  iat = 1,nat
          do kk = 1,3
             x1(kk,iat) = x(kk,iat) + v(kk,iat)*1.d-5*delt
             if (x1(kk,iat) .gt. box) x1(kk,iat) = x1(kk,iat) - box
             if (x1(kk,iat) .lt. 0.0) x1(kk,iat) = x1(kk,iat) + box
          end do
        end do
        CALL shake(x,x1,x1)
        do iat = 1,nat
          do kk = 1,3
            diff = x1(kk,iat) - x(kk,iat)
            if (abs(diff) .gt. el2) diff = diff - dsign(box,diff)
            v(kk,iat) = diff*1.d+5/delt
          end do
        end do
      end if
C
C     *****Energy block
C
      do iat = 1,nat
        do kk = 1,3
          f(kk,iat) = 0.0d0
        end do
      end do
C
      unbd = 0.0d0
C
      CALL en2cen()
      CALL en3cen()
      CALL en4cen()
      CALL improper()
C
C     *****self correction
C
        ewself = 0.0d0
        do iat = 1,nat
          itype = atomtype(iat)
          ci = q(itype)
          ewself = ewself + ci*ci
        end do
        ewself = 332.08d0*alpha*ewself/rootpi
C
        CALL DoDummyCoords()   ! update positions of dummy atoms
        CALL reciprocal()
        CALL correct()
        CALL interch()
        CALL recipQmu()
        CALL DoDummy(f)   ! update positions of dummy atoms
C
      poten = ebond + ebend + etort + eopbs + unbd
C
C     *****End of energy block
C
C
C     ******no. of degrees of freedom
C
      natreal=0
      do iat=1,nat
        if (ldummy(iat)) then
          v(1,iat)=0.0d0
          v(2,iat)=0.0d0
          v(3,iat)=0.0d0
         else
          natreal=natreal+1
        end if
      end do
      ndof = 3*natreal - nconst
      write(6,*)'Total number of degrees of freedom = ',ndof
C
C     *****calculate initial kinetic energy
C
      mv2tot = 0.0d0
      do iat = 1,nat
        mv2tot  = mv2tot + (v(1,iat)*v(1,iat)+v(2,iat)*v(2,iat)+
     +               v(3,iat)*v(3,iat)) * mass(iat)
      end do
C
      kinen = 0.5d0 * mv2tot/(4184.d0*1000.d0)
      write (6,*)'Initial kinetic energy = ',kinen
      toten = kinen + poten
      temp   = mv2tot/(dble(ndof) * gask * 1000.0d0)
      volinv =1.0d0/(box*box*box)
      dndof = dble(ndof-3)/dble(ndof)
      etrunc = esumninj*volinv
      write(6,*) 'etrunc=',etrunc
C
C     *****pressure terms
C
      ptrunc = psumninj*volinv*volinv
      pke = 45.42d0*dble(ndof-3)*temp*volinv
      pvir  = 2.2857d+04*vir*volinv
      pintr =  pvir + ptrunc
      pres       =  pke + pintr
C 
C     *****initialize properties
C
      do iat = 1,nat
          rmass = 4.184d-4 * massinv(iat)
        do kk = 1,3
          accel(kk,iat)     = f(kk,iat)*rmass
          xn(kk,iat)    = x(kk,iat) - v(kk,iat)*1.d-5*delt
     +                 + 0.5d0*accel(kk,iat)*delt*delt
          if (xn(kk,iat) .gt. box)
     +      xn(kk,iat) = xn(kk,iat) - box
          if (xn(kk,iat) .lt. 0.0d0)
     +      xn(kk,iat) = xn(kk,iat) + box
          xn_1(kk,iat)  = x(kk,iat) - 2.*v(kk,iat)*1.d-5*delt
     +                 + 2.0*accel(kk,iat)*delt*delt
          if (xn_1(kk,iat) .gt. box)
     +      xn_1(kk,iat) = xn_1(kk,iat) - box
          if (xn_1(kk,iat) .lt. 0.0)
     +      xn_1(kk,iat) = xn_1(kk,iat) + box
        end do
      end do
C
      return
      end
      SUBROUTINE integrator()
C 
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      real*8 x2(3,maxat),fref(3,maxat)
      integer kinterch,istep,iat,kk,jj
      real*8 rmass,aa,aa2,arg2,poly,bb,scale,diff
      real*8 fs(3,maxat)
      logical newpresstmp
C
      if (nvt) CALL nhcint()
      if (npt) CALL nptint()
      if (nve) CALL getkin()
C
      kount = kount + 1
      kinterch = 0
C
      do istep = 1,multibig
C
        do iat = 1, nat
          rmass = 41.84d0*massinv(iat)
          do kk = 1,3
            v(kk,iat) = v(kk,iat) + 0.5d0*delt*fnow(kk,iat)*rmass
          end do
        end do
C
C     *****update particle positions
C
        if (nvt .or. nve) then
          do iat = 1,nat
            do kk = 1,3
              x(kk,iat) = x(kk,iat) + v(kk,iat)*delt*1.0d-5
            end do
          end do
        end if
C
        if (npt) then
          aa = dexp(delt*0.5d0*vlogv)
          aa2 = aa*aa
          arg2 = (vlogv*delt*0.5d0)*(vlogv*delt*0.5d0)
          poly = (((e8*arg2+e6)*arg2+e4)*arg2+e2)*arg2+1.d0 
          bb = aa*poly*delt
          do iat = 1,nat
            do kk = 1,3
              x(kk,iat) = x(kk,iat)*aa2 + v(kk,iat)*bb*1.0d-5
            end do
          end do
          xlogv = xlogv + vlogv*delt
          scale = exp(vlogv*delt)
          box = box * scale
        end if
C
C     *****apply periodic boundary conditions
C
        do iat= 1,nat
          do kk = 1,3
            if (x(kk,iat) .gt. box) x(kk,iat) = x(kk,iat) - box
            if (x(kk,iat) .lt. 0.0) x(kk,iat) = x(kk,iat) + box
          end do
        end do 
        CALL DoDummyCoords()   ! update positions of dummy atoms
C
        el2 = box*0.5d0
C
        do iat = 1,nat
          do kk = 1,3
            f(kk,iat) = 0.0d0
            fshort(kk,iat) = 0.0d0
          end do
        end do
C
        if (istep .eq. multibig) then
          if (mod(kount,kvir) .eq. 0) then
            newpress = .true.
           else
            newpress = .false.
          end if
        endif
        do jj = 1,3
          do kk = 1,3
            tvirpo(kk,jj) = 0.0d0
          end do
        end do
        vir = 0.0d0
C
        CALL en2cen()
        CALL en3cen()
        CALL improper()
C
        do iat = 1,nat
          do kk = 1,3
            fref(kk,iat)=f(kk,iat)
            fnow(kk,iat) = fref(kk,iat)
          end do
        end do
      
         if ((mod(istep,multimed).eq.0).and.
     +                 (istep.ne.multibig)) then
C
C     *****Energy block
C
C
C
           CALL en4cen()
           CALL interchs()
           CALL DoDummy(f)
C
C     *****End of energy block
C
C
C
           do iat = 1,nat
             do kk = 1,3
               fnow(kk,iat) = fref(kk,iat) +
     $                       multimed*(f(kk,iat)-fref(kk,iat)) 
             end do
           end do
         endif   ! medium timestep 
C          
         if (istep .eq. multibig)then
C
C     *****Energy block
C
           do iat = 1,nat
            do kk = 1,3
             f(kk,iat) = 0.0d0
            end do
           end do
           CALL en4cen()
           do iat = 1,nat
            do kk = 1,3
             fshort(kk,iat) = fref(kk,iat)+f(kk,iat)
             f(kk,iat) = fref(kk,iat)+f(kk,iat)
            end do
           end do
C
           unbd = 0.0d0
C
           CALL reciprocal()
           CALL correct()
           CALL interch()
           CALL recipQmu()
           CALL DoDummy(fshort)
           CALL DoDummy(f)
C
           if (newpress) then
             vir=vir+virdummy
             do kk = 1,3
               do jj=1,3
                 tvirpo(kk,jj)=tvirpo(kk,jj)+tvirdummy(kk,jj)
               end do
             end do
           endif
C        
           do iat = 1,nat
            do kk = 1,3
              fnow(kk,iat) = fref(kk,iat)+
     $                      multimed*(fshort(kk,iat)-fref(kk,iat))
            end do
           end do
C
           if (mod(kount,knben) .eq. 0) CALL nbenergy()
C
C     *****End of energy block
C
           if (constrain .and. newpress)then
             do iat = 1,nat
              rmass = 41.84d0*massinv(iat)
              do kk =1 ,3
                x2(kk,iat) = x(kk,iat)+1.0d-5*delt*v(kk,iat)+
     +                  1.0d-5*(delt*delt)*f(kk,iat)*rmass
                if (x2(kk,iat) .gt. box) x2(kk,iat) = x2(kk,iat) - box
                if (x2(kk,iat) .lt. 0.0) x2(kk,iat) = x2(kk,iat) + box
              end do
             end do
             CALL shake(x,x2,x2)
           end if
C
          do iat = 1,nat
            do kk = 1,3
              fnow(kk,iat)=fnow(kk,iat)+
     $                     multibig*(f(kk,iat)-fshort(kk,iat))
            end do
          end do
        end if   !(istep .eq. multibig) 
C
        if (constrain)then
          do iat = 1,nat
            rmass = 41.84d0*massinv(iat)
            do kk =1 ,3
              x2(kk,iat) = x(kk,iat)+1.0d-5*delt*v(kk,iat)+
     +              1.0d-5*(delt*delt)*fnow(kk,iat)*rmass
              if (x2(kk,iat) .gt. box) x2(kk,iat) = x2(kk,iat) - box
              if (x2(kk,iat) .lt. 0.0) x2(kk,iat) = x2(kk,iat) + box
            end do
          end do
C
          newpresstmp=newpress
          newpress=.false.
          CALL shake(x,x2,x2)
          newpress=newpresstmp
C
          do iat = 1,nat
            do kk = 1,3
              diff = x2(kk,iat) - x(kk,iat)
              if (abs(diff) .gt. el2) diff = diff - dsign(box,diff)
              fnow(kk,iat) =(diff - 1.0d-5*delt*v(kk,iat))*mass(iat)/
     +                 (41.84d-5*delt*delt)
            end do
          end do
        end if
C
        do iat = 1, nat
          rmass = 41.84d0*massinv(iat)
          do kk = 1,3
            v(kk,iat) = v(kk,iat) + 0.5d0*delt*fnow(kk,iat)*rmass
          end do
        end do
      end do
C
      do iat = 1,nat
        do kk = 1,3
          delx(kk,iat) = x(kk,iat) - xn_1(kk,iat)
           if (abs(delx(kk,iat)) .gt. el2 ) delx(kk,iat) =
     +                 delx(kk,iat) - dsign(box,delx(kk,iat))
          xn_1(kk,iat) = x(kk,iat)
        end do
      end do
C
      if (nvt) CALL nhcint()
      if (npt) CALL nptint()
      if (nve) CALL getkin()
C
      poten = ebond + ebend + etort + eopbs + unbd
C
      CALL result()
C
      return
      end
      SUBROUTINE interch()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer iat,itype,neightot,k,jat,jtype,itypee
      integer ipiece
      real*8 deltainv,z,zpiece,zpiece2,zpiece3,zpiece4,zpiece5,ff
      real*8 f1,f2,f3,ffe,xij1,xij2,xij3
      integer nneigh,iter,iflag,kk,jj,kx,ky,kz,kmag2,klimit2
      real*8 tau,tauval
      real*8 dtau,dtauval
      integer*2 listpol(maxnay,maxat),listmpol(maxat)
      real*8 rx(maxnay,maxat),ry(maxnay,maxat),rz(maxnay,maxat)
      real*8 rr(maxnay,maxat),rrinv(maxnay,maxat)
      real*8 find(3,maxat),ffx,ffy,ffz
      real*8 pxnew(maxat),pynew(maxat),pznew(maxat)
      real*8 tx(maxat),ty(maxat),tz(maxat)
      real*8 rxij,ryij,rzij,pjr,pir,diff2,uind,ci,cj,r5inv
      real*8 qq,dx,dy,dz,pipj,p1,p2,p3,p4,p5,ee1,ee2,ee3,ee4,ee5
      real*8 r,rinv,zinv,r3inv,alphar,fact,derfc,derf,tfact
      real*8 ee6,gtaper,staper
      real*8 virind,tvirind(3,3)
      real*8 twoalphapi,facti,factj
C
C     additional variables used for damping of p*grad(T)*p interactions
C     at close distances
C
      real*8 c1(maxcharges,maxcharges)
      real*8 expc1r3(maxnay,maxat),expc1r3val,three_c1r2
      real*8 lamda3,lamda5,dlamda3,dlamda5,c1r3,p6
      real*8 xun(3,maxat)
C
      real*4 DipoleChain(3,maxnch)
      real*4 MassChain(maxnch),RcmChain(3,maxnch),qChain(maxnch)
C
      integer kymin,kzmin,ichain
C
      common /dipChain/ DipoleChain  ! dipoles moments of chains 
C
C     *****begin calculations
C
      unbde=0.0d0
      virind = 0.0d0
      do kk=1,3
        do jj=1,3
          tvirind(kk,jj)=0.0d0
        end do
      end do
      deltainv = 1.0d0/deltaspline
      el2 = box*0.5d0
      twoalphapi = 2.0d0*alpha/rootpi
C
C     initialize c1 array for damping of polarization
C
      do itype=1,maxcharges
        do jtype=itype,maxcharges
         if (pol(itype).gt.1.0d-5.and.pol(jtype).gt.1.0d-5) then
           c1(itype,jtype)=a_thole/dsqrt(pol(itype)*pol(jtype))
           c1(jtype,itype)=c1(itype,jtype)
          else
           c1(itype,jtype)=1.0d6
           c1(jtype,itype)=1.0d6
         endif 
        end do
      end do
C
C     *****begin calculation of non-bonded and electrostatic energies
C          and forces
C
      do iat = 1, nat - 1
        nneigh = 0
        itype  = atomtype(iat)
        neightot = listm(iat)
        do k =  1, neightot
          jat =  list(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
C
          if (z .gt. ro2) goto 32
C
          jtype = atomtype(jat)
          itypee = typee(itype,jtype)
C
          if (z .le. 0.5d0) then
            write(6,'("close approach**",F10.2,4I6)')
     $       ,z,itype,jtype,iat,jat
D            write(6,'(3F10.2)') (x(kk,iat),kk=1,3)
D            write(6,'(3F10.2)') (x(kk,jat),kk=1,3)
            z = 0.5d0
          end if
C
          ipiece = int((z - zs(0))*deltainv)+ 1
C
C     *****determine position within piece of spline
C
          zpiece = z - zs(ipiece-1)
C
C     *****determine powers for spline calculations
C
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
C
C     *****determine energy
C
          unbd = unbd   
     +                  +     coefft(1,ipiece,itypee) 
     +                  +     coefft(2,ipiece,itypee)*zpiece
     +                  +     coefft(3,ipiece,itypee)*zpiece2
     +                  +     coefft(4,ipiece,itypee)*zpiece3
     +                  +     coefft(5,ipiece,itypee)*zpiece4
     +                  +     coefft(6,ipiece,itypee)*zpiece5
C
C     *****determine force 
C
          ff   =  
     +                  +     coeffft(2,ipiece,itypee) 
     +                  +     coeffft(3,ipiece,itypee)*zpiece
     +                  +     coeffft(4,ipiece,itypee)*zpiece2
     +                  +     coeffft(5,ipiece,itypee)*zpiece3
     +                  +     coeffft(6,ipiece,itypee)*zpiece4
C
C     *****insert
C
          f1 = ff*xij1
          f2 = ff*xij2
          f3 = ff*xij3
          f(1,iat) = f(1,iat) + f1
          f(2,iat) = f(2,iat) + f2
          f(3,iat) = f(3,iat) + f3
          f(1,jat) = f(1,jat) - f1
          f(2,jat) = f(2,jat) - f2
          f(3,jat) = f(3,jat) - f3
C
          r = dsqrt(z)
          rinv = 1.0d0/r
          zinv = rinv*rinv
          r3inv = zinv*rinv
C
C     *****storing distances for polarization
C
          nneigh = nneigh + 1
          listpol(nneigh,iat) = jat
          rx(nneigh,iat) = xij1
          ry(nneigh,iat) = xij2
          rz(nneigh,iat) = xij3
          rr(nneigh,iat) = r
          expc1r3(nneigh,iat)=dexp(-c1(itype,jtype)*r**3)  ! for damped pol.
          rrinv(nneigh,iat) = rinv
C
          alphar = alpha*r
          fact = (derfc(alphar)*rinv + twoalphapi*
     +                      dexp(-alphar*alphar))*zinv
          facti = fact*q(itype)
          factj = fact*q(jtype)
          elf(1,iat) = elf(1,iat) - factj*xij1
          elf(2,iat) = elf(2,iat) - factj*xij2
          elf(3,iat) = elf(3,iat) - factj*xij3
          elf(1,jat) = elf(1,jat) + facti*xij1
          elf(2,jat) = elf(2,jat) + facti*xij2
          elf(3,jat) = elf(3,jat) + facti*xij3
C
C
         if (z .lt. rshort2)then
          if (z .gt. ros2)then
          gtaper = (dsqrt(zs(ipiece-1)) - (ros))/driftmax
          staper = 1.0d0+(gtaper*gtaper*(2.0d0*gtaper-3.0d0))
          else
          staper = 1.0d0
          end if
          fshort(1,iat) = fshort(1,iat) + f1*staper
          fshort(2,iat) = fshort(2,iat) + f2*staper
          fshort(3,iat) = fshort(3,iat) + f3*staper
          fshort(1,jat) = fshort(1,jat) - f1*staper
          fshort(2,jat) = fshort(2,jat) - f2*staper
          fshort(3,jat) = fshort(3,jat) - f3*staper
         end if
C
C     ***** virial update every newpress 
C      
          if (newpress) then
          tvirpo(1,1) = tvirpo(1,1) - ff*xij1*xij1
          tvirpo(2,1) = tvirpo(2,1) - ff*xij2*xij1
          tvirpo(3,1) = tvirpo(3,1) - ff*xij3*xij1
          tvirpo(1,2) = tvirpo(1,2) - ff*xij1*xij2
          tvirpo(2,2) = tvirpo(2,2) - ff*xij2*xij2
          tvirpo(3,2) = tvirpo(3,2) - ff*xij3*xij2
          tvirpo(1,3) = tvirpo(1,3) - ff*xij1*xij3
          tvirpo(2,3) = tvirpo(2,3) - ff*xij2*xij3
          tvirpo(3,3) = tvirpo(3,3) - ff*xij3*xij3
              ffe   =      
     +        +            coefffee(2,ipiece,itypee) 
     +        +            coefffee(3,ipiece,itypee)*zpiece
     +        +            coefffee(4,ipiece,itypee)*zpiece2
     +        +            coefffee(5,ipiece,itypee)*zpiece3
     +        +            coefffee(6,ipiece,itypee)*zpiece4
              ff = ff - ffe 
              unbde =  unbde 
     +                  + coeffee(1,ipiece,itypee) 
     +                  + coeffee(2,ipiece,itypee)*zpiece 
     +                  + coeffee(3,ipiece,itypee)*zpiece2
     +                  + coeffee(4,ipiece,itypee)*zpiece3
     +                  + coeffee(5,ipiece,itypee)*zpiece4
     +                  + coeffee(6,ipiece,itypee)*zpiece5
          vir = vir - ff*z
          end if
32      continue
        end do
        listmpol(iat) = nneigh
      end do
C
      if (lredonefour) then
       do iat = 1, nat 
        itype  = atomtype(iat)
        do k =  1, listm14(iat)
          jat =  list14(k,iat)
          jtype = atomtype(jat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
          zinv=1.0d0/z
          itypee = typee(itype,jtype)
C
          ipiece = int((z - zs(0))*deltainv)+ 1
C
C     *****determine position within piece of spline
C
          zpiece = z - zs(ipiece-1)
C
C     *****determine powers for spline calculations
C
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
C
C     *****determine energy
C
          unbd = unbd -       redfactor*(   
     $                        coefft(1,ipiece,itypee) 
     $                  +     coefft(2,ipiece,itypee)*zpiece
     $                  +     coefft(3,ipiece,itypee)*zpiece2
     $                  +     coefft(4,ipiece,itypee)*zpiece3
     $                  +     coefft(5,ipiece,itypee)*zpiece4
     $                  +     coefft(6,ipiece,itypee)*zpiece5)
C
C     *****determine force 
C
          ff   =  
     $                  +     coeffft(2,ipiece,itypee) 
     $                  +     coeffft(3,ipiece,itypee)*zpiece
     $                  +     coeffft(4,ipiece,itypee)*zpiece2
     $                  +     coeffft(5,ipiece,itypee)*zpiece3
     $                  +     coeffft(6,ipiece,itypee)*zpiece4
C
C     *****insert
C
          ff=-redfactor*ff
          f1 = ff*xij1
          f2 = ff*xij2
          f3 = ff*xij3
          f(1,iat) = f(1,iat) + f1
          f(2,iat) = f(2,iat) + f2
          f(3,iat) = f(3,iat) + f3
          f(1,jat) = f(1,jat) - f1
          f(2,jat) = f(2,jat) - f2
          f(3,jat) = f(3,jat) - f3
C
         if (z .lt. rshort2)then
           if (z .gt. ros2)then
            gtaper = (dsqrt(zs(ipiece-1)) - (ros))/driftmax
            staper = 1.0d0+(gtaper*gtaper*(2.0d0*gtaper-3.0d0))
           else
            staper = 1.0d0
           end if
           fshort(1,iat) = fshort(1,iat) + f1*staper
           fshort(2,iat) = fshort(2,iat) + f2*staper
           fshort(3,iat) = fshort(3,iat) + f3*staper
           fshort(1,jat) = fshort(1,jat) - f1*staper
           fshort(2,jat) = fshort(2,jat) - f2*staper
           fshort(3,jat) = fshort(3,jat) - f3*staper
          end if
C
          r = dsqrt(z)
          rinv=1.0d0/r
C
          alphar = alpha*r
          fact = (derfc(alphar)*rinv + twoalphapi*
     +                      dexp(-alphar*alphar))*zinv
          fact=(-redfactor)*fact
          facti = fact*q(itype)
          factj = fact*q(jtype)
          elf(1,iat) = elf(1,iat) - factj*xij1
          elf(2,iat) = elf(2,iat) - factj*xij2
          elf(3,iat) = elf(3,iat) - factj*xij3
          elf(1,jat) = elf(1,jat) + facti*xij1
          elf(2,jat) = elf(2,jat) + facti*xij2
          elf(3,jat) = elf(3,jat) + facti*xij3
C
C
C     ***** virial update every newpress 
C      
          if (newpress) then
          tvirpo(1,1) = tvirpo(1,1) - ff*xij1*xij1
          tvirpo(2,1) = tvirpo(2,1) - ff*xij2*xij1
          tvirpo(3,1) = tvirpo(3,1) - ff*xij3*xij1
          tvirpo(1,2) = tvirpo(1,2) - ff*xij1*xij2
          tvirpo(2,2) = tvirpo(2,2) - ff*xij2*xij2
          tvirpo(3,2) = tvirpo(3,2) - ff*xij3*xij2
          tvirpo(1,3) = tvirpo(1,3) - ff*xij1*xij3
          tvirpo(2,3) = tvirpo(2,3) - ff*xij2*xij3
          tvirpo(3,3) = tvirpo(3,3) - ff*xij3*xij3
              ffe   =       
     $        +            coefffee(2,ipiece,itypee) 
     $        +            coefffee(3,ipiece,itypee)*zpiece
     $        +            coefffee(4,ipiece,itypee)*zpiece2
     $        +            coefffee(5,ipiece,itypee)*zpiece3
     $        +            coefffee(6,ipiece,itypee)*zpiece4
              ffe=(-redfactor)*ffe
              ff = ff - ffe 
              unbde =  unbde -redfactor*(
     $                    coeffee(1,ipiece,itypee) 
     $                  + coeffee(2,ipiece,itypee)*zpiece 
     $                  + coeffee(3,ipiece,itypee)*zpiece2
     $                  + coeffee(4,ipiece,itypee)*zpiece3
     $                  + coeffee(5,ipiece,itypee)*zpiece4
     $                  + coeffee(6,ipiece,itypee)*zpiece5)
          vir = vir - ff*z
          end if
        end do
       end do
      endif
C
      vir = vir + unbde
C
      if (.not.lpolarizable) return
C
      if (lredQ_mu14) then
       do iat = 1, nat
        itype  = atomtype(iat)
        do k =  1, listm14(iat)
          jat =  list14(k,iat)
          jtype = atomtype(jat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
          zinv=1.0d0/z
C
          r = dsqrt(z)
          rinv=1.0d0/r
          alphar = alpha*r
          fact = (derfc(alphar)*rinv + twoalphapi*
     +                      dexp(-alphar*alphar))*zinv
          fact=(-redQmufactor)*fact
          facti = fact*q(itype)
          factj = fact*q(jtype)
          elf(1,iat) = elf(1,iat) - factj*xij1
          elf(2,iat) = elf(2,iat) - factj*xij2
          elf(3,iat) = elf(3,iat) - factj*xij3
          elf(1,jat) = elf(1,jat) + facti*xij1
          elf(2,jat) = elf(2,jat) + facti*xij2
          elf(3,jat) = elf(3,jat) + facti*xij3
C
        end do
       end do
      endif   ! (lredQ_mu14)
C
      iter = 0
7000  continue
      iter = iter + 1
      do iat = 1,nat
        tx(iat) = 0.0d0
        ty(iat) = 0.0d0
        tz(iat) = 0.0d0
      end do
C
      iflag = 0
      do iat = 1,nat
        itype=atomtype(iat)
        do k = 1,listmpol(iat)
          jat = listpol(k,iat)
          jtype=atomtype(jat)
          rinv = rrinv(k,iat)
          r=rr(k,iat)
          tauval = tau(r)
          zinv = rinv*rinv
          fact = rinv*zinv*tauval
          rxij = rx(k,iat)
          ryij = ry(k,iat)
          rzij = rz(k,iat)
          c1r3=c1(itype,jtype)*r**3    ! damped pol.
          expc1r3val=expc1r3(k,iat)            ! damped pol.
          lamda3=1.d0-expc1r3val               ! damped pol.
          lamda5=1.d0-expc1r3val*(1.d0+c1r3)   ! damped pol.
C
          pjr = 3.0d0*zinv*fact*(px(jat)*rxij+py(jat)*ryij+
     +               pz(jat)*rzij)*lamda5
          pir = 3.0d0*zinv*fact*(px(iat)*rxij+py(iat)*ryij+
     +               pz(iat)*rzij)*lamda5
          tfact = arf-fact*lamda3
C
          tx(iat) = tx(iat) + pjr*rxij + px(jat)*tfact
          ty(iat) = ty(iat) + pjr*ryij + py(jat)*tfact
          tz(iat) = tz(iat) + pjr*rzij + pz(jat)*tfact
C
          tx(jat) = tx(jat) + pir*rxij + px(iat)*tfact
          ty(jat) = ty(jat) + pir*ryij + py(iat)*tfact
          tz(jat) = tz(jat) + pir*rzij + pz(iat)*tfact
C
        end do
        pxnew(iat) = pol(itype)*(elf(1,iat)+tx(iat)+arf*px(iat))
        pynew(iat) = pol(itype)*(elf(2,iat)+ty(iat)+arf*py(iat))
        pznew(iat) = pol(itype)*(elf(3,iat)+tz(iat)+arf*pz(iat))
        dx = pxnew(iat) - px(iat)
        dy = pynew(iat) - py(iat)
        dz = pznew(iat) - pz(iat)
        diff2 = dx*dx + dy*dy + dz*dz
        px(iat) = pxnew(iat)
        py(iat) = pynew(iat)
        pz(iat) = pznew(iat)
        if (diff2 .gt. tolpol)iflag = 1
      end do
C
      if (iter.gt.100) then
        write(6,*) 'iter>100'
        write(6,*) 'polarization did not converge'
        stop
      endif 
      if (iflag .ne. 0) goto 7000
C
C     *****induced energy calculation
C
      uind=0.0d0
      do iat = 1,nat
        uind = uind - (elf(1,iat)*px(iat) + elf(2,iat)*py(iat) +
     +                    elf(3,iat)*pz(iat))
      end do
C
      unbd = unbd + uind*0.5d0*332.08d0
C
C     Calculate diple moment of the box that includes induced dipole moments
C
      if (kount.gt.1.and.lboxdip.and.(mod(kount,kboxdip).eq.0))then
        do kk=1,3
          DipoleTot(kk)=0.0d0
          DipolePol(kk)=0.0d0
        end do
        do ichain=1,maxnch
          DipoleChain(1,ichain)=0.0d0
          DipoleChain(2,ichain)=0.0d0
          DipoleChain(3,ichain)=0.0d0
          qChain(ichain)=0.0d0
          MassChain(ichain)=0.0d0
          RcmChain(1,ichain)=0.0d0
          RcmChain(2,ichain)=0.0d0
          RcmChain(3,ichain)=0.0d0
        end do
        CALL unwrap(x,xun)
        do iat=1,nat   ! first to induced dipole contribution
          itype = atomtype(iat)
          ci = q(itype)
          ichain=chain(iat) 
          DipoleChain(1,ichain)=DipoleChain(1,ichain)+xun(1,iat)*ci
     $       +px(iat)
          DipoleChain(2,ichain)=DipoleChain(2,ichain)+xun(2,iat)*ci
     $       +py(iat)
          DipoleChain(3,ichain)=DipoleChain(3,ichain)+xun(3,iat)*ci
     $       +pz(iat)
          qChain(ichain)=qChain(ichain)+ci
          MassChain(ichain)=MassChain(ichain)+mass(iat)
          RcmChain(1,ichain)=RcmChain(1,ichain)+xun(1,iat)*mass(iat)
          RcmChain(2,ichain)=RcmChain(2,ichain)+xun(2,iat)*mass(iat)
          RcmChain(3,ichain)=RcmChain(3,ichain)+xun(3,iat)*mass(iat)
        end do
        do ichain=1,maxnch
          RcmChain(1,ichain)=RcmChain(1,ichain)/MassChain(ichain)
          RcmChain(2,ichain)=RcmChain(2,ichain)/MassChain(ichain)
          RcmChain(3,ichain)=RcmChain(3,ichain)/MassChain(ichain)
          DipoleChain(1,ichain)=DipoleChain(1,ichain)
     $         -RcmChain(1,ichain)*qChain(ichain)
          DipoleChain(2,ichain)=DipoleChain(2,ichain)
     $         -RcmChain(2,ichain)*qChain(ichain)
          DipoleChain(3,ichain)=DipoleChain(3,ichain)
     $         -RcmChain(3,ichain)*qChain(ichain)
        end do
        do ichain=1,maxnch
          DipoleTot(1)=DipoleTot(1)+DipoleChain(1,ichain)
          DipoleTot(2)=DipoleTot(2)+DipoleChain(2,ichain)
          DipoleTot(3)=DipoleTot(3)+DipoleChain(3,ichain)
        end do
        do iat=1,nat
          DipolePol(1)=DipolePol(1)+px(iat)
          DipolePol(2)=DipolePol(2)+py(iat)
          DipolePol(3)=DipolePol(3)+pz(iat)
        end do
      endif   ! (lboxdip .and. mod(kount,kboxdip) .eq. 0)
C
C     *****calculate induced forces
C
      do iat = 1,nat
        do kk = 1,3
          find(kk,iat) = 0.0d0
        end do
      end do
C
      do iat = 1,nat - 1
        itype = atomtype(iat)
        ci = q(itype)
        do k = 1,listmpol(iat)
          jat = listpol(k,iat)
          jtype = atomtype(jat)
          cj = q(jtype)
          r = rr(k,iat)
          rinv = rrinv(k,iat)
          z = r*r
          zinv = rinv*rinv
          r3inv = zinv*rinv
          r5inv = r3inv*zinv
          rxij = rx(k,iat)
          ryij = ry(k,iat)
          rzij = rz(k,iat)
C
          c1r3=c1(itype,jtype)*r**3    ! damped pol.
          expc1r3val=expc1r3(k,iat)            ! damped pol.
          lamda3=1.d0-expc1r3val               ! damped pol.
          lamda5=1.d0-expc1r3val*(1.d0+c1r3)   ! damped pol.
          three_c1r2=3.d0*c1(itype,jtype)*z    ! damped pol.
          dlamda3=three_c1r2*expc1r3val                ! damped pol.
          dlamda5=three_c1r2*c1r3*expc1r3val           ! damped pol.
C
          pjr = px(jat)*rxij+py(jat)*ryij+ pz(jat)*rzij
          pir = px(iat)*rxij+py(iat)*ryij+ pz(iat)*rzij
          pipj = px(jat)*px(iat) + py(jat)*py(iat) + pz(jat)*pz(iat)
          qq = ci*pjr - cj*pir
C
C     *****p*grad(T)*p contribution to the force and virial
C
          tauval = tau(r)
          dtauval = dtau(r)
          p1 = 5.0d0*zinv*pjr*pir*lamda5 - pipj*lamda3
          p2 = dtauval*rinv*(3.0d0*r5inv*pir*pjr*lamda5-
     $            pipj*r3inv*lamda3)
          p3 = 3.0d0*r5inv*tauval
          p4 = p3*p1 - p2
C
C         p6 is a derivative of the damped terms
C
          p6=(3.0d0*r5inv*pir*pjr*dlamda5-pipj*r3inv*dlamda3)*rinv
C
C
C          p5 = 3.0d0*r5inv*pir*pjr-pipj*r3inv
C
C          virind = virind - p5*(3.0d0*tauval - dtauval*r)
C
C     *****p*grad(E) contribution to the force and virial
C
          alphar = alpha*r
          ee1 = derfc(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
C          ee4 = ee2*(1.0d0+alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
C
C          virind = virind - 2.0d0*qq*(ee1+ee4)
          f1 = p4 + ee5 - p6*tauval
          f2 = -(ee6*ci + p3*pir*lamda5)
          f3 = ee6*cj - p3*pjr*lamda5
C
          ffx=f1*rxij+f2*px(jat)+f3*px(iat)
          ffy=f1*ryij+f2*py(jat)+f3*py(iat)
          ffz=f1*rzij+f2*pz(jat)+f3*pz(iat)
          find(1,iat)=find(1,iat)+ffx
          find(2,iat)=find(2,iat)+ffy
          find(3,iat)=find(3,iat)+ffz
          find(1,jat)=find(1,jat)-ffx
          find(2,jat)=find(2,jat)-ffy
          find(3,jat)=find(3,jat)-ffz
C    added in v1.2, calculation of the virial due to real part of Q-mu
          virind=virind-(ffx*rxij+ffy*ryij+ffz*rzij)
          tvirind(1,1) = tvirind(1,1) - ffx*rxij
          tvirind(2,1) = tvirind(2,1) - ffy*rxij
          tvirind(3,1) = tvirind(3,1) - ffz*rxij
          tvirind(1,2) = tvirind(1,2) - ffx*ryij
          tvirind(2,2) = tvirind(2,2) - ffy*ryij
          tvirind(3,2) = tvirind(3,2) - ffz*ryij
          tvirind(1,3) = tvirind(1,3) - ffx*rzij
          tvirind(2,3) = tvirind(2,3) - ffy*rzij
          tvirind(3,3) = tvirind(3,3) - ffz*rzij
C
        end do
      end do
C
C     ***  reciprocal part for Q-mu has been moved to recipQmu()
C
C     ***  correct2 part
C
      do iat = 1, nat - 1
        itype = atomtype(iat)
        ci = q(itype)
        neightot = listmex(iat)
        do k  = 1, neightot
          jat = listex(k,iat)
          jtype = atomtype(jat)
          cj = q(jtype)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if( abs(xij1) .gt. el2)xij1 = xij1 - dsign(box,xij1)
          if( abs(xij2) .gt. el2)xij2 = xij2 - dsign(box,xij2)
          if( abs(xij3) .gt. el2)xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          zinv = 1.0d0/z
          rinv = 1.0d0/r
          r3inv = zinv*rinv
C
          pir=px(iat)*xij1+py(iat)*xij2+pz(iat)*xij3
          pjr=px(jat)*xij1+py(jat)*xij2+pz(jat)*xij3
          qq = ci*pjr - cj*pir
          alphar = alpha*r
          ee1 = -derf(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
C          ee4 = ee2*(1.0d0+alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
C
          f1 = ee5*xij1 - ee6*(ci*px(jat)-cj*px(iat))
          f2 = ee5*xij2 - ee6*(ci*py(jat)-cj*py(iat))
          f3 = ee5*xij3 - ee6*(ci*pz(jat)-cj*pz(iat))
C
C           virind = virind - 2.0d0*qq*(ee4 + ee1)  ! using virial
           virind = virind - ee5*z + ee6*qq ! using force
C
          find(1,iat) = find(1,iat) + f1
          find(2,iat) = find(2,iat) + f2
          find(3,iat) = find(3,iat) + f3
          find(1,jat) = find(1,jat) - f1
          find(2,jat) = find(2,jat) - f2
          find(3,jat) = find(3,jat) - f3
          tvirind(1,1) = tvirind(1,1) - f1*xij1
          tvirind(2,1) = tvirind(2,1) - f2*xij1
          tvirind(3,1) = tvirind(3,1) - f3*xij1
          tvirind(1,2) = tvirind(1,2) - f1*xij2
          tvirind(2,2) = tvirind(2,2) - f2*xij2
          tvirind(3,2) = tvirind(3,2) - f3*xij2
          tvirind(1,3) = tvirind(1,3) - f1*xij3
          tvirind(2,3) = tvirind(2,3) - f2*xij3
          tvirind(3,3) = tvirind(3,3) - f3*xij3
C
        end do
      end do

      if (lredonefour) then
      do iat = 1,nat 
        itype = atomtype(iat)
        ci = q(itype)
        do k = 1,listm14(iat)
          jat = list14(k,iat)
          jtype = atomtype(jat)
          cj = q(jtype)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          rinv = 1.0d0/r
          zinv=rinv*rinv
          z = r*r
          zinv = rinv*rinv
          r3inv = zinv*rinv
          rxij = xij1
          ryij = xij2
          rzij = xij3
          pjr = px(jat)*rxij+py(jat)*ryij+ pz(jat)*rzij
          pir = px(iat)*rxij+py(iat)*ryij+ pz(iat)*rzij
          pipj = px(jat)*px(iat) + py(jat)*py(iat) + pz(jat)*pz(iat)
          qq = ci*pjr - cj*pir
C
C     *****p*grad(E) contribution to the force and virial
C
          alphar = alpha*r
          ee1 = derfc(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
C
          f1 = ee5
          f2 =-ee6*ci 
          f3 = ee6*cj
C
          ffx=(f1*rxij+f2*px(jat)+f3*px(iat))*(-redfactor)
          ffy=(f1*ryij+f2*py(jat)+f3*py(iat))*(-redfactor)
          ffz=(f1*rzij+f2*pz(jat)+f3*pz(iat))*(-redfactor)
          find(1,iat)=find(1,iat)+ffx
          find(2,iat)=find(2,iat)+ffy
          find(3,iat)=find(3,iat)+ffz
          find(1,jat)=find(1,jat)-ffx
          find(2,jat)=find(2,jat)-ffy
          find(3,jat)=find(3,jat)-ffz
C    added in v1.2, calculation of the virial due to real part of Q-mu
          virind=virind-(ffx*rxij+ffy*ryij+ffz*rzij)
          tvirind(1,1) = tvirind(1,1) - ffx*rxij
          tvirind(2,1) = tvirind(2,1) - ffy*rxij
          tvirind(3,1) = tvirind(3,1) - ffz*rxij
          tvirind(1,2) = tvirind(1,2) - ffx*ryij
          tvirind(2,2) = tvirind(2,2) - ffy*ryij
          tvirind(3,2) = tvirind(3,2) - ffz*ryij
          tvirind(1,3) = tvirind(1,3) - ffx*rzij
          tvirind(2,3) = tvirind(2,3) - ffy*rzij
          tvirind(3,3) = tvirind(3,3) - ffz*rzij
C
C     *****correct2 part
C
          alphar = alpha*r
          ee1 = -derf(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
C
          f1 =(ee5*xij1 - ee6*(ci*px(jat)-cj*px(iat)))*(redfactor)
          f2 =(ee5*xij2 - ee6*(ci*py(jat)-cj*py(iat)))*(redfactor)
          f3 =(ee5*xij3 - ee6*(ci*pz(jat)-cj*pz(iat)))*(redfactor)
C
          virind = virind-redfactor*(ee5*z - ee6*qq) ! using force
C
          find(1,iat) = find(1,iat) + f1
          find(2,iat) = find(2,iat) + f2
          find(3,iat) = find(3,iat) + f3
          find(1,jat) = find(1,jat) - f1
          find(2,jat) = find(2,jat) - f2
          find(3,jat) = find(3,jat) - f3
          tvirind(1,1) = tvirind(1,1) - f1*rxij
          tvirind(2,1) = tvirind(2,1) - f2*rxij
          tvirind(3,1) = tvirind(3,1) - f3*rxij
          tvirind(1,2) = tvirind(1,2) - f1*ryij
          tvirind(2,2) = tvirind(2,2) - f2*ryij
          tvirind(3,2) = tvirind(3,2) - f3*ryij
          tvirind(1,3) = tvirind(1,3) - f1*rzij
          tvirind(2,3) = tvirind(2,3) - f2*rzij
          tvirind(3,3) = tvirind(3,3) - f3*rzij
C
         end do
        end do
       endif
C
      if (lredQ_mu14) then
      do iat = 1,nat - 1 
        itype = atomtype(iat)
        ci = q(itype)
        do k = 1,listm14(iat)
          jat = list14(k,iat)
          jtype = atomtype(jat)
          cj = q(jtype)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
C
          r = dsqrt(z)
          rinv = 1.0d0/r
          zinv=rinv*rinv
          z = r*r
          zinv = rinv*rinv
          r3inv = zinv*rinv
          rxij = xij1
          ryij = xij2
          rzij = xij3
          pjr = px(jat)*rxij+py(jat)*ryij+ pz(jat)*rzij
          pir = px(iat)*rxij+py(iat)*ryij+ pz(iat)*rzij
          pipj = px(jat)*px(iat) + py(jat)*py(iat) + pz(jat)*pz(iat)
          qq = ci*pjr - cj*pir
C
C     *****p*grad(E) contribution to the force and virial
C
          alphar = alpha*r
          ee1 = derfc(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
C
          f1 = ee5
          f2 =-ee6*ci 
          f3 = ee6*cj 
C
          ffx=(f1*rxij+f2*px(jat)+f3*px(iat))*(-redQmufactor)
          ffy=(f1*ryij+f2*py(jat)+f3*py(iat))*(-redQmufactor)
          ffz=(f1*rzij+f2*pz(jat)+f3*pz(iat))*(-redQmufactor)
          find(1,iat)=find(1,iat)+ffx
          find(2,iat)=find(2,iat)+ffy
          find(3,iat)=find(3,iat)+ffz
          find(1,jat)=find(1,jat)-ffx
          find(2,jat)=find(2,jat)-ffy
          find(3,jat)=find(3,jat)-ffz
C    added in v1.2, calculation of the virial due to real part of Q-mu
          virind=virind-(ffx*rxij+ffy*ryij+ffz*rzij)
          tvirind(1,1) = tvirind(1,1) - ffx*rxij
          tvirind(2,1) = tvirind(2,1) - ffy*rxij
          tvirind(3,1) = tvirind(3,1) - ffz*rxij
          tvirind(1,2) = tvirind(1,2) - ffx*ryij
          tvirind(2,2) = tvirind(2,2) - ffy*ryij
          tvirind(3,2) = tvirind(3,2) - ffz*ryij
          tvirind(1,3) = tvirind(1,3) - ffx*rzij
          tvirind(2,3) = tvirind(2,3) - ffy*rzij
          tvirind(3,3) = tvirind(3,3) - ffz*rzij
C
C
C     *****correct2 part
C
          alphar = alpha*r
          ee1 = -derf(alphar)*r3inv
C          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
C          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
C    ee2 and ee3 are the same as above
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
C
          f1 =(ee5*xij1-ee6*(ci*px(jat)-cj*px(iat)))*(redQmufactor)
          f2 =(ee5*xij2-ee6*(ci*py(jat)-cj*py(iat)))*(redQmufactor)
          f3 =(ee5*xij3-ee6*(ci*pz(jat)-cj*pz(iat)))*(redQmufactor)
C
          virind = virind-redQmufactor*(ee5*z - ee6*qq) ! using force
C
          find(1,iat) = find(1,iat) + f1
          find(2,iat) = find(2,iat) + f2
          find(3,iat) = find(3,iat) + f3
          find(1,jat) = find(1,jat) - f1
          find(2,jat) = find(2,jat) - f2
          find(3,jat) = find(3,jat) - f3
          tvirind(1,1) = tvirind(1,1) - f1*rxij
          tvirind(2,1) = tvirind(2,1) - f2*rxij
          tvirind(3,1) = tvirind(3,1) - f3*rxij
          tvirind(1,2) = tvirind(1,2) - f1*ryij
          tvirind(2,2) = tvirind(2,2) - f2*ryij
          tvirind(3,2) = tvirind(3,2) - f3*ryij
          tvirind(1,3) = tvirind(1,3) - f1*rzij
          tvirind(2,3) = tvirind(2,3) - f2*rzij
          tvirind(3,3) = tvirind(3,3) - f3*rzij
C
         end do
        end do
       end if  ! (lredQ_mu14)
C
      do iat = 1,nat
        do kk = 1,3
          f(kk,iat) = f(kk,iat) + 332.08d0*find(kk,iat)
        end do
      end do
C   
      if (newpress) then
        vir = vir + virind*332.08d0
        do kk=1,3
          do jj=1,3
            tvirpo(kk,jj)=tvirpo(kk,jj)+332.08d0*tvirind(kk,jj)
          end do
        end do
C      write(6,*) "virind=",virind*332.08d0*2.2857d+04/(box**3)
      endif
C     
      return
      end
      SUBROUTINE interchs()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer iat,ichain,itype,neightot,k,jat,jtype,itypee
      integer ipiece,kk
      real*8 deltainv,z,zpiece,zpiece2,zpiece3,zpiece4,zpiece5,ff
      real*8 f1,f2,f3,ffe,xij1,xij2,xij3
C
C     *****begin calculations
C
      deltainv = 1.0d0/deltaspline
      el2 = box*0.5d0
C
C     *****begin calculation of non-bonded and electrostatic energies
C          and forces
C
      do iat = 1, nat - 1
        itype  = atomtype(iat)
        neightot = listms(iat)
        do k =  1, neightot
          jat =  lists(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
C
          if (z .gt. rshort2) goto 32
C
          jtype = atomtype(jat)
          itypee = typee(itype,jtype)
C
          if (z .le. 0.5d0) then
            write(6,'("close approach",F10.2,4I6)')
     $       ,z,itype,jtype,iat,jat
D            write(6,'(3F10.2)') (x(kk,iat),kk=1,3)
D            write(6,'(3F10.2)') (x(kk,jat),kk=1,3)
            z = 0.5d0
          end if
C
          ipiece = int((z - zs(0))*deltainv)+ 1
C
C     *****determine position within piece of spline
C
          zpiece = z - zs(ipiece-1)
C
C     *****determine powers for spline calculations
C
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
C
C     *****determine force 
C
          ff   =  
     +                  +     coeffft1(2,ipiece,itypee) 
     +                  +     coeffft1(3,ipiece,itypee)*zpiece
     +                  +     coeffft1(4,ipiece,itypee)*zpiece2
     +                  +     coeffft1(5,ipiece,itypee)*zpiece3
     +                  +     coeffft1(6,ipiece,itypee)*zpiece4
C
C     *****insert
C
          f1 = ff*xij1
          f2 = ff*xij2
          f3 = ff*xij3
          f(1,iat) = f(1,iat) + f1
          f(2,iat) = f(2,iat) + f2
          f(3,iat) = f(3,iat) + f3
          f(1,jat) = f(1,jat) - f1
          f(2,jat) = f(2,jat) - f2
          f(3,jat) = f(3,jat) - f3
C
32      continue
        end do
      end do
C
      if (lredonefour) then
      do iat = 1, nat 
        itype  = atomtype(iat)
        do k =  1, listm14(iat)
          jat =  list14(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
C
          jtype = atomtype(jat)
          itypee = typee(itype,jtype)
C
          ipiece = int((z - zs(0))*deltainv)+ 1
C
C     *****determine position within piece of spline
C
          zpiece = z - zs(ipiece-1)
C
C     *****determine powers for spline calculations
C
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
C
C     *****determine force 
C
          ff   =  
     +                  +     coeffft1(2,ipiece,itypee) 
     +                  +     coeffft1(3,ipiece,itypee)*zpiece
     +                  +     coeffft1(4,ipiece,itypee)*zpiece2
     +                  +     coeffft1(5,ipiece,itypee)*zpiece3
     +                  +     coeffft1(6,ipiece,itypee)*zpiece4
C
C     *****insert
C
          ff=ff*(-redfactor)
          f1 = ff*xij1
          f2 = ff*xij2
          f3 = ff*xij3
          f(1,iat) = f(1,iat) + f1
          f(2,iat) = f(2,iat) + f2
          f(3,iat) = f(3,iat) + f3
          f(1,jat) = f(1,jat) - f1
          f(2,jat) = f(2,jat) - f2
          f(3,jat) = f(3,jat) - f3
C
        end do
      end do
      endif
C
      return
      end
      SUBROUTINE intsetup()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer i,kk,iat
      real*8 wys(5)
C
C
C     *****initialization of system parameters
C
      stepout = delt*multibig
      kave = nint(dble(nave)/stepout)
      kvir = nint(dble(nvir)/stepout)
      knben = nint(dble(nnben)/stepout)
      koutput = nint(dble(noutput)/stepout)
      kinit = nint(dble(ninit)/stepout)
      kboxdip = nint(dble(nboxdip)/stepout)
      kcoords = nint(dble(ncoords)/stepout)
      kvelocs = nint(dble(nvelocs)/stepout)
      kstress = nint(dble(nstress)/stepout)
      if (kave .eq. 0) kave = 1
      if (kinit .eq. 0) kinit = 1
      if (kvir .eq. 0) kvir = 1
      if (koutput .eq. 0) koutput = 1
      if (knben .eq. 0) knben = 1
C
      nhcstep1 = 5
      nhcstep2 = 5
C
C     *****thermostat mass
C
      qtmass(1) = dble(ndof)*gaskinkc*tstart/(qfreq**2)
C
C     *****barostat mass
C
      wtmass(1) = dble(ndof + 3)*gaskinkc*tstart/(wfreq**2)
C
C     *****parameters in nose-hoover + andersen-hoover methods
C
      do i = 2,10
        qtmass(i) = gaskinkc*tstart/(qfreq**2)
        wtmass(i)=dble(ndof + 3)*gaskinkc*tstart/(3.d0*wfreq**2)
      end do
C
C     *****parameters in the Yoshida/Suzuki integration
C
      wys(1) = 1.d0
      if (nhcstep2.eq.5) then
        wys(1) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
        wys(2) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
        wys(3) =-4.d0**(1.d0/3.d0)/(4.d0-4.d0**(1.d0/3.d0))
        wys(4) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
        wys(5) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
      end if
      if (nhcstep2.eq.3) then
        wys(1) = 1.d0/(2.d0-2.d0**(1.d0/3.d0))
        wys(2) =-2.d0**(1.d0/3.d0)/(2.d0-2.d0**(1.d0/3.d0))
        wys(3) = 1.d0/(2.d0-2.d0**(1.d0/3.d0))
      end if
C
      do i = 1,nhcstep2
        wdt1(i) = wys(i)*stepout/dble(nhcstep1)
        wdt2(i) = wdt1(i)/2.0d0
        wdt4(i) = wdt1(i)/4.0d0
        wdt8(i) = wdt1(i)/8.0d0
      end do
C
C     *****parameters for yoshida suzuki integration
C
      e2 = 1.d0/6.d0
      e4 = e2/20.d0
      e6 = e4/42.d0
      e8 = e6/72.d0
      gnkt = tstart * dble(ndof) * gaskinkc
      gn1kt = (ndof + 1)*tstart*gaskinkc
      gkt  = tstart * gaskinkc
      ondf = 1.d0 + (1.d0/dble(nat))
C
      do iat = 1,nat
        do kk = 1,3
          fnow(kk,iat) = f(kk,iat)
        end do
      end do
C
      return
      end
      SUBROUTINE nbenergy()
C
      implicit none
C
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer iat,ichain,itype,neightot,k,jat,jtype,jchain,ipiece
      integer itypee,ifunct,innb,nfunct,ncall,kk
      real*8 uninter,unintra,deltainv,entot
      real*8 xij1,xij2,xij3,z,zpiece,zpiece2,zpiece3,zpiece4,zpiece5
      real*8 en(3),enfintra(3,maxnnb),enfinter(3,maxnnb),enftot(3)
C
      data nfunct /3/
      data ncall /0/
      ncall = ncall + 1
C
      el2 = box/2.0d0
      if (ncall .eq. 1)then
      deltainv = 1.0d0/deltaspline
      entot = 0.0d0
      unintra = 0.0d0
      uninter = 0.0d0
      do ifunct = 1,nfunct
        en(ifunct) = 0.0d0
        do innb = 1,maxnnb
        enfintra(ifunct,innb) = 0.0d0
        enfinter(ifunct,innb) = 0.0d0
        end do
      end do
      end if
C
C     *****begin calculation of non-bonded and electrostatic energies
C          and forces
C
      do iat = 1, nat - 1
        ichain = chain(iat)
        itype  = atomtype(iat)
        neightot = listm(iat)
        do k =  1, neightot
          jat =  list(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if(abs(xij1) .gt. el2) xij1 = xij1 - dsign(box,xij1)
          if(abs(xij2) .gt. el2) xij2 = xij2 - dsign(box,xij2)
          if(abs(xij3) .gt. el2) xij3 = xij3 - dsign(box,xij3)
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
C
          if (z .gt. ro2) goto 332
C
          jtype = atomtype(jat)
          jchain = chain(jat)
          itypee = typee(itype,jtype)
C
          if (z .le. 0.5d0) then
            write(6,'("close approach*",F10.2,4I6)')
     $       ,z,itype,jtype,iat,jat
D            write(6,'(3F10.2)') (x(kk,iat),kk=1,3)
D            write(6,'(3F10.2)') (x(kk,jat),kk=1,3)
            z = 0.5d0
          end if
C
          ipiece = int((z - zs(0))*deltainv)+ 1
C
C     *****determine position within piece of spline
C
          zpiece = z - zs(ipiece-1)
C
C     *****determine powers for spline calculations
C
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
C
          do ifunct = 1,nfunct
            en(ifunct) = 
     +           coeff1(ifunct,ipiece,itypee)
     +         + coeff2(ifunct,ipiece,itypee)*zpiece
     +         + coeff3(ifunct,ipiece,itypee)*zpiece2
     +         + coeff4(ifunct,ipiece,itypee)*zpiece3
     +         + coeff5(ifunct,ipiece,itypee)*zpiece4
     +         + coeff6(ifunct,ipiece,itypee)*zpiece5
          end do
C
        en(1) = electrostatic(itypee,1)/(dsqrt(z))
C
C
      do ifunct = 1,nfunct
        if (ichain .eq. jchain) then
          enfintra(ifunct,itypee) = enfintra(ifunct,itypee)+en(ifunct)
          unintra = unintra + en(ifunct)
         else
          enfinter(ifunct,itypee) = enfinter(ifunct,itypee)+en(ifunct)
          uninter = uninter + en(ifunct)
        end if
        entot = entot + en(ifunct)
      end do
 332  continue
      end do
      end do
C
      write(59,*)
      write(59,*)'Average nonbonded      energy = ',entot/ncall
      write(59,*)'Average intramolecular energy = ',unintra/ncall
      write(59,*)'Average intermolecular energy = ',uninter/ncall
      write(59,*)
      write(59,*)'*****************INTRA MOLECULAR SPLIT************'
      write(59,*)' Interaction      Electrostatic     Polarization   '
     +   ,'   LJ/Exp-6    '
C
      do ifunct = 1,3
        enftot(ifunct) = 0.0d0
      end do
C
      do itype = 1,maxcharges
        do jtype = itype,maxcharges
          innb = typee(itype,jtype)
        write(59,259)atom_labels(itype),atom_labels(jtype),
     +         (enfintra(ifunct,innb)/ncall,ifunct=1,3)
        end do
      end do
      do ifunct = 1,3
        do innb = 1,maxnnb
          enftot(ifunct) = enftot(ifunct) + enfintra(ifunct,innb)/ncall
        end do
      end do
        write(59,*)clineo
        write(59,260)(enftot(ifunct),ifunct=1,3)
        write(59,*)clineo
      write(59,*)
      write(59,*)'*****************INTER MOLECULAR SPLIT************'
      write(59,*)' Interaction      Electrostatic     Polarization   '
     +   ,'   LJ/Exp-6    '
C
      do ifunct = 1,3
        enftot(ifunct) = 0.0d0
      end do
C
      do itype = 1,maxcharges
        do jtype = itype,maxcharges
          innb = typee(itype,jtype)
        write(59,259)atom_labels(itype),atom_labels(jtype),
     +         (enfinter(ifunct,innb)/ncall,ifunct=1,3)
        end do
      end do
      do ifunct = 1,3
        do innb = 1,maxnnb
          enftot(ifunct) = enftot(ifunct) + enfinter(ifunct,innb)/ncall
        end do
      end do
        write(59,*)clineo
        write(59,260)(enftot(ifunct),ifunct=1,3)
        write(59,*)clineo
      close(59)
 259  format(a5,'_',a5,5x,3f15.5)
 260  format('  Totals   ',5x,3f15.5)
      return
      end
C
      SUBROUTINE nhcint()
C     
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     local variables
C
      integer i,j,inos,iat,kk
      real*8 ekin2,scale,aa
C
      CALL getkin()
C
      ekin2 = 2.d0 * kinen
      scale = 1.d0
      glogs(1) = (ekin2 - gnkt)/qtmass(1)
      do i = 1,nhcstep1
        do j = 1,nhcstep2
          vlogs(nnos) = vlogs(nnos)+glogs(nnos)*wdt4(j)
          do inos = 1, nnos-1
            aa = dexp(-wdt8(j)*vlogs(nnos+1-inos)) 
            vlogs(nnos-inos) = vlogs(nnos-inos)*aa*aa
     +                        + wdt4(j)*glogs(nnos-inos)*aa
          end do
C
          aa = dexp(-wdt2(j)*vlogs(1))
          scale = scale*aa
          glogs(1) = (scale*scale*ekin2 - gnkt)/qtmass(1)
 
          do inos = 1,nnos
            xlogs(inos) = xlogs(inos) + vlogs(inos)* wdt2(j)
          end do
C
          do inos = 1, nnos-1
            aa = dexp(-wdt8(j)*vlogs(inos+1))
            vlogs(inos) = vlogs(inos)*aa*aa
     +                   + wdt4(j)*glogs(inos)*aa
            glogs(inos+1) = (qtmass(inos)*vlogs(inos)*vlogs(inos)
     +                     - gkt) / qtmass(inos+1)
          end do
          vlogs(nnos) = vlogs(nnos) + glogs(nnos)*wdt4(j)
        end do
      end do
C
      do iat = 1, nat
        do kk = 1,3
          v(kk,iat) = v(kk,iat) * scale
        end do
      end do
C
      return
      end
C
      subroutine nptint()
C     
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C
      integer inos,iat,kk,i,j
      real*8 aa,cons,scale,ekin2,box3
C
      cons = 1.4582454d-5
C
      CALL getkin()
C
      ekin2 = kinen*2.d0
      scale = 1.d0
      box3 = box*box*box
      glogs(1)=(ekin2+wtmass(1)*vlogv*vlogv-gn1kt)/qtmass(1)
      glogv = (ondf*ekin2+3.d0*(pintr-pfix)*box3*cons)/wtmass(1)
C
C     *****start yoshida/suzuki multiple step method 
C
      do i = 1,nhcstep1
        do j = 1,nhcstep2
          vlogs(nnos) = vlogs(nnos)+glogs(nnos)*wdt4(j)
          do inos = 1, nnos-1
            aa = dexp(-wdt8(j)*vlogs(nnos+1-inos)) 
            vlogs(nnos-inos) = vlogs(nnos-inos)*aa*aa
     +                       + wdt4(j)*glogs(nnos-inos)*aa
          end do
C
          aa = dexp(-wdt8(j)*vlogs(1))
          vlogv  = vlogv*aa*aa + wdt4(j)*glogv*aa
          aa = exp(-wdt2(j)*(vlogs(1)+ondf*vlogv))
          scale = scale*aa
          ekin2 = ekin2 * aa * aa
          glogv = (ondf*ekin2+3.d0*(pintr-pfix)*cons*box3)/wtmass(1)
          do inos = 1,nnos
            xlogs(inos) = xlogs(inos) + vlogs(inos)* wdt2(j)
          end do
C 
          aa = dexp(-wdt8(j)*vlogs(1))
          vlogv = vlogv*aa*aa + wdt4(j)*glogv*aa
          glogs(1) = (ekin2 + wtmass(1)*vlogv*vlogv - gn1kt)/qtmass(1)
          do  inos = 1, nnos-1
            aa =dexp(-wdt8(j)*vlogs(inos+1))
            vlogs(inos) = vlogs(inos)*aa*aa
     +                  + wdt4(j)*glogs(inos)*aa
            glogs(inos+1) = (qtmass(inos)*vlogs(inos)*vlogs(inos)
     +                    - gkt) / qtmass(inos+1)
          end do
          vlogs(nnos) = vlogs(nnos) + glogs(nnos)*wdt4(j)
        end do
      end do
C
      do iat = 1, nat
        do kk = 1,3
          v(kk,iat) = v(kk,iat) * scale
        end do
      end do
C
      return
      end
      SUBROUTINE output1()
C   
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer inos,nfile1,nfile2,kk,iat,itype,jj,istart,n
      real*4 timeo,timev,timef,qi
      real*4 short(3,maxat),shstress(6),shvel(3,maxat)
      real*4 timed,times,timex
      real*4 DipoleChain(3,maxnch)
C
      common /dipChain/ DipoleChain  ! dipoles moments of chains 
C
      data nfile1 /0/
      data nfile2 /0/
      timeo = real(koutput)*stepout
      timex = real(kcoords)*stepout
      timev = real(kvelocs)*stepout
      times = real(kstress)*stepout
      timed = real(kboxdip)*stepout ! time dipole moment
      timef = dble(nsteps)*stepout
C
C     *****total dipole moment and charge flux (file = fort.76)
C
      if (lboxdip .and. mod(kount,kboxdip) .eq. 0)then
C
        open (76,file='fort.76',
     +     form='unformatted',status='old',access='append')
        write (76)timed
        write (76)DipoleTot,DipolePol,DipoleChain
        close (76)
      end if
C
C     *****stress tensor(file = fort.78)
C
      if (lstress .and. mod(kount,kstress) .eq. 0)then
        shstress(1) = stress(1,1)
        shstress(2) = stress(2,2)
        shstress(3) = stress(3,3)
        shstress(4) = 0.5*(stress(1,2)+stress(2,1))
        shstress(5) = 0.5*(stress(1,3)+stress(3,1))
        shstress(6) = 0.5*(stress(2,3)+stress(3,2))
C
        open (78,file='fort.78',
     +     form='unformatted',status='old',access='append')
C       open (78,file='fort.78',
C    +     form='unformatted',status='old',iostat=ios) ! RS6000(ibm)
        write (78)times
        write (78)shstress
        close (78)
      end if
C
C     *****coordinates(file = fort.77)
C
      if (lcoords .and. mod(kount,kcoords) .eq. 0) then
C
        nfile1 = nfile1 + 1
        do iat = 1,maxat
          do kk =1,3
            short(kk,iat) = x(kk,iat)
          end do
        end do
        open (77,file='fort.77',
     +   form='unformatted',status='old',access='append')
C        open (77,file='fort.77',
C    +   form='unformatted',status='old',iostat=ios) ! RS6000(ibm)
        write (77) nat,timex,nfile1,box
        write (77) short
        close (77)
       end if
C
C     *****velocities(file = fort.79)
C
      if (lvelocs .and. mod(kount,kvelocs) .eq. 0) then
C
        do iat = 1,maxat
          do kk =1,3
            shvel(kk,iat) = v(kk,iat)
          end do
        end do
        open (79,file='fort.79',
     +   form='unformatted',status='old',access='append')
C       open (79,file='fort.79',
C    +   form='unformatted',status='old',iostat=ios) ! RS6000(ibm)
        write (79) nat,timev
        write (79) shvel
        close (79)
      end if
C
C     ***** output coordinates and velocities periodically(file = fort.66)
C
      if (mod(kount,koutput) .eq. 0)then
        nfile2 = nfile2 + 1
        open (66,file="coords.out",status='unknown',err=661)
        write(66,999)cline
        write(66,199)
        write(66,999)cline
        write(66,998)
        write(66,200) timeo*nfile2/1000.0d0,
     +              nint(timeo*nfile2*100.0d0/timef)
        write(66,998)
        write(66,198)
        write(66,197)
        write(66,999)cline
        write(66,998)
        do iat = 1,nat
          itype = atomtype(iat)
          write(66,220)(x(kk,iat),kk = 1,3),
     +      atom_labels(itype)(1:3),atom_labels(itype)(4:5)
          write(66,221) (v(kk,iat),kk = 1,3)
        end do
        write(66,999)cline
        write(66,201)
        write(66,*)box
        write(66,202)
        write(66,*)(xlogs(inos),inos = 1,nnos)
        write(66,203)
        write(66,*)(vlogs(inos),inos = 1,nnos)
        write(66,204)
        write(66,*)(glogs(inos),inos = 1,nnos)
        write(66,205)
        write(66,*)xlogv
        write(66,206)
        write(66,*)vlogv
        write(66,207)
        write(66,*)glogv
        close (66)
661     continue
      end if
 998  format('*')
 999  format(a72)
 199  format('*           ATOM COORDINATES (Angstroms) AND ',
     +      'VELOCITIES (m/sec)')
 198  format('*        X              Y              Z  Atomtype',
     +       ' Chargetype')
 197  format('*        Vx             Vy             Vz')
 200  format('*     Time = ',f15.3,' picoseconds (',i3,'% of ',
     +         'the simulation run)')
 201  format('*   Box (Angstroms) ')
 202  format('*   Xlogs (Thermostat position)')
 203  format('*   Vlogs (Thermostat velocity)')
 204  format('*   Glogs (Thermostat force)')
 205  format('*   Xlogv (Barostat position)')
 206  format('*   Vlogv (Barostat velocity)')
 207  format('*   Glogv (Barostat force)')
 220  format (3f15.6,3x,a3,3x,a2)
 221  format (1x,4f15.6)
C
      return
      end
      SUBROUTINE output2()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer iprop,nens
      character*20 ens(3),ensemble
C
      data ens /'Microcanonical','Canonical','Isobaric-isothermal'/
C
      open (70,file='fort.70',status='unknown')
C
C     *****compute overall sd in properties
C
      do iprop = 1,maxprop
        if ((simsqpr(iprop) - simpr(iprop)**2) .lt. 1.0d-12)then
          stdev(iprop) = 0.0d0
         else
          stdev(iprop) = dsqrt(simsqpr(iprop) - simpr(iprop)**2)
        end if
      end do
C
      if (nve)ensemble = ens(1)
      if (nvt)ensemble = ens(2)
      if (npt)ensemble = ens(3)
      if (.not. lboxdip)nboxdip = 0
      if (.not. lcoords)ncoords = 0
      if (.not. lvelocs)nvelocs = 0
      if (.not. lstress)nstress = 0
C
C     *****write statistics to file 70
C
      write(70,*)clineo
      write(70,*)'                      Simulation parameters'
      write(70,*)clineo
      write(70,*)
      write(70,*)'Ensemble of simulation    = ',ensemble
      write(70,*)'Cut-off Radius            = ',ro
      write(70,*)'Nose-hoover chains        = ',nnos
      write(70,*)'Q-frequency               = ',qfreq
      write(70,*)'W-frequency               = ',wfreq
      if (tbpol)then
        write(70,*)'Cut-off Radius (pol)    = ',rop
        write(70,*)'Scale Radius (pol)      = ',rscalep
        write(70,*)'Epsilon (pol)           = ',epsilonp
        write(70,*)'Repsilon (pol)          = ',repsilonp
      end if
      write(70,*)'Average Calculation       = ',nave
      write(70,*)'Virial  Calculation       = ',nvir
      write(70,*)'Nbenergy Calculation      = ',nnben
      write(70,*)'Final coords output       = ',noutput
      write(70,*)'Boxdipole output          = ',nboxdip
      write(70,*)'Coordinates output        = ',ncoords
      write(70,*)'Velocities output         = ',nvelocs
      write(70,*)'Stress tensor output      = ',nstress
      write(70,*)'Interchain interactions   = ',inter
      write(70,*)'1-4interaction exclusion  = ',ex14
      write(70,*)'Temperature fixing        = ',fixtemp
      write(70,*)'Bond length constraints   = ',constrain
      write(70,*)'Aromatic H constraints    = ',constrain
      write(70,*)'Tolerance in shake        = ',tol
      write(70,*)'Total number of chains    = ',nch
      write(70,*)'Total number of ions      = ',nions
      write(70,*)'Number of timesteps       = ',nsteps
      write(70,*)'Length of timestep(fs)    = ',stepout
      write(70,*)'Total simulation time(ns) = ',stepout*nsteps*1.0d-6
      write(70,*)'Electrostatics : Regular Ewald'
      write(70,*)'  Alpha = ',alphai
      write(70,*)'  Klimit = ',klimit
      write(70,*)'Integrator : ',
     + 'Multiple timestep integrator(vibrational+double cutoff)'
      write(70,*)'  Multimed = ',multimed
      write(70,*)'  Multibig = ',multibig
      write(70,*)'  Rshort = ',ros
      write(70,*)'Polarization : ',
     + 'Regular Ewald'
      write(70,*)'  Tolerance(pol) = ',tolpol
      write(70,*)'  Tapering = ',tapering
      write(70,*)'  Rscale(RF) = ',rtaper
      write(70,*)'  Epsilon(RF) = ',epsrf
      write(70,*)'Source code : Lucretius 8.0'
      write(70,*)'  Generated by Oleg Borodin '
      write(70,*)'  Generated on Mon Dec 10 09:54:28 MST 2001 '
      write(70,*)
      write(70,*)clineo
      write(70,*)'               Averages over simulation length'
      write(70,*)clineo
      write(70,*)
      write(70,*)'Ave kin en  of simulation = ',simpr(6)
      write(70,*)'Std dev in  K.E.          = ',stdev(6)
      write(70,*)'Ave tot en  of simulation = ',simpr(4)
      write(70,*)'Std dev in  T.E.          = ',stdev(4)
      write(70,*)'Ave enthalpy of sim.      = ',simpr(15)
      write(70,*)'Std dev in  Enthalpy      = ',stdev(15)
      write(70,*)'Rms pol bead displacement = ',dsqrt(spol/nat)
      write(70,*)'Temperature of simulation = ',simpr(1)
      write(70,*)'Ave  simulation  pressure = ',simpr(2)
      write(70,*)'Std simulation  pressure  = ',stdev(2)
      write(70,*)'Ave Nonbonded energy      = ',simpr(14)
      write(70,*)'Initial box               = ',boxini
      write(70,*)'Final box                 = ',box
      write(70,*)'Average box               = ',simpr(3)
      write(70,*)'Std dev in box            = ',stdev(3)
      write(70,*)'Std dev in Hamiltonian    = ',stdev(5)
      write(70,*)'        *****   Average stress tensor   ***** '
      write(70,255)simpr(16),simpr(17),simpr(18)
      write(70,255)simpr(19),simpr(20),simpr(21)
      write(70,255)simpr(22),simpr(23),simpr(24)
      write(70,*)
      write(70,*)clineo

C
 255  format (5x,3f15.4)
      return
      end
C
      SUBROUTINE parse_line(line,lstart,istart,iend)
C
      implicit none
C
C     *****shared variables
C
      integer lstart,istart,iend
      character*144 line
C
C     *****local variables
C
      character*1 value
      integer ipos
C
      ipos = lstart -1
      value = ' '
      do while (value .eq. ' ')
        ipos = ipos + 1
        value = line(ipos:ipos)
      end do
      istart = ipos
      do while (value .ne. ' ')
        ipos = ipos + 1
        value = line(ipos:ipos)
      end do
      iend = ipos - 1
C
      return
      end
C
      double precision FUNCTION pdforce(r)
C 
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      real*8 r
C
C     *****local variables
C
      real*8 derfc,delta,eofr
C
C     *****begin force evaluation
C
      pdforce = 0.0d0
      if (.not. tbpol)return
      if (r .gt. rscalep) then ! in scaled region
        delta = rop-rscalep
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pdforce = (2.d0/(eofr*r**5))*(-6.d0*(r-rscalep)   /delta**2
     +                      +6.d0*(r-rscalep)**2/delta**3)
     +          +(-10.d0/(eofr*r**6) 
     +         - 2.0d0*log(epsilonp)*repsilonp/(eofr*r**7)) 
     +            *(1.d0 - 3.d0*(r-rscalep)**2/delta**2  
     +                  + 2.d0*(r-rscalep)**3/delta**3)
      else if (r .gt. repsilonp) then
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pdforce = (-10.d0/(eofr*r**6) 
     +         - 2.0d0*log(epsilonp)*repsilonp/(eofr*r**7))
       else
         pdforce = (-10.d0/(r**6))
      end if
      return
      end
      double precision FUNCTION pforce(r)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      real*8 r
C
C     *****local variables
C
      real*8 derfc,delta,eofr
C
C     *****begin force evaluation
C
      pforce = 0.0d0
      if (.not. tbpol)return
      if (r .gt. rscalep) then ! in scaled region
        delta = rop-rscalep
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pforce = (2.d0/(eofr*r**5))*
     +              (1.d0-3.d0*(r-rscalep)**2/delta**2
     +                      +2.d0*(r-rscalep)**3/delta**3)
       else if (r .gt. repsilonp) then
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pforce = (2.0d0/(eofr*r**5))
       else
        pforce = (2.d0/(r**5))
      end if
      return
      end
      double precision FUNCTION pintforce(xa,flower,xb,fupper)
C
      implicit none
C
C     *****shared variables
C
      real*8 xa,flower,xb,fupper
      integer ifunct
C
C     *****local variables
C
      real*8 t(100,100)
      real*8 pforce,sum,odelx,relerror
      integer j,i,icol
C
      relerror = 1.0d-4
C
C     *****begin calculations
C
      t(1,1)=(xb-xa)/2.0d0*(flower+fupper)
      t(2,1)=t(1,1)/2.0d0+(xb-xa)*
     +          pforce((xa+xb)/2.0d0)/2.0d0
      t(2,2)=(4.0d0*t(2,1)-t(1,1))/3.0d0
C
      do j=3,100
        odelx=(xb-xa)/2.0d0**(j-2)
        sum=0.0d0
        do i=1,2**(j-2)
	    sum=sum+pforce(xa+(dble(i)-0.5d0)*odelx)
        end do
        t(j,1)=0.5d0*(t(j-1,1)+odelx*sum)
        do icol=2,j
          t(j,icol)=(4.0d0**(icol-1)*t(j,icol-1)-t(j-1,icol-1))/
     +	               (4.0d0**(icol-1)-1.0)
        end do
        if (abs((t(j,j)-t(j-1,j-1))/t(j,j)) .le. relerror) then
          pintforce = -t(j,j)
          goto 110
        end if
      end do
      write (6,*) 'integral did not converge'
110   continue
C
      return
      end
      SUBROUTINE read11()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer ch,iat,ibond,itbond,j,ibend,ib1,ib2,nchains
      integer imult,itemp,ich,icheck1,icheck2,itort,ideform,ib
      integer iba,itot,itest,itbend,ittort,itdefs,ibase
      integer nbondsch(maxnch),nbendsch(maxnch),ntortsch(maxnch),
     +                ndeformsch(maxnch),natch(maxnch),nbondex(maxnch)
      character*144 line
C
C
C     *****read in data
C
      itbond = 0
      itbend = 0
      ittort = 0
      itdefs = 0
      open(11,file="connectivity.dat",status="old")
      CALL read_a_line(11,line)
      read (line,*) nchains, nions
      nch = nchains + nions
      nat = 0
      nbonds = 0
      nbends = 0
      ntorts = 0
      ndeforms = 0
      do ich = 1, nch
        CALL read_a_line(11,line)
        read (line,*) natch(ich),nbondsch(ich),
     +     nbendsch(ich),ntortsch(ich),ndeformsch(ich)
        nat = nat + natch(ich)
        nbondex(ich) = nbonds
        nbonds = nbonds + nbondsch(ich)
        nbends = nbends + nbendsch(ich)
        ntorts = ntorts + ntortsch(ich)
        ndeforms = ndeforms + ndeformsch(ich)
      end do
      write (6,*) 'nat = ',nat
      write (6,*) 'nbonds = ',nbonds
      write (6,*) 'nbends = ',nbends
      write (6,*) 'ntorts = ',ntorts
      write (6,*) 'ndeforms = ',ndeforms
      write (6,*) 'nions = ',nions
      write (6,*) 'reading bonds'
      do ich = 1,nch
        if (nbondsch(ich) .eq. 0)then
          CALL read_a_line(11,line)
          read (line,*) iat
          chain(iat) = ich
        end if
        do ibond = 1,nbondsch(ich)
          CALL read_a_line(11,line)
          itbond = itbond + 1
          read (line,*) (bonds(j,itbond), j=1,3)
        end do
      end do
      write (6,*) 'reading bends'
      do ich = 1,nch
        ibase = nbondex(ich)
        do ibend = 1,nbendsch(ich)
          CALL read_a_line(11,line)
          itbend = itbend + 1
          read (line,*) (bends(j,itbend), j=1,4)
C
C     *****determine which bonds make up each bend
C
          ib1 = bends(1,itbend)
          ib2 = bends(2,itbend)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              bends(5,itbend) = imult*ibond
             goto 111
            end if
          end do
          write (6,*) 'bend:first bond not found'
111       continue
          ib1 = bends(2,itbend)
          ib2 = bends(3,itbend)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              bends(6,itbend) = imult*ibond
              goto 121
            end if
          end do
          write (6,*) 'bend:second bond not found'
121       continue
        end do
      end do
      write (6,*) 'reading torsions'
      do ich = 1,nch
        ibase = nbondex(ich)
        do itort = 1,ntortsch(ich)
          CALL read_a_line(11,line)
          ittort = ittort + 1
          read (line,*) (torts(j,ittort), j=1,5)
C
C     *****determine which bonds make up each torsion
C
          ib1 = torts(1,ittort)
          ib2 = torts(2,ittort)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              torts(6,ittort) = imult*ibond
              goto 211
            end if
          end do
          write (6,*) 'torsion:first bond not found'
211       continue
          ib1 = torts(2,ittort)
          ib2 = torts(3,ittort)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              torts(7,ittort) = imult*ibond
              goto 221
            end if
          end do
          write (6,*) 'torsion:second bond not found'
221       continue
          ib1 = torts(3,ittort)
          ib2 = torts(4,ittort)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              torts(8,ittort) = imult*ibond
              goto 231
            end if
          end do
          write (6,*) 'torsion:third bond not found'
231       continue
        end do
      end do
      write (6,*) 'reading deformations'
      do ich = 1,nch
        ibase = nbondex(ich)
        do ideform = 1,ndeformsch(ich)
          CALL read_a_line(11,line)
          itdefs = itdefs + 1
          read (line,*) (deforms(j,itdefs), j=1,5),idpar(itdefs)
C
C     *****determine which bonds make up each deformation
C
          ib1 = deforms(1,itdefs)
          ib2 = deforms(2,itdefs)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              deforms(6,itdefs) = imult*ibond
              goto 411
            end if
          end do
          write (6,*) 'deform:first bond not found'
411       continue
          ib1 = deforms(2,itdefs)
          ib2 = deforms(3,itdefs)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              deforms(7,itdefs) = imult*ibond
              goto 421
            end if
          end do
          write (6,*) 'deform:second bond not found'
421       continue
          ib1 = deforms(2,itdefs)
          ib2 = deforms(4,itdefs)
          imult = 1
          if (ib2 .lt. ib1) then
            imult = -1
            itemp = ib1
            ib1 = ib2
            ib2 = itemp
          end if
          do ib = 1,nbondsch(ich)
            ibond = ibase + ib
            icheck1 = bonds(1,ibond)
            icheck2 = bonds(2,ibond)
            if (icheck1 .eq. ib1 .and. 
     +          icheck2 .eq. ib2) then
              deforms(8,itdefs) = imult*ibond
              goto 431
            end if
          end do
          write (6,*) 'deform:third bond not found'
431       continue
        end do
      end do
C
C     *****determine chain of each atom
C
      itot=0
      do ich = 1, nch
        do ib = 1,nbondsch(ich)
          itot = itot + 1
          do iba = 1,2
            itest = bonds(iba,itot)
            chain(itest) = ich
          end do
        end do
      end do
C     
      close(11)
      return 
      end 
C
      SUBROUTINE read12()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer number,intert,itype,jtype,nq,iinter,jinter,icheck,jcheck
      integer iin,inontype,jnontype,inb,irow,icol,j,innb,ibond,ibend
      integer itort,ideform,ii,n2
      real*8 a,b,c,d,remass,f1,f2,f3
      integer non_of_charge(maxcharges)
      character*2 int
      character*3 check,label1,label2
      character*21 label
      character*6 nnb_label
      character*144 line
C
      integer i
C
C
      ncharge = 0
      lpolarizable=.false.
      open(12,file="ff.dat",status="old") 
      CALL read_a_line(12,line)
      read (line,*) number
C
C     *****assign typen initially           
C
      intert = 0
      do itype = 1,number
        do jtype = itype,number
          intert = intert + 1
          typen(itype,jtype) = intert
          typen(jtype,itype) = intert
         end do
      end do
      number_nnb = intert
      CALL read_a_line(12,line)
      write(6,*) "Nonboned self-terms"
      do itype = 1,number
        nq = 0
        read (line,'(a3)') non_labels(itype)                 
        intert =  typen(itype,itype)
        read (line(4:144),*) a,b,c,d,remass
        write(6,'(a3,x,4F15.2,F10.4)')non_labels(itype),a,b,c,d,remass
        nonbonded(1,intert) = a
        nonbonded(2,intert) = b
        nonbonded(3,intert) = c
        electrostatic(intert,2) = -d*2.0d0
        atmasses(itype) = remass
C      
C       *******look for charges for this type
C
        CALL read_a_line(12,line)
        if (line(1:1) .ne. ' ') then
          nq = nq + 1
          write(int,'(i2)') nq
          ncharge = ncharge +  1
          q(ncharge) = 0.0d0
          pol(ncharge) = 0.0d0
          atom_labels(ncharge) = non_labels(itype)//int
          write (6,*) 'no charge and polarizability for atom ',
     +                     atom_labels(ncharge),
     +                       ' was found:  a zero charge is assumed'
          goto 134
         else
          do while (line(1:1) .eq. ' ')
            nq = nq + 1
            write(int,'(i2)') nq
            ncharge = ncharge + 1
C
C      added in v1.0
C
            lonepairtype(ncharge)=.false.
            if (non_labels(itype).eq."Lp".or.
     $           non_labels(itype).eq."LP") then
              lonepairtype(ncharge)=.true.       ! added by Oleg
            endif
C
C     end of addition
C
            atom_labels(ncharge) = non_labels(itype)//int
            read (line(2:144),*) q(ncharge),pol(ncharge)
            write(6,'("Q=",F12.4," pol.=",F12.4)')
     $              q(ncharge),pol(ncharge)
            if (pol(ncharge).gt.1.0e-4) lpolarizable=.true.
            non_of_charge(ncharge) = itype                   
            CALL read_a_line(12,line)
          end do
        end if
134     continue  
      end do
C
C     *****assign default for off diagonal
C          a = gm   b = gm   c = gm
C
      intert = 0
      do itype = 1,number
        iinter =  typen(itype,itype)
        do jtype = itype,number
          jinter =  typen(jtype,jtype)
          intert = intert + 1
          a = sqrt(nonbonded(1,iinter)*nonbonded(1,jinter))
          b = 0.5*(nonbonded(2,iinter)+nonbonded(2,jinter))
          c = sqrt(nonbonded(3,iinter)*nonbonded(3,jinter))
          nonbonded(1,intert) = a
          nonbonded(2,intert) = b
          nonbonded(3,intert) = c
        end do
      end do
C
C     *****read non-default nonbonded interactions
C
      do while (line(1:1) .ne. '!')
        read(line,'(a3)') label1
        do icheck = 1,number
          if (label1 .eq. non_labels(icheck)) then
            itype = icheck
            goto 222
          end if
        end do
222     read (line(5:7),'(a3)') label2
        do jcheck = itype + 1,number
          if (label2 .eq. non_labels(jcheck)) then
            jtype = jcheck
            goto 223
          end if
        end do
        stop
223     intert =  typen(itype,jtype)
        read (line(8:144),*) a,b,c,d
        write(6,'(a3,x,a3,4F15.2,F10.4)')
     $     label1,label2,a,b,c,d
        nonbonded(1,intert) = a
        nonbonded(2,intert) = b
        nonbonded(3,intert) = c
        electrostatic(intert,2) = -d*2.0d0
        CALL read_a_line(12,line)
      end do
C
      intert = 0
      do itype = 1,ncharge
        inontype = non_of_charge(itype)                   
        do jtype = itype,ncharge
          jnontype = non_of_charge(jtype)                   
          intert = intert + 1
          typee(itype,jtype) = intert
          typee(jtype,itype) = intert
          inb = typen(inontype,jnontype)
          map(intert) = inb
         end do
      end do
      do irow = 1,ncharge
	do icol = 1,ncharge
	  itype = typee(irow,icol)
	  electrostatic(itype,1) = 332.08d0*q(irow)*q(icol)
	end do
      end do
C
C     *****valence force field
C
      CALL read_a_line(12,line)
      read (line,*) nbondt
      write(6,*) "Bonds"
      do ibond = 1,nbondt
        CALL read_a_line(12,line)
        label = line(1:11)
        read (line(12:144),*) f1,f2,f3
        write(6,'(a10,x,3F12.3)') label,f1,f2,f3
        stretch(1,ibond) = f1
        stretch(2,ibond) = f2
        stretch(3,ibond) = f3
      end do
      CALL read_a_line(12,line)
      read (line,*) nbendt
      write(6,*) "Bends"
      do ibend = 1,nbendt
        CALL read_a_line(12,line)
        label = line(1:15)
        read (line(16:144),*) f1,f2
        write(6,'(a15,x,2F12.3)') label,f1,f2
        bend(1,ibend) = f1
        bend(2,ibend) = f2
      end do
      CALL read_a_line(12,line)
      read (line,*) ntortt
      write(6,*) "Torsions"
      do itort = 1,ntortt
        CALL read_a_line(12,line)
        label = line(1:20)
        read (line(21:144),*)n2,(twist(ii,itort),ii=0,n2-1)
        write(6,'(a20,x,I5,8F9.5)')label,n2,(twist(ii,itort),ii=0,n2-1)
        nprms(itort) = n2
      end do
      CALL read_a_line(12,line)
      read (line,*) ndeformt
      write(6,*) "Deformations"
      do ideform = 1,ndeformt
        CALL read_a_line(12,line)
        label = line(1:20)
        read (line(21:144),*) f1
        write(6,'(a20,x,F10.3)') label,f1
        deform(ideform) = f1
      end do
 114  format (a10,10f5.1)
      CALL read_a_line(12,line)
      read (line,*) ndummytypes
      write(6,*) "Dummy atoms"
      do i=1,maxcharges
        typedummy(i)=0
      end do 
      do itype=1,ndummytypes
        CALL read_a_line(12,line)
        label = line(1:20)
        read(line(21:144),*) i,adummy(itype),bdummy(itype),
     $                     cdummy(itype),LpGeomType(itype),
     $  LpBendAtomType(1,itype),LpBendAtomType(2,itype),
     $  LpBendAtomType(3,itype) ! added in v2.13
        write(6,'(a20,I3,3F12.4,4I3)')label,i,adummy(itype),
     $      bdummy(itype),cdummy(itype),LpGeomType(itype),
     $  LpBendAtomType(1,itype),LpBendAtomType(2,itype),
     $  LpBendAtomType(3,itype) ! added in v2.13
        typedummy(i)=itype
      end do
C
      write(6,*) "finished reading force field"
C
      close(12)
      return
      end
C
      SUBROUTINE read25()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      character*12 integ,elect
      character*19 polter
      character*144 line
C
      open(25,file="mdrun.params",status="old")
      CALL read_a_line(25,line)
      read (line,*) nve, nvt, npt
      CALL read_a_line(25,line)
      read (line,*) rread,driftmax,delt,nsteps
      CALL read_a_line(25,line)
      read (line,*) tstart,pfix,nnos,qfreq,wfreq
      CALL read_a_line(25,line)
      read (line,*) tbpol,rop,rscalep,epsilonp,repsilonp
      CALL read_a_line(25,line)
      read (line,*) ninit,nave,nvir,nnben,noutput
      CALL read_a_line(25,line)
      read (line,*) lboxdip,lcoords,lstress,lvelocs
      CALL read_a_line(25,line)
      read (line,*) nboxdip,ncoords,nstress,nvelocs
      CALL read_a_line(25,line)
      read (line,*) chvol, boxnew, printout
      CALL read_a_line(25,line)
      read (line,*) inter, ex14,fixtemp
      CALL read_a_line(25,line)
      read (line,*) lredonefour,redfactor,lredQ_mu14,redQmufactor
      write(6,*) "reduce all 1-4 except mu-mu",lredonefour
      write(6,*) "reduce Q-mu 1-4 ",lredQ_mu14," by ",redQmufactor
      CALL read_a_line(25,line)
      read (line,*) constrain, tol
      CALL read_a_line(25,line)
      read (line,'(a12)')elect 
      read (line(13:72),*) alphai,klimit
      CALL read_a_line(25,line)
      read (line,'(a12)')integ 
      read (line(13:72),*) multimed,multibig,ros
      CALL read_a_line(25,line)
      read (line,'(a19)')polter 
      read (line(23:72),*) tolpol,tapering,rtaper,epsrf
      write (6,*)"tolpol,tapering,rtaper,epsrf",tolpol,tapering,rtaper,
     $    epsrf
      close(25)
      return       
      end          
C
      SUBROUTINE read26()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,iat,lstart,istart,iend,icharge,itype,imap,inos
      character*144 line
      character*5 check
      character*2 int
      real*8 shift(3)
C
C
C     *****begin read
C
      shift(1) = 10000.d0
      shift(2) = 10000.d0
      shift(3) = 10000.d0
C
C
      do iat = 1, nat
        ldummy(iat)=.false.
      end do
C
      open(26,file="coords.inp",status="old")
      ndummy=0
      do iat = 1, nat
        CALL read_a_line(26,line)
        lstart = 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:iend),*) x(1,iat)
        lstart = iend + 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:iend),*) x(2,iat)
        lstart = iend + 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:iend),*) x(3,iat)
        do kk = 1,3
         if (x(kk,iat) .lt. shift(kk)) shift(kk) = x(kk,iat)
        end do
C
        lstart = iend + 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:istart+2),'(a3)') check 
C
        lstart = iend + 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:iend),*) icharge
        write (int,'(i2)') icharge
        check = check(1:3)//int
C
C       *****assign atom type
C
        do itype = 1,maxcharges
          if (check .eq. atom_labels(itype)) then
            atomtype(iat) = itype
            if (typedummy(itype).gt.0) then 
              ldummy(iat)=.true.
              ndummy=ndummy+1
            end if
            imap = map(itype)
            mass(iat) = atmasses(imap)
            goto 12
          end if
        end do
        write (6,*) 'atomtype not found'
        stop
12      continue
C
        CALL read_a_line(26,line)
        read (line,*) (v(kk,iat),kk=1,3)
      end do
C
C     ***** read extended lagrangian variables
C
      CALL read_a_line(26,line)
      read(line,*)box
      write(6,*) "initial box=",box
      CALL read_a_line(26,line)
      read(line,*)(xlogs(inos),inos = 1,nnos)
      CALL read_a_line(26,line)
      read(line,*)(vlogs(inos),inos = 1,nnos)
      CALL read_a_line(26,line)
      read(line,*)(glogs(inos),inos = 1,nnos)
      CALL read_a_line(26,line)
      read(line,*)xlogv
      CALL read_a_line(26,line)
      read(line,*)vlogv
      CALL read_a_line(26,line)
      read(line,*)glogv
C
C     *******shift positions of necessary
C
      if (shift(1) .lt. 0.d0 .or. shift(2) .lt. 0.d0 .or.
     +      shift(3) .lt. 0.d0) then
        write (6,*) 'shifting positions'
        do iat = 1,nat
          do kk = 1,3
            x(kk,iat) = x(kk,iat) - shift(kk) + 1.d-8
          end do
        end do
      end if
C
      close(26)
      return
      end
      SUBROUTINE read_a_line(iunit,line)
C
      implicit none
C
C     *****shared variables
C
      integer iunit
      character*144 line
C
1     read(iunit,'(a144)',end=100) line
      if (line(1:1) .eq. '*') goto 1
      return
100   continue
      write (6,*) 'end of file ',iunit,' was reached'
      stop
      end
C
      SUBROUTINE reciprocal()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,iat,kx,ky,kz,kmag2,klimit2,itype
      real*8 front,twopifact,fouralpha2inv,factor,rkmag2,btens,akvec
      real*8 cossum,sinsum,ci,dotik,sfh,prefact,f1,front2,elfact
      real*8 recip,tmp
      real*8 tvirtmp(3,3)
C
      real*8 cos_x(maxat,0:kmax)
      real*8 cos_y(maxat,0:kmax)
      real*8 cos_z(maxat,0:kmax)
      real*8 cos_xy(maxat)
      real*8 cos_xyz(maxat)
      real*8 sin_x(maxat,0:kmax)
      real*8 sin_y(maxat,0:kmax)
      real*8 sin_z(maxat,0:kmax)
      real*8 sin_xy(maxat)
      real*8 sin_xyz(maxat)
C
      integer kymin,kzmin,kyabs,kzabs
C
      common /ewaldrecip/ cos_x,cos_y,cos_z,sin_x,sin_y,sin_z
C
      front = 2.0d0*pi/box
      front2 = front*front
      twopifact = 2.0d0*pi/(box*box*box)
      fouralpha2inv = 1.0d0/(4.0d0*alpha*alpha)
      elfact = 2.0d0*front*twopifact
      klimit2 = klimit*klimit
C
      recip = 0.0d0
      do iat = 1,nat
        do kk = 1,3
          fewald(kk,iat) = 0.0d0
          elf(kk,iat) = 0.0d0
        end do
      end do   
C
      do jj = 1,3
        do kk = 1,3
          tvirtmp(kk,jj) = 0.0d0
        end do
      end do
C
      do iat = 1,nat
        cos_x(iat,0)=1.0d0
        sin_x(iat,0)=0.d0
        cos_y(iat,0)=1.0d0
        sin_y(iat,0)=0.0d0
        cos_z(iat,0)=1.0d0
        sin_z(iat,0)=0.0d0
      end do
      do iat = 1,nat
        cos_x(iat,1)=dcos(x(1,iat)*front)
        sin_x(iat,1)=dsin(x(1,iat)*front)
        cos_y(iat,1)=dcos(x(2,iat)*front)
        sin_y(iat,1)=dsin(x(2,iat)*front)
        cos_z(iat,1)=dcos(x(3,iat)*front)
        sin_z(iat,1)=dsin(x(3,iat)*front)
      end do
       do kx = 2,klimit
          cos_x(1:nat,kx)=cos_x(1:nat,kx-1)*cos_x(1:nat,1)-
     $                    sin_x(1:nat,kx-1)*sin_x(1:nat,1)
          sin_x(1:nat,kx)=sin_x(1:nat,kx-1)*cos_x(1:nat,1)+
     $                    cos_x(1:nat,kx-1)*sin_x(1:nat,1)
       end do
       do ky = 2,klimit
          cos_y(1:nat,ky)=cos_y(1:nat,ky-1)*cos_y(1:nat,1)-
     $                    sin_y(1:nat,ky-1)*sin_y(1:nat,1)
          sin_y(1:nat,ky)=sin_y(1:nat,ky-1)*cos_y(1:nat,1)+
     $                    cos_y(1:nat,ky-1)*sin_y(1:nat,1)
       end do
       do kz = 2,klimit
          cos_z(1:nat,kz)=cos_z(1:nat,kz-1)*cos_z(1:nat,1)-
     $                    sin_z(1:nat,kz-1)*sin_z(1:nat,1)
          sin_z(1:nat,kz)=sin_z(1:nat,kz-1)*cos_z(1:nat,1)+
     $                    cos_z(1:nat,kz-1)*sin_z(1:nat,1)
       end do
      kymin=0
      kzmin=1
      factor = 2.0d0
      do kx = 0,klimit
        do ky = kymin,klimit
         if (ky.ge.0) then
           cos_xy(1:nat)=cos_x(1:nat,kx)*cos_y(1:nat,ky)-
     $                   sin_x(1:nat,kx)*sin_y(1:nat,ky)
           sin_xy(1:nat)=sin_x(1:nat,kx)*cos_y(1:nat,ky)+
     $                   cos_x(1:nat,kx)*sin_y(1:nat,ky)
         else    ! negative  ky
          kyabs=-ky
           cos_xy(1:nat)=cos_x(1:nat,kx)*cos_y(1:nat,kyabs)+
     $                   sin_x(1:nat,kx)*sin_y(1:nat,kyabs)
           sin_xy(1:nat)=sin_x(1:nat,kx)*cos_y(1:nat,kyabs)-
     $                   cos_x(1:nat,kx)*sin_y(1:nat,kyabs)
         endif
         do kz = kzmin,klimit
          kmag2 = kx*kx + ky*ky + kz*kz
          if (kmag2 .lt. klimit2 ) then
            rkmag2 = dble(kmag2)*front2
            btens = 2.0d0*front2*(1.0d0+rkmag2*fouralpha2inv)/rkmag2
            akvec = factor*dexp(-rkmag2*fouralpha2inv)/rkmag2
            cossum = 0.0d0
            sinsum = 0.0d0
            if (kz.ge.0) then  
             cos_xyz(1:nat)=cos_xy(1:nat)*cos_z(1:nat,kz)-
     $                      sin_xy(1:nat)*sin_z(1:nat,kz)
             sin_xyz(1:nat)=sin_xy(1:nat)*cos_z(1:nat,kz)+
     $                      cos_xy(1:nat)*sin_z(1:nat,kz)
            else    ! negative kz
             kzabs=-kz
             cos_xyz(1:nat)=cos_xy(1:nat)*cos_z(1:nat,kzabs)+
     $                      sin_xy(1:nat)*sin_z(1:nat,kzabs)
             sin_xyz(1:nat)=sin_xy(1:nat)*cos_z(1:nat,kzabs)-
     $                      cos_xy(1:nat)*sin_z(1:nat,kzabs)
            endif   ! (kz.ge.0)
            do iat = 1,nat
              itype = atomtype(iat)
              ci = q(itype)
              cossum = cossum + cos_xyz(iat)*ci
              sinsum = sinsum + sin_xyz(iat)*ci
            end do
            sfh = (cossum*cossum + sinsum*sinsum)*akvec
            recip = recip + sfh
            prefact = akvec*elfact
            do iat = 1,nat
              tmp=prefact*(cossum*sin_xyz(iat)-
     $                   sinsum*cos_xyz(iat))
              elf(1,iat) = elf(1,iat)+tmp*kx
              elf(2,iat) = elf(2,iat)+tmp*ky
              elf(3,iat) = elf(3,iat)+tmp*kz
              f1 = q(atomtype(iat))*tmp*332.08d0
              fewald(1,iat) = fewald(1,iat)+f1*kx
              fewald(2,iat) = fewald(2,iat)+f1*ky
              fewald(3,iat) = fewald(3,iat)+f1*kz
            end do
            tvirtmp(1,1)=tvirtmp(1,1)+sfh*(1.0-btens*kx*kx)
            tvirtmp(1,2)=tvirtmp(1,2)+sfh*(0.0-btens*kx*ky)
            tvirtmp(1,3)=tvirtmp(1,3)+sfh*(0.0-btens*kx*kz)
            tvirtmp(2,1)=tvirtmp(2,1)+sfh*(0.0-btens*ky*kx)
            tvirtmp(2,2)=tvirtmp(2,2)+sfh*(1.0-btens*ky*ky)
            tvirtmp(2,3)=tvirtmp(2,3)+sfh*(0.0-btens*ky*kz)
            tvirtmp(3,1)=tvirtmp(3,1)+sfh*(0.0-btens*kz*kx)
            tvirtmp(3,2)=tvirtmp(3,2)+sfh*(0.0-btens*kz*ky)
            tvirtmp(3,3)=tvirtmp(3,3)+sfh*(1.0-btens*kz*kz)
          end if
         end do
         kzmin=-klimit
        end do
        kymin=-klimit
      end do
C
      recip = recip*twopifact*332.08d0 - ewself
C
      do jj = 1,3
        do kk = 1,3
          tvirpo(kk,jj)=tvirpo(kk,jj)+332.08d0*tvirtmp(kk,jj)*twopifact
        end do
      end do
C
       vir = vir + recip
       unbd = unbd + recip
C
      return
      end 
C
      SUBROUTINE recipQmu()
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer kk,jj,iat,kx,ky,kz,kmag2,klimit2,itype
      real*8 front,twopifact,fouralpha2inv,rkmag2,btens,akvec
      real*8 cossum,sinsum,ci,dotik,sfh,prefact,f1,front2,elfact
      real*8 recip,tmp
      real*8 tvirtmp(3,3)
C
      real*8 cos_x(maxat,0:kmax)
      real*8 cos_y(maxat,0:kmax)
      real*8 cos_z(maxat,0:kmax)
      real*8 cos_xy(maxat)
      real*8 cos_xyz(maxat)
      real*8 sin_x(maxat,0:kmax)
      real*8 sin_y(maxat,0:kmax)
      real*8 sin_z(maxat,0:kmax)
      real*8 sin_xy(maxat)
      real*8 sin_xyz(maxat)
      real*8 f332
C
      real*8 virind
      real*8 find(3,maxat)
C
      real*8 costermp(maxat),sintermp(maxat),costermx(maxat)
      real*8 sintermx(maxat)
      real*8 hx,hy,hz  ! kx, ky,kz mulpiplied by front
C
      real*8 pk,cost,sint
      real*8 cossump,sinsump,cossumx,sinsumx,prefact1,prefact2
      real*8 twoalphapi,facti,factj,virfact
C
      integer kymin,kzmin,kyabs,kzabs
C
      real*8 sum_muksinx(3,3),sum_mukcosx(3,3)
      real*8 tmpxx,tmpxy,tmpxz,tmpyy,tmpyz,tmpzz,factstress
      real*8 strtmp(3,3)
      integer i,j
C
      common /ewaldrecip/ cos_x,cos_y,cos_z,sin_x,sin_y,sin_z
C
C     *****reciprocal2 part
      virind=0.0
      do iat=1,nat
        find(1,iat)=0.0d0
        find(2,iat)=0.0d0
        find(3,iat)=0.0d0
      end do
      do i=1,3
        do j=1,3
          strtmp(i,j)=0.0d0
        end do
      end do
C
      front = 2.0d0*pi/box
      front2 = front*front
      twopifact = 2.0d0*pi/(box*box*box)
      fouralpha2inv = 1.0d0/(4.0d0*alpha*alpha)
      elfact = 2.0d0*twopifact*front
      virfact = 4.0d0*twopifact
      klimit2 = klimit*klimit
C
      kymin=0
      kzmin=1
      do kx = 0,klimit
        hx=dble(kx)*front
        do ky = kymin,klimit  
          hy=dble(ky)*front
          if (ky.ge.0) then
           cos_xy(1:nat)=cos_x(1:nat,kx)*cos_y(1:nat,ky)-
     $                   sin_x(1:nat,kx)*sin_y(1:nat,ky)
           sin_xy(1:nat)=sin_x(1:nat,kx)*cos_y(1:nat,ky)+
     $                   cos_x(1:nat,kx)*sin_y(1:nat,ky)
          else    ! negative  ky
          kyabs=-ky
           cos_xy(1:nat)=cos_x(1:nat,kx)*cos_y(1:nat,kyabs)+
     $                   sin_x(1:nat,kx)*sin_y(1:nat,kyabs)
           sin_xy(1:nat)=sin_x(1:nat,kx)*cos_y(1:nat,kyabs)-
     $                   cos_x(1:nat,kx)*sin_y(1:nat,kyabs)
          endif
          do kz = kzmin,klimit 
            hz=dble(kz)*front
            kmag2 = kx*kx + ky*ky + kz*kz
            if (kmag2 .ge. klimit2) goto 342
            rkmag2 = dble(kmag2)*front2
            akvec = 2.0*dexp(-rkmag2*fouralpha2inv)/rkmag2  ! 2 accounts for the half sum used
            btens = 2.0d0*front2*(1.0d0+rkmag2*fouralpha2inv)/rkmag2
            if (kz.ge.0) then
             cos_xyz(1:nat)=cos_xy(1:nat)*cos_z(1:nat,kz)-
     $                      sin_xy(1:nat)*sin_z(1:nat,kz)
             sin_xyz(1:nat)=sin_xy(1:nat)*cos_z(1:nat,kz)+
     $                      cos_xy(1:nat)*sin_z(1:nat,kz)
            else    ! negative kz
             kzabs=-kz
             cos_xyz(1:nat)=cos_xy(1:nat)*cos_z(1:nat,kzabs)+
     $                      sin_xy(1:nat)*sin_z(1:nat,kzabs)
             sin_xyz(1:nat)=sin_xy(1:nat)*cos_z(1:nat,kzabs)-
     $                      cos_xy(1:nat)*sin_z(1:nat,kzabs)
            endif   ! (kz.ge.0)
            cossump = 0.0d0
            sinsump = 0.0d0
            cossumx = 0.0d0
            sinsumx = 0.0d0
            do i=1,3
              do j=i,3
                sum_mukcosx(i,j)=0.0d0
                sum_muksinx(i,j)=0.0d0
              end do
            end do
            do iat = 1,nat
              itype = atomtype(iat)
              ci = q(itype)
              pk = px(iat)*hx + py(iat)*hy + pz(iat)*hz
              cost = cos_xyz(iat)
              sint = sin_xyz(iat)
              costermx(iat) = ci*cost
              sintermx(iat) = ci*sint
              costermp(iat) = pk*cost
              sintermp(iat) = pk*sint
              cossump = cossump + pk*cost
              sinsump = sinsump + pk*sint
              cossumx = cossumx + ci*cost
              sinsumx = sinsumx + ci*sint
C
              if (lstress) then
              tmpxx=px(iat)*hx
              tmpxy=0.5*(px(iat)*hy+py(iat)*hx)
              tmpxz=0.5*(px(iat)*hz+pz(iat)*hx)
              tmpyy=py(iat)*hy
              tmpyz=0.5*(py(iat)*hz+pz(iat)*hy)
              tmpzz=pz(iat)*hz
C
              sum_muksinx(1,1)=sum_muksinx(1,1)+tmpxx*sint
              sum_muksinx(1,2)=sum_muksinx(1,2)+tmpxy*sint
              sum_muksinx(1,3)=sum_muksinx(1,3)+tmpxz*sint
              sum_muksinx(2,2)=sum_muksinx(2,2)+tmpyy*sint
              sum_muksinx(2,3)=sum_muksinx(2,3)+tmpyz*sint
              sum_muksinx(3,3)=sum_muksinx(3,3)+tmpzz*sint
              sum_mukcosx(1,1)=sum_mukcosx(1,1)+tmpxx*cost
              sum_mukcosx(1,2)=sum_mukcosx(1,2)+tmpxy*cost
              sum_mukcosx(1,3)=sum_mukcosx(1,3)+tmpxz*cost
              sum_mukcosx(2,2)=sum_mukcosx(2,2)+tmpyy*cost
              sum_mukcosx(2,3)=sum_mukcosx(2,3)+tmpyz*cost
              sum_mukcosx(3,3)=sum_mukcosx(3,3)+tmpzz*cost
              endif
            end do
            prefact1 = akvec*elfact  
            do iat = 1,nat
              f1 = prefact1*(costermp(iat)*cossumx - costermx(iat)*
     $          cossump + sintermp(iat)*sinsumx-sintermx(iat)*sinsump)
              find(1,iat) = find(1,iat) + f1*dble(kx)
              find(2,iat) = find(2,iat) + f1*dble(ky)
              find(3,iat) = find(3,iat) + f1*dble(kz)
            end do
C
            do i=1,3
              do j=i,3
                strtmp(i,j)=strtmp(i,j)+akvec*(sum_mukcosx(i,j)
     $            *sinsumx-sum_muksinx(i,j)*cossumx)
              end do
            end do
            if (lstress) then
             factstress=akvec*(sinsumx*cossump-cossumx*sinsump)
             strtmp(1,1)=strtmp(1,1)+factstress*(1.0d0-btens*kx*kx)
             strtmp(1,2)=strtmp(1,2)+factstress*(-btens*kx*ky)
             strtmp(1,3)=strtmp(1,3)+factstress*(-btens*kx*kz)
             strtmp(2,2)=strtmp(2,2)+factstress*(1.0d0-btens*ky*ky)
             strtmp(2,3)=strtmp(2,3)+factstress*(-btens*ky*kz)
             strtmp(3,3)=strtmp(3,3)+factstress*(1.0d0-btens*kz*kz)
            endif
C
            prefact2 = akvec*virfact*(rkmag2*fouralpha2inv-1.0d0)
            virind = virind + prefact2*(cossumx*sinsump-
     +                  sinsumx*cossump)
C
 342       continue
          end do
          kzmin=-klimit
        end do
        kymin=-klimit
      end do
C
      f332=332.08d0
      do iat=1,nat
        do kk=1,3
          f(kk,iat)=f(kk,iat)+f332*find(kk,iat)
        end do
      end do
      if (newpress) then
       vir=vir+f332*virind
       f332=2.0d0*332.08d0*twopifact   
       tvirpo(1,1)=tvirpo(1,1)+f332*strtmp(1,1)
       tvirpo(1,2)=tvirpo(1,2)+f332*strtmp(1,2)
       tvirpo(1,3)=tvirpo(1,3)+f332*strtmp(1,3)
       tvirpo(2,1)=tvirpo(2,1)+f332*strtmp(1,2)
       tvirpo(2,2)=tvirpo(2,2)+f332*strtmp(2,2)
       tvirpo(2,3)=tvirpo(2,3)+f332*strtmp(2,3)
       tvirpo(3,1)=tvirpo(3,1)+f332*strtmp(1,3)
       tvirpo(3,2)=tvirpo(3,2)+f332*strtmp(2,3)
       tvirpo(3,3)=tvirpo(3,3)+f332*strtmp(3,3)
      end if
      return
C      
C     energy contribution has been accounted for in interch()
C
      end 
C
      SUBROUTINE result()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer i,kk,jj,kstar,iprop,inos
      real*8 vol,ekb,eks,ekt,ept,ekk,ekp,ekc,ekm,hamilton
      data kstar /0/
      SAVE kstar
C
      open (65,file='fort.65',status='old',access='append')
C
      toten = poten + kinen
C
      vol = box*box*box
      ekb = 0.5d0*vlogv*vlogv*wtmass(1)
      eks = 0.5d0*qtmass(1)*vlogs(1)*vlogs(1)
      ekt = gn1kt*xlogs(1)
      ept = 1.4582454d-5*pfix*vol
      ekk = kinen
      ekp = poten
      ekc = 0.0d0
      ekm = 0.0d0
      do inos = 2,nnos
        ekc = ekc + 0.5d0*qtmass(inos)*vlogs(inos)*vlogs(inos)
        ekm = ekm + gkt*xlogs(inos)
      end do
      hamilton = ekb + eks + ekt + ept + ekk + ekp + ekc + ekm
C
      prop(1) = temp
      prop(2) = pres
      prop(3) = box
      prop(4) = toten
      prop(5) = hamilton
      prop(6) = ekk ! kinen
      prop(7) = ekp ! poten
      prop(8) = eks
      prop(9) = ekt
      prop(10) = ept
      prop(11) = ekb
      prop(12) = ekc
      prop(13) = ekm
      prop(14) = unbd
      prop(15) = toten + ept ! enthalpy
      prop(16) = stress(1,1)
      prop(17) = stress(2,1)
      prop(18) = stress(3,1)
      prop(19) = stress(1,2)
      prop(20) = stress(2,2)
      prop(21) = stress(3,2)
      prop(22) = stress(1,3)
      prop(23) = stress(2,3)
      prop(24) = stress(3,3)
C
      if (kount .eq. kinit)then
        do i = 1,maxprop
          avgsumpr(i) = 0.0d0
        end do
      end if
C
C     *****calculate diffusion
C
      CALL diffusion()
C
C     *****keep track of sums over last nave iterations
C
      do i = 1,maxprop
        sumpr(i) = sumpr(i) + prop(i)
        sumsqpr(i) = sumsqpr(i) + prop(i)*prop(i)
      end do
C
C     *****update overall averages & output them every 'nave' iterations
C
      if (mod(kount,kave) .eq. 0) then
        do i = 1,maxprop
          avgpr(i) = sumpr(i) / kave
          avgsqpr(i) = sumsqpr(i) /kave 
          sumpr(i) = 0.0d0
          sumsqpr(i) = 0.0d0
        end do
         if (kount.gt.kinit)then
          kstar  = kstar  + 1
          do i = 1,maxprop
            avgsumpr(i) = avgsumpr(i) + avgpr(i)
            avgsumsqpr(i) = avgsumsqpr(i) + avgsqpr(i)
            simpr(i) = avgsumpr(i) / kstar
            simsqpr(i) = avgsumsqpr(i) / kstar
          end do
       end if
      write(65,250)dble(kount)*stepout,(avgpr(iprop),iprop=1,5)
      end if
C
      if (printout) then
        if ( kount .le. nsteps)then
          write(6,220)kount,box,pres,temp,toten
          if (lstress)then
            write(6,*)'              *****   Stress Tensor  ***** '
            write(6,'(f15.2)')dble(kount)*stepout
            do kk = 1,3
              write(6,255)(stress(kk,jj),jj=1,3)
            end do
          end if
        end if
      end if
 220  format('Kount =',i7,' Box = ',f8.4,' P =',f11.2,' T = ',f9.3,
     +              ' E = ',f11.3)
 221  format (1x,4f15.6)
 250  format(f11.2,f9.2,f12.3,f10.4,2f12.2)
 255  format(8x,3f15.4)
      close (65)
C
      return
      end
      SUBROUTINE setup()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer iat,itype,jat,jtype,ii,imap,i,kk
      real*8 total,ro3,betae,beta,sumninj,value
      real*8 vcum(3)
      integer isum
      real*8 rint,rsum,rrf3,tau,w ! used for mu-mu damping func. calc
C
C     *****chvol stuff
C
      if (chvol) deltbox = (boxnew - box)/nsteps
C
C     *****check total charge of system
C
      total = 0.0d0
      do iat = 1,nat
        itype = atomtype(iat)
        total = total + q(itype)
      end do
      write (6,*) '*****total charge = ',total
C
C     *****eliminate center of mass velocity
C
      natreal=0
      do iat=1,nat
        if (.not.ldummy(iat)) natreal=natreal+1
      end do
      do kk = 1,3
        vcum(kk) = 0.0d0
      end do
      do iat = 1,nat
        do kk = 1,3
          vcum(kk) = vcum(kk) + v(kk,iat)*mass(iat)
        end do
      end do
      write (6,*) 'vcum',(vcum(kk)/natreal,kk=1,3)
      do iat = 1,nat
        if (.not.ldummy(iat)) then
          do kk = 1,3
            v(kk,iat) = v(kk,iat) - vcum(kk)/(dble(natreal)*mass(iat))
          end do
        end if
      end do
C
C     *****calculate inverse masses for stochast and shake
C
      do iat = 1,nat
        if (.not.ldummy(iat)) then
          massinv(iat) = 1.0d0/mass(iat)
         else
          massinv(iat) = 0.0d0
         endif
      end do
C
C     *****assign nonbonded cutoffs
C
      ro = rread
      ro2 = ro*ro
      rcut = ro + driftmax
      rcut2 = rcut*rcut
C
C
C     *****calculate sums for ptrunc and etrunc
C
      ro3 = ro**3
      betae = -(2.0d0*pi)/3.d0
      beta = betae*68570.d0*2.d0
      sumninj = 0.d0
      do iat = 1,nat
        itype = atomtype(iat)
        do jat = 1,nat
          jtype = atomtype(jat)
          ii = typee(itype,jtype)
          imap = map(ii)
          value = nonbonded(3,imap)
          sumninj = sumninj + value                     
        end do
      end do
      sumninj = sumninj/ro3
      psumninj = sumninj*beta
      esumninj = sumninj*betae
C
C
C     *****Polarization stuff: mu-mu tapering function constants  
C
C     tauden is the denomenator for tau tapering function
C
      if (tapering) then 
         tauden = ((ro-rtaper)**3)/6.0d0
       else     ! if no tapering set rtaper=ro, so that tau=1,dtau=0
         tauden = 1.0d0
         rtaper=ro
      end if
C
C     *****calculation of arf
C
      rsum = 0.0d0
      do isum = 1,1000
        rint = rtaper+(ro-rtaper)*(0.5+dble(isum))/1000.0d0
        rsum = rsum + rint*rint*tau(rint)
      end do
      rsum = rsum*3.0d0*(ro-rtaper)/1000.0d0
      rrf3 = rsum + rtaper**3
      arf = (epsrf - 1.0d0)/(rrf3*(epsrf + 0.5d0))
C
C
      if (constrain) then ! bond length constraints
        do i = 1,nbondt
          stretch(2,i) = stretch(3,i)
        end do
      end if
C
      do iat = 1,nat
        do kk = 1,3
          sumdel(kk,iat) = 0.0d0
          delx(kk,iat)   = 0.0d0
        end do
      end do
C
      return
      end
      SUBROUTINE shake(xref,xo,xshaken)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      real*8 xref(3,maxat),xo(3,maxat),xshaken(3,maxat)
C
C     *****local variables
C
      integer kk,jj,iat,jat,ibond,ih,ihyd,ihx,ic,icarb,iter,iatom,i
      integer iatom_m1,iatom_p1,icarb1,icarb2,icarb3
      real*8 g,facti,factj,c1mass,c2mass,c3mass,dv,dvi,eta,etac1,etac2
      real*8  hmass,sigma,rijmag,dot,rijshaken,delta,etac3,boxinv
      real*8 xrefun(3,maxat),xoun(3,maxat),xouno(3,maxat)
      real*8 rijref(3,maxbonds)
      real*8 force(3),fact,delt2inv
      real*8 dhdxaref(3,3,3,maxdeforms)
      real*8 dhdxa(3,3,3,maxdeforms)
      real*8 dotsuma(maxdeforms,3),denoma(maxdeforms,3)
      real*8 bxa(3),aminusb(3),avect(3),bvect(3),p(3),vect(3),xp(3,2)
C
C     *****commons
C
C
C     *****unwrap xref,xo
C
      CALL unwrap(xref,xrefun)
      CALL unwrap(xo,xoun)
C
C     *****store unwrapped initial coordinates if calculating pressure
C
      if (newpress) then
        do iat = 1,nat
          xouno(1,iat) = xoun(1,iat)
          xouno(2,iat) = xoun(2,iat)
          xouno(3,iat) = xoun(3,iat)
        end do
      end if
C
C     *****calculate rijref
C
      do ibond = 1,nbcon
          iat = bondcon(1,ibond)
          jat = bondcon(2,ibond)
          do  kk = 1,3
            rijref(kk,ibond) = xrefun(kk,iat) - xrefun(kk,jat)
          end do
       end do
C
C     *****calculate dhdxref and dhdx
C     *****aromatic hydrogens
C
      if (numaromatic .gt. 0) then
        CALL hydriva(xrefun,dhdxaref)       
        CALL hydriva(xoun, dhdxa)
        do  i = 1,numaromatic
          ihyd  = iaromatic(4,i)
          hmass = massinv(ihyd)
          do ihx = 1,3
            dotsuma(i,ihx) = 0.d0
            do ic = 1,3
              icarb = iaromatic(ic,i)
              dotsuma(i,ihx) = dotsuma(i,ihx)
     +                            +  massinv(icarb)*
     +        (dhdxa(ihx,ic,1,i)*dhdxaref(ihx,ic,1,i)
     +        +dhdxa(ihx,ic,2,i)*dhdxaref(ihx,ic,2,i)
     +        +dhdxa(ihx,ic,3,i)*dhdxaref(ihx,ic,3,i))
            end do
            denoma(i,ihx) = 1.0d0/(hmass + dotsuma(i,ihx))
          end do
        end do
      end if
C
C     *****begin iterative application of constraints
C
 
      iter = 0
1     sigma = 0.d0
      iter = iter + 1
      do ibond = 1,nbcon
        iat = bondcon(1,ibond)
        jat = bondcon(2,ibond)
        rijmag = 0.d0
        dot = 0.d0
        do kk = 1,3
          rijshaken = xoun(kk,iat) - xoun(kk,jat)
          dot = dot + rijref(kk,ibond)*rijshaken
          rijmag = rijmag + rijshaken*rijshaken
        end do
        delta = d2(ibond) - rijmag
        sigma = sigma + delta*delta
        g = delta*massred(ibond)/dot
        facti = g*massinv(iat)
        factj = -g*massinv(jat)
        do kk = 1,3
          xoun(kk,iat) = xoun(kk,iat) + facti*rijref(kk,ibond)
          xoun(kk,jat) = xoun(kk,jat) + factj*rijref(kk,ibond)
        end do
      end do
C
C     *****do aromatic hydrogens
C
        do i = 1,numaromatic
C
C     *****calculate predicted hydrogen positions
C
          iatom = iaromatic(2,i)
          iatom_m1 = iaromatic(1,i)
          iatom_p1 = iaromatic(3,i)
          icarb1 = iatom_m1
          icarb2 = iatom
          icarb3 = iatom_p1
          ih = iaromatic(4,i)
          c1mass = massinv(icarb1)
          c2mass = massinv(icarb2)
          c3mass = massinv(icarb3)
          hmass = massinv(ih)
          do kk = 1,3
            avect(kk) = xoun(kk,iatom)    - xoun(kk,iatom_m1)
            bvect(kk) = xoun(kk,iatom_p1) - xoun(kk,iatom)
          end do
C
C     *****calculate aminusb
C     *****calculate dv
C
          dv = 0.d0
          do kk = 1,3
            aminusb(kk) = avect(kk) - bvect(kk)
            dv = dv + aminusb(kk)*aminusb(kk)
          end do
C
          dvi = dbond(i)/dsqrt(dv)
C
C     *****calculate  v
C
          do kk = 1,3
            vect(kk) = aminusb(kk)*dvi
          end do
C
C     *****calculate hydrogen positions
C
          do kk = 1,3
            xp(kk,1) = xoun(kk,iatom) + idpar(i)*vect(kk) 
          end do
C
C     *****calculate constrained hydrogen posititions
C
          do ihx = 1,3
            eta = xp(ihx,1) - xoun(ihx,ih)
            sigma = sigma + eta*eta
            eta = eta*denoma(i,ihx)*hmass
c            eta = eta*denoma(i,ihx)/hmass
            xoun(ihx,ih) = eta*hmass + xoun(ihx,ih)
            etac1 = eta*c1mass     
            xoun(1,icarb1) = xoun(1,icarb1)
     +             -etac1*dhdxaref(ihx,1,1,i)
            xoun(2,icarb1) = xoun(2,icarb1)
     +             -etac1*dhdxaref(ihx,1,2,i)
            xoun(3,icarb1) = xoun(3,icarb1)
     +             -etac1*dhdxaref(ihx,1,3,i)
            etac2 = eta*c2mass     
            xoun(1,icarb2) = xoun(1,icarb2)
     +             -etac2*dhdxaref(ihx,2,1,i)
            xoun(2,icarb2) = xoun(2,icarb2)
     +             -etac2*dhdxaref(ihx,2,2,i)
            xoun(3,icarb2) = xoun(3,icarb2)
     +             -etac2*dhdxaref(ihx,2,3,i)
            etac3 = eta*c3mass     
            xoun(1,icarb3) = xoun(1,icarb3)
     +             -etac3*dhdxaref(ihx,3,1,i)
            xoun(2,icarb3) = xoun(2,icarb3)
     +             -etac3*dhdxaref(ihx,3,2,i)
            xoun(3,icarb3) = xoun(3,icarb3)
     +             -etac3*dhdxaref(ihx,3,3,i)
          end do
        end do
C
C     *****check for convergence
C
      sigma = sigma/nconst
      if (sigma .gt. tol) goto 1
C
C     *****apply periodic boundary conditions to shaken unwrapped coor.
C
      boxinv = 1.0d0/box
      do iat = 1,nat
        do kk = 1,3
          xshaken(kk,iat) = xoun(kk,iat)
          if (xshaken(kk,iat) .gt. box)
     +        xshaken(kk,iat) = xshaken(kk,iat)
     +           - box*(int(xshaken(kk,iat)*boxinv))
          if (xshaken(kk,iat) .lt. 0.0)
     +        xshaken(kk,iat) = xshaken(kk,iat)
     +           - box*(int(xshaken(kk,iat)*boxinv) - 1)
        end do
      end do
C
C     *****calculate virial if position shake
C
      if (newpress) then
        delt2inv = 1.0d0/(4.184d-4*delt*delt)
        do iat = 1,nat
          fact = mass(iat)*delt2inv
          do kk = 1,3
            force(kk) = (xoun(kk,iat) - xouno(kk,iat))*fact
            vir = vir + xoun(kk,iat)*force(kk)
            do jj = 1,3
              tvirpo(kk,jj) = tvirpo(kk,jj)+force(kk)*xoun(jj,iat)
            end do
          end do
        end do
      end if
C
      return
      end
C
      SUBROUTINE spline()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****local variables
C
      integer nfunct,ipoint,ifunct,innb,icoeff
      integer pivot(6),numpoints,kk
      real*8 aa(6),cc(6,6),xtemp(6),z,r2min
      real*8 fs(0:maxpoints), fsd(0:maxpoints), fsdd(0:maxpoints)
      real*8 coefff2(3,maxpoints,maxnnb)
      real*8 coefff3(3,maxpoints,maxnnb)
      real*8 coefff4(3,maxpoints,maxnnb)
      real*8 coefff5(3,maxpoints,maxnnb)
      real*8 coefff6(3,maxpoints,maxnnb)
      real*8 funct,functd,functdd
      real*8 gtaper,staper
C
C
      nfunct = 3
      r2min = 0.5d0
      numpoints = maxpoints
      deltaspline = 0.25d0
C
      do innb = 1,maxnnb
        do ipoint = 1,numpoints
          do kk=1,6
            coefft(kk,ipoint,innb) = 0.0d0
            coeffft(kk,ipoint,innb) = 0.0d0
            coeffee(kk,ipoint,innb) = 0.0d0
            coefffee(kk,ipoint,innb) = 0.0d0
          end do
        end do
      end do
C
C     *****initialize independent variable vector
C
C
      do ipoint = 0,numpoints
        zs(ipoint) = r2min + (ipoint)*deltaspline
      end do
C
      CALL cload(cc,deltaspline)
      CALL gauss(cc,6,6,pivot)
C
C     *****determine coefficient matrix 
C
      do ifunct = 1,nfunct
        do innb = 1,maxnnb
          do ipoint = 0,numpoints
            z = zs(ipoint)
            fs(ipoint) = funct(z,innb,ifunct)
            fsd(ipoint) = functd(z,innb,ifunct)
            fsdd(ipoint) = functdd(z,innb,ifunct)
          end do
C
          do ipoint = 1,numpoints
C
            aa(1) = fs(ipoint-1)
            aa(2) = fs(ipoint)
            aa(3) = fsd(ipoint-1)
            aa(4) = fsd(ipoint)
            aa(5) = fsdd(ipoint-1)
            aa(6) = fsdd(ipoint)
C
C     *****solve for coefficients
C
            CALL gsolve(cc,6,6,aa,xtemp,pivot)
C
            coeff1(ifunct,ipoint,innb) = xtemp(1)
            coeff2(ifunct,ipoint,innb) = xtemp(2)
            coeff3(ifunct,ipoint,innb) = xtemp(3)
            coeff4(ifunct,ipoint,innb) = xtemp(4)
            coeff5(ifunct,ipoint,innb) = xtemp(5)
            coeff6(ifunct,ipoint,innb) = xtemp(6)
C
            coefff2(ifunct,ipoint,innb) = xtemp(2)*2.0d0
            coefff3(ifunct,ipoint,innb) = xtemp(3)*4.0d0
            coefff4(ifunct,ipoint,innb) = xtemp(4)*6.0d0
            coefff5(ifunct,ipoint,innb) = xtemp(5)*8.0d0
            coefff6(ifunct,ipoint,innb) = xtemp(6)*10.0d0
          end do
        end do
      end do
C
C     *****calculate totals
C
      do innb = 1,maxnnb
        do ipoint = 1,numpoints
          do ifunct = 1,nfunct
            z = zs(ipoint)
            gtaper = (dsqrt(z) - (ros))/driftmax
            staper = 1.0d0+(gtaper*gtaper*(2.0d0*gtaper-3.0d0))
            coefft(1,ipoint,innb) = coefft(1,ipoint,innb) + 
     +                              coeff1(ifunct,ipoint,innb)
            coefft(2,ipoint,innb) = coefft(2,ipoint,innb) + 
     +                              coeff2(ifunct,ipoint,innb)
            coefft(3,ipoint,innb) = coefft(3,ipoint,innb) + 
     +                              coeff3(ifunct,ipoint,innb)
            coefft(4,ipoint,innb) = coefft(4,ipoint,innb) + 
     +                              coeff4(ifunct,ipoint,innb)
            coefft(5,ipoint,innb) = coefft(5,ipoint,innb) + 
     +                              coeff5(ifunct,ipoint,innb)
            coefft(6,ipoint,innb) = coefft(6,ipoint,innb) + 
     +                              coeff6(ifunct,ipoint,innb)
C
            coeffft(2,ipoint,innb) = coeffft(2,ipoint,innb) + 
     +                              coefff2(ifunct,ipoint,innb)
            coeffft(3,ipoint,innb) = coeffft(3,ipoint,innb) + 
     +                              coefff3(ifunct,ipoint,innb)
            coeffft(4,ipoint,innb) = coeffft(4,ipoint,innb) + 
     +                              coefff4(ifunct,ipoint,innb)
            coeffft(5,ipoint,innb) = coeffft(5,ipoint,innb) + 
     +                              coefff5(ifunct,ipoint,innb)
            coeffft(6,ipoint,innb) = coeffft(6,ipoint,innb) + 
     +                              coefff6(ifunct,ipoint,innb)
          end do
          if (z .lt. rshort2)then
            do icoeff = 2,6
              coeffft1(icoeff,ipoint,innb) = coeffft(icoeff,ipoint,innb)
              if (z .gt. ros2)then
             coeffft1(icoeff,ipoint,innb) = coeffft(icoeff,ipoint,innb)*
     +                                       staper
              end if
            end do
          end if
        end do
      end do
C
      do innb = 1,maxnnb
        do ipoint = 1,maxpoints
          coeffee(1,ipoint,innb) = coeff1(1,ipoint,innb)
          coeffee(2,ipoint,innb) = coeff2(1,ipoint,innb)
          coeffee(3,ipoint,innb) = coeff3(1,ipoint,innb)
          coeffee(4,ipoint,innb) = coeff4(1,ipoint,innb)
          coeffee(5,ipoint,innb) = coeff5(1,ipoint,innb)
          coeffee(6,ipoint,innb) = coeff6(1,ipoint,innb)
          coefffee(2,ipoint,innb) = coefff2(1,ipoint,innb)
          coefffee(3,ipoint,innb) = coefff3(1,ipoint,innb)
          coefffee(4,ipoint,innb) = coefff4(1,ipoint,innb)
          coefffee(5,ipoint,innb) = coefff5(1,ipoint,innb)
          coefffee(6,ipoint,innb) = coefff6(1,ipoint,innb)
        end do
      end do
C
      return
      end
      double precision FUNCTION tau(r)
C
      include "params.h"
      include "dimensions.h"
C
C     *****shared variables
C
      real*8 r
C
C     *****local variables
C
      real*8 w
C
      include "commons.h"
C
      w= r-rtaper
C
      tau = 1.0d0
      if (r.gt.rtaper) then
        tau = tau + w*w*(w/3.0d0+0.5*(rtaper-ro))/tauden
      endif
C
      return
      end
C
      double precision FUNCTION dtau(r)
C
      include "params.h"
      include "dimensions.h"
C
C     *****shared variables
C
      real*8 r
      include "commons.h"
C
      if (r.gt.rtaper) then
        dtau = ((r-ro)*(r-rtaper)**2)/tauden
       else
        dtau=0.0d0
      end if
C
      return
      end
C
      SUBROUTINE unwrap(xrf,xunrf)
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
C     *****shared variables
C
      real*8 xrf(3,maxat),xunrf(3,maxat)
C
C     *****local variables
C
      integer iat,ibond,kk,jat
      real*8 diff
C
C
      el2 = box/2.0d0
      do iat = 1,nat
        xunrf(1,iat) = xrf(1,iat)
        xunrf(2,iat) = xrf(2,iat)
        xunrf(3,iat) = xrf(3,iat)
      end do
C
C     *****unwrap by bonds
C
        do ibond = 1,nbonds
          iat = bonds(1,ibond)
          jat = bonds(2,ibond)
          do kk = 1,3
            diff = xrf(kk,jat) - xrf(kk,iat)
            if (abs(diff) .gt. el2)
     +        diff = diff - dsign(box,diff)
            xunrf(kk,jat) = xunrf(kk,iat) + diff
          end do
        end do
C
      return
      end
C
      SUBROUTINE exclude()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
      integer i,j,iat,nex,ibond,ix,iex,icheck,iy,ineigh,itype
      integer jat,jtype,kat,k,ibend,itmp,i14a,i14b,ibond1,ibond2
      logical HasBeenUsed
      logical Found14
      do iat=1,nat
        listm14(iat)=0
        listmex(iat)=0
      end do     ! end of additions in v3.0
C
      do 10000, iat = 1, nat - 1  ! search tables for each atom
        nex=0  ! initialized number excluded to 0
C
C     *****determine all 2 center interactions with iat
C
          do 2600, ibond=1,nbonds
            if (bonds(1,ibond) .eq. iat ) then ! exclude
              ix=bonds(2,ibond)
              if (ix .gt. iat) then
                do iex = 1,nex
                  icheck = listex(iex,iat)
                  if (ix .eq. icheck) goto 2600 ! duplicate
                end do
                nex=nex+1
                listex(nex,iat)=ix
              end if
             else if (bonds(2,ibond) .eq. iat) then
              ix=bonds(1,ibond)
              if (ix .gt. iat) then
                do iex = 1,nex
                  icheck = listex(iex,iat)
                  if (ix .eq. icheck) goto 2600 ! duplicate
                end do
                nex=nex+1
                listex(nex,iat)=ix
              end if
            end if
2600      continue
C
C     *****determine all 3 center interactions with iat as first atom
C          or last atom
C          check atoms bonded to 3 center removed atom for 4 center
C          interactions
C
          do 2700, ibend=1,nbends
            if (bends(1,ibend) .eq. iat ) then ! exclude
              ix=bends(3,ibend)
C
C     *****check for four center interactions
C
           if (ex14)then
            do 2610, ibond = 1,nbonds
              if (bonds(1,ibond) .eq. ix) then
                iy = bonds(2,ibond)
              if (iy .gt. iat)then
                do 2605, ineigh = 1,nex
                  if (listex(ineigh,iat) .eq. iy) go to 2610
2605              continue
                 nex = nex + 1
                 listex(nex,iat) = iy
              end if
             else if (bonds(2,ibond) .eq. ix) then
                  iy = bonds(1,ibond)
                  if (iy .gt. iat)then
                do 2607, ineigh = 1,nex
                  if (listex(ineigh,iat) .eq. iy) go to 2610
2607             continue
                 nex = nex + 1
                 listex(nex,iat) = iy
                end if
              end if
2610         continue
           end if
C
C     *****do three center
C
              if (ix .gt. iat) then
                do iex = 1,nex
                  icheck = listex(iex,iat)
                  if (ix .eq. icheck) goto 2700 ! duplicate
                end do
               nex=nex+1
               listex(nex,iat)=ix
              end if
             else if (bends(3,ibend) .eq. iat) then
              ix=bends(1,ibend)
C
C     *****four center
C
            if (ex14)then
             do 2620, ibond = 1,nbonds
               if(bonds(1,ibond) .eq. ix)then
                 iy=bonds(2,ibond)
                 if (iy .gt. iat)then
                  do 2615, ineigh = 1,nex
                    if (listex(ineigh,iat) .eq. iy) goto 2620
2615                continue
                   nex = nex + 1
                   listex(nex,iat) = iy
                  end if
               else if (bonds(2,ibond) .eq. ix)then
                iy = bonds(1,ibond)
                if (iy .gt. iat)then
                 do 2617, ineigh = 1,nex
                  if (listex(ineigh,iat) .eq. iy) goto 2620
2617              continue
                 nex = nex + 1
                 listex(nex,iat) = iy
                 end if
              end if
2620         continue
          end if
C
C     *****do three center
C
              if (ix .gt. iat) then
                do iex = 1,nex
                  icheck = listex(iex,iat)
                  if (ix .eq. icheck) goto 2700 ! duplicate
                end do
                nex=nex+1
                listex(nex,iat)=ix
              end if
            end if
2700      continue
4000    continue
        listmex(iat)=nex
10000 continue
C
C
C    make a list of all 1-4 centers for reduced 1-4 interactions 
C
         if (lredonefour.or.lredQ_mu14) then
           do ibend=1,nbends
             iat=bends(1,ibend)
             kat=bends(3,ibend)
             do ibond=1,nbonds
               Found14=.false.
               ibond1=bonds(1,ibond)
               ibond2=bonds(2,ibond)
               if (ibond1.eq.kat.and.ibond2.ne.bends(2,ibend)) then
                 i14a=iat
                 i14b=ibond2
                 Found14=.true.
               end if
               if (ibond2.eq.kat.and.ibond1.ne.bends(2,ibend)) then
                 i14a=iat
                 i14b=ibond1
                 Found14=.true.
               end if
               if (ibond1.eq.iat.and.ibond2.ne.bends(2,ibend)) then
                 i14a=kat
                 i14b=ibond2
                 Found14=.true.
               end if
               if (ibond2.eq.iat.and.ibond1.ne.bends(2,ibend)) then
                 i14a=kat
                 i14b=ibond1
                 Found14=.true.
               end if
               if (Found14) then
                 if (ldummy(i14a).or.ldummy(i14b)) Found14=.false.
               end if
               if (i14a.eq.i14b) Found14=.false.
               if (Found14) then
                 if (i14a.gt.i14b) then
                   itmp=i14a
                   i14a=i14b
                   i14b=itmp
                 end if
                 HasBeenUsed=.false. 
                 do ineigh = 1,listm14(i14a)
                   if (list14(ineigh,i14a).eq.i14b) HasBeenUsed=.true.
                 end do
                 do ineigh = 1,listm14(i14b)
                   if (list14(ineigh,i14b).eq.i14a) HasBeenUsed=.true.
                 end do
C
C  remove 1-4 that are part of listex (matters for cyclic molecules) (v2.7b)
C
                 do i=1,listmex(i14a)
                   if (listex(i,i14a).eq.i14b) HasBeenUsed=.true.
                 end do
C             
                 if (.not.HasBeenUsed) then
                   listm14(i14a)=listm14(i14a)+1
                   list14(listm14(i14a),i14a)=i14b
                 end if
               end if 
             end do
          end do
        endif ! (lredonefour)
C
      return
      end
C
C     *** DoDummy subroutine puts two dummy atoms based upon bend coordinates
C         and distributes the force on them correctly
C         see jcc 1999_v20_p786
C         v2.0 ether-based and carbonyl-based Lps are treated separately
C 
      Subroutine DoDummy(fdummy)
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
      integer idummy,idat,kk,jj,itype,jat,kat,iat
      real*8 aa,bb,cc,rid(3),rij(3),rjk(3),rik(3),rjd(3),rkd(3),factor
      real*8 rij_a_rjk(3),absrij_a_rjkinv,f1(3),Fd(3) ! rij_a_rjk  = r_ij + a*r_jk
      real*8 fdummy(3,maxat),ftmp,Fddotrid,Fddotrjk,rjkabs,rjkabs2
      real*8 tmp1,tmp2,tmp3
      real*8 rijabs2,rikabs2
      real*8 cross(3)     ! cross product of r_ij x r_ik
      real*8 crossdotFd   ! [r_ij x r_ik] . Fd
      real*8 rijdotrik    ! rij dot product rik
      real*8 crossabs
      real*8 fiout(3),fjout(3),fkout(3)
      real*8 zero
      parameter (zero=1.0d-12)
C
      el2=box*0.5d0
      virdummy=0.0d0   ! virdummy and tvirdummy are common variables
      do kk=1,3
       tvirdummy(kk,1)=0.0d0
       tvirdummy(kk,2)=0.0d0
       tvirdummy(kk,3)=0.0d0
      end do
C
      do idummy=1,ndummy
        idat=jikdummy(4,idummy) ! idat is the atom number for the dummy atom
        do kk=1,3
          Fd(kk)=fdummy(kk,idat)
        end do
        itype=jikdummy(5,idummy) ! dummy atom type
        aa=adummy(itype)
        bb=bdummy(itype)
        cc=cdummy(itype)
        jat=jikdummy(1,idummy) 
        iat=jikdummy(2,idummy) 
        kat=jikdummy(3,idummy) 
        absrij_a_rjkinv=0.0d0
        rijabs2=0.0d0
        rikabs2=0.0d0
        rjkabs2=0.0d0
        rijdotrik=0.0d0
        do kk=1,3
          rij(kk)=x(kk,jat)-x(kk,iat)
          if (abs(rij(kk)).gt.el2) rij(kk)=rij(kk) - dsign(box,rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)
C
          rjk(kk)=x(kk,kat)-x(kk,jat)
          if (abs(rjk(kk)).gt.el2) rjk(kk)=rjk(kk) - dsign(box,rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)
C
          rik(kk)=x(kk,kat)-x(kk,iat)
          if (abs(rik(kk)).gt.el2) rik(kk)=rik(kk) - dsign(box,rik(kk))
C
          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        end do
        rjkabs=dsqrt(rjkabs2)
        absrij_a_rjkinv=1.0d0/dsqrt(absrij_a_rjkinv)
        factor=-bb*absrij_a_rjkinv
        do kk=1,3
          rid(kk)=factor*rij_a_rjk(kk)
        end do
        Fddotrid=0.0d0  
        do kk=1,3
          Fddotrid=Fddotrid+Fd(kk)*rid(kk)
        end do
        if (abs(bb).gt.zero) then
          do kk=1,3
            f1(kk)=Fddotrid*rid(kk)/(bb*bb)
          end do
         else
           f1(1)=0.0d0
           f1(2)=0.0d0
           f1(3)=0.0d0
        end if
C
        if (LpGeomType(itype).le.2) then
C
C        [rij x rik]
C
          cross(1)= rij(2)*rik(3) - rik(2)*rij(3)
          cross(2)=-rij(1)*rik(3) + rik(1)*rij(3)
          cross(3)= rij(1)*rik(2) - rik(1)*rij(2)
          crossdotFd=cross(1)*Fd(1)+cross(2)*Fd(2)+cross(3)*Fd(3)
C
C       grad w.r.t iat, jat, kat
C
          crossabs=dsqrt(rijabs2*rikabs2-rijdotrik**2)
          if (abs(crossabs).lt.zero) crossabs=zero
          tmp1=1.0/crossabs
C
          fiout(1)=(Fd(2)*(rik(3)-rij(3))+Fd(3)*(rij(2)-rik(2)))*tmp1
          fiout(2)=(Fd(1)*(rij(3)-rik(3))+Fd(3)*(rik(1)-rij(1)))*tmp1
          fiout(3)=(Fd(1)*(rik(2)-rij(2))+Fd(2)*(rij(1)-rik(1)))*tmp1
C
          fjout(1)=(Fd(2)*(-rik(3))+Fd(3)*rik(2))*tmp1
          fjout(2)=(Fd(3)*(-rik(1))+Fd(1)*rik(3))*tmp1
          fjout(3)=(Fd(1)*(-rik(2))+Fd(2)*rik(1))*tmp1
C
          fkout(1)=(Fd(2)*rij(3)-Fd(3)*rij(2))*tmp1
          fkout(2)=(Fd(3)*rij(1)-Fd(1)*rij(3))*tmp1
          fkout(3)=(Fd(1)*rij(2)-Fd(2)*rij(1))*tmp1
C
          tmp3=tmp1**3
          do kk=1,3
            fiout(kk)=fiout(kk) + crossdotFd*tmp3*(rikabs2*rij(kk) +
     $       rijabs2*rik(kk) - rijdotrik*(rik(kk) + rij(kk)))
            fjout(kk)=fjout(kk) - crossdotFd*tmp3*(rikabs2*rij(kk) -
     $       rijdotrik*rik(kk))
            fkout(kk)=fkout(kk) - crossdotFd*tmp3*(rijabs2*rik(kk) -
     $       rijdotrik*rij(kk))
          end do
C
          do kk=1,3
            rid(kk)=rid(kk)+cc*cross(kk)*tmp1
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box) x(kk,idat)=x(kk,idat)-box
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box
             rkd(kk)=x(kk,idat)-x(kk,kat)
             if (abs(rkd(kk)).gt.el2) rkd(kk)=rkd(kk)-dsign(box,rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
             if (abs(rjd(kk)).gt.el2) rjd(kk)=rjd(kk)-dsign(box,rjd(kk))
          end do
C
C  *** calculate forces on atoms jat,iat,kat due to lone pair
C  *** reuse tmp1,tmp3 below with a different meaning 
C
          do kk=1,3
           ftmp=factor*(Fd(kk) - f1(kk))
           tmp1=Fd(kk)-ftmp + fiout(kk)*cc
           tmp2=(1.0d0-aa)*ftmp + fjout(kk)*cc
           tmp3=aa*ftmp + fkout(kk)*cc
           fdummy(kk,iat)=fdummy(kk,iat)+tmp1
           fdummy(kk,jat)=fdummy(kk,jat)+tmp2
           fdummy(kk,kat)=fdummy(kk,kat)+tmp3
           if (newpress) then
             virdummy=virdummy-(rid(kk)*tmp1+rjd(kk)*tmp2+rkd(kk)*tmp3)
             tvirdummy(kk,1)=tvirdummy(kk,1)-(rid(1)*tmp1+rjd(1)*tmp2+
     $                               rkd(1)*tmp3)
             tvirdummy(kk,2)=tvirdummy(kk,2)-(rid(2)*tmp1+rjd(2)*tmp2+
     $                               rkd(2)*tmp3)
             tvirdummy(kk,3)=tvirdummy(kk,3)-(rid(3)*tmp1+rjd(3)*tmp2+
     $                               rkd(3)*tmp3)
           endif
          end do
        endif    !  (LpGeomType(itype).le.2)
C
C    handle lone pairs on carbonyls (LpGeomType = 3)
C
        if (LpGeomType(itype).eq.3) then
          do kk=1,3
            rid(kk)=rid(kk)+cc*rjk(kk)/rjkabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box) x(kk,idat)=x(kk,idat)-box
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box
             rkd(kk)=x(kk,idat)-x(kk,kat)
             if (abs(rkd(kk)).gt.el2) rkd(kk)=rkd(kk)-dsign(box,rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
             if (abs(rjd(kk)).gt.el2) rjd(kk)=rjd(kk)-dsign(box,rjd(kk))
          end do
C
C  *** calculate forces on atoms jat,iat,kat due to lone pair
C  *** reuse tmp1,tmp3 below with a different meaning 
C
          Fddotrjk=0.0d0
          do kk=1,3
            Fddotrjk=Fddotrjk+rjk(kk)*Fd(kk)
          end do
          do kk=1,3
           ftmp=factor*(Fd(kk) - f1(kk))
           tmp1=Fd(kk)-ftmp  
           tmp2=(1.0d0-aa)*ftmp-
     $        cc*(Fd(kk)-Fddotrjk*rjk(kk)/rjkabs2)/rjkabs 
           tmp3=aa*ftmp + 
     $        cc*(Fd(kk)-Fddotrjk*rjk(kk)/rjkabs2)/rjkabs 
           fdummy(kk,iat)=fdummy(kk,iat)+tmp1
           fdummy(kk,jat)=fdummy(kk,jat)+tmp2
           fdummy(kk,kat)=fdummy(kk,kat)+tmp3
           if (newpress) then
             virdummy=virdummy-(rid(kk)*tmp1+rjd(kk)*tmp2+rkd(kk)*tmp3)
             tvirdummy(kk,1)=tvirdummy(kk,1)-(rid(1)*tmp1+rjd(1)*tmp2+
     $                               rkd(1)*tmp3)
             tvirdummy(kk,2)=tvirdummy(kk,2)-(rid(2)*tmp1+rjd(2)*tmp2+
     $                               rkd(2)*tmp3)
             tvirdummy(kk,3)=tvirdummy(kk,3)-(rid(3)*tmp1+rjd(3)*tmp2+
     $                               rkd(3)*tmp3)
           endif
          end do
        end if   !  (LpGeomType(itype).eq.3)
C       zero the force on the dummy atom after it has been distributed
        do kk=1,3
          fdummy(kk,idat)=0.0d0
        end do
      end do     ! next idummy
C
D     if (newpress) then
D      write(6,*) "virdummy=",virdummy*2.2857d+04/(box**3)
D      do kk=1,3
D       write(6,'(3F12.3)')(tvirdummy(kk,jj)*2.2857d+04/(box**3),jj=1,3)
D      end do
D     endif
      return
      end  
C
C     *** DoDummy subroutine puts two dummy atoms based upon bend coordinates
C         and distributes the force on them correctly
C         see jcc 1999_v20_p786
C 
      Subroutine DoDummyCoords()
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
      integer idummy,idat,kk,itype,jat,kat,iat
      real*8 aa,bb,cc,rid(3),rij(3),rjk(3),rik(3),factor
      real*8 rij_a_rjk(3),absrij_a_rjkinv       ! rij_a_rjk  = r_ij + a*r_jk
      real*8 rjkabs2,rjkabs
      real*8 zero
      real*8 rijabs2,rikabs2
      real*8 cross(3)        ! cross product of r_ij x r_ik
      real*8 rijdotrik    ! rij dot product rik
      real*8 crossabs
      parameter (zero=1.0d-12)
C
      el2=box*0.5d0
      do idummy=1,ndummy
        idat=jikdummy(4,idummy) ! idat is the atom number for the dummy atom
        itype=jikdummy(5,idummy) ! dummy atom type
        aa=adummy(itype)
        bb=bdummy(itype)
        cc=cdummy(itype)
        jat=jikdummy(1,idummy) 
        iat=jikdummy(2,idummy) 
        kat=jikdummy(3,idummy) 
        absrij_a_rjkinv=0.0d0
        rijabs2=0.0d0
        rikabs2=0.0d0
        rjkabs2=0.0d0
        rijdotrik=0.0d0
        do kk=1,3
          rij(kk)=x(kk,jat)-x(kk,iat)
          if (abs(rij(kk)).gt.el2) rij(kk)=rij(kk) - dsign(box,rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)
C
          rjk(kk)=x(kk,kat)-x(kk,jat)
          if (abs(rjk(kk)).gt.el2) rjk(kk)=rjk(kk) - dsign(box,rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)
C
          rik(kk)=x(kk,kat)-x(kk,iat)
          if (abs(rik(kk)).gt.el2) rik(kk)=rik(kk) - dsign(box,rik(kk))
C
          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        end do
        rjkabs=dsqrt(rjkabs2)
        crossabs=dsqrt(rijabs2*rikabs2-rijdotrik**2)
        if (abs(crossabs).lt.zero) crossabs=zero
        absrij_a_rjkinv=1.0d0/dsqrt(absrij_a_rjkinv)
        factor=-bb*absrij_a_rjkinv
        do kk=1,3
          rid(kk)=factor*rij_a_rjk(kk)
        end do
        if (LpGeomType(itype).le.2) then
C
C         [rij x rik]
C
          cross(1)= rij(2)*rik(3) - rik(2)*rij(3)
          cross(2)=-rij(1)*rik(3) + rik(1)*rij(3)
          cross(3)= rij(1)*rik(2) - rik(1)*rij(2)
          do kk=1,3
            rid(kk)=rid(kk)+cc*cross(kk)/crossabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box) x(kk,idat)=x(kk,idat)-box
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box
          end do
        end if     ! (LpGeomType(itype).le.2)
C
        if (LpGeomType(itype).eq.3) then
          do kk=1,3
            rid(kk)=rid(kk)+cc*rjk(kk)/rjkabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box) x(kk,idat)=x(kk,idat)-box
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box
          end do
        end if     ! (LpGeomType(itype).eq.3)
      end do       ! do idummy=1,ndummy
      return
      end  
C
      SUBROUTINE CheckBondNumber()
C
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
C
      integer iat,jat,ichain,ibond
      logical AtomUsed(maxat)
      logical ChainUsed(maxat)
C
C     *** check to make sure that bonds numbering is
C     *** consistent with unwrap for Lucretius
C
      do iat=1,nat
        AtomUsed(iat)=.false.
        ichain=Chain(iat)
        ChainUsed(ichain)=.false.
      end do
      do ibond = 1,nbonds
        iat = bonds(1,ibond)
        jat = bonds(2,ibond)
        ichain=Chain(iat)
        if (.not.AtomUsed(iat)) then
          if (.not.ChainUsed(ichain)) then
             ChainUsed(ichain)=.true.   ! first atom from the chain
           else
             write(6,*) "Bond Sorting Problem",iat,jat
             stop
          endif
        end if
        AtomUsed(iat)=.true.
        AtomUsed(jat)=.true.
      end do
      return
      end
C
C *** DoDummyInit() subroutine assigns lone pairs with C>0
C     cc=abs(cdummy) and cc=-abs(cdummy)
C     It assings a positive c to the first dummy atom (force center)
C     of a lone pair and a negative c to the second dummy atom of a lone pair
C     It also modifies listex() and list14()
C 
C     ! IMPORTANT ! It needs to be called in initialize right after exclude()        
C     but before boxes()
      Subroutine DoDummyInit()
      implicit none
      include "params.h"
      include "dimensions.h"
      include "commons.h"
      integer nCallDummyInit
      integer ibond,ibend,ineigh,iox,iex,jtmp,i,j,kk,ipos,itmp
      integer idummy,jdummy,idat,iat,jat,itype,jtype,iatConnect
      integer iatCheck,jatCheck,katCheck,itypeLpCheck ! added in v2.12
      integer SkipBend,idType  ! added in v2.13
      real*8 cci
      logical HasBeenFound
      logical HasBeenChanged(maxdummy)
C 
      data nCallDummyInit/0/
      save nCallDummyInit
C
      if (nCallDummyInit.ge.1) then
        write(6,*) "DoDummyInit() needs to be called only once"
        write(6,*) "Multiple calls to DoDummyInit() are found"
        return
      end if
      nCallDummyInit=nCallDummyInit+1 
C
C *** for all dummy atoms find the corresponding bond and then bends
C
      idat=0
      do idummy=1,nat
        if (ldummy(idummy)) then
          idat=idat+1
          itype=typedummy(atomtype(idummy))
          itypeLpCheck=-1
          HasBeenFound=.false.
          do ibond=1,nbonds
           if (bonds(1,ibond).eq.idummy) then
             iat=bonds(2,ibond)
             HasBeenFound=.true.
           end if
           if (bonds(2,ibond).eq.idummy) then
             iat=bonds(1,ibond)
             HasBeenFound=.true.
           end if
          end do
          if (.not.HasBeenFound) then
            write(6,*)"Cannot find an atom the Lp ",idummy,
     $                " is connected to"
            write(6,*) "Program exits"
            stop
          end if
          if (LpGeomType(itype).le.2) then  ! 1 (1Lp) or 2 (2Lp ether-like) 
            do ibend=1,nbends               ! added in v2.12
              jatCheck=bends(1,ibend)
              iatCheck=bends(2,ibend)
              katCheck=bends(3,ibend)
             do idType=1,ndummytypes
              if (iatCheck.eq.iat.and.
     $           atomtype(jatCheck).eq.LpBendAtomType(1,idType).and.
     $           atomtype(iatCheck).eq.LpBendAtomType(2,idType).and.
     $           atomtype(katCheck).eq.LpBendAtomType(3,idType)) then
                 jikdummy(1,idat)=jatCheck
                 jikdummy(2,idat)=iatCheck
                 jikdummy(3,idat)=katCheck
                 jikdummy(4,idat)=idummy
                 jikdummy(5,idat)=itype
                 itypeLpCheck=itype
              endif
              if (atomtype(jatCheck).ne.atomtype(katCheck)) then  ! check the bend from another side
                if (iatCheck.eq.iat.and.
     $            atomtype(jatCheck).eq.LpBendAtomType(3,idType).and.
     $            atomtype(iatCheck).eq.LpBendAtomType(2,idType).and.
     $            atomtype(katCheck).eq.LpBendAtomType(1,idType)) then
                  jikdummy(1,idat)=katCheck
                  jikdummy(2,idat)=iatCheck
                  jikdummy(3,idat)=jatCheck
                  jikdummy(4,idat)=idummy
                  jikdummy(5,idat)=itype
                  itypeLpCheck=itype
                end if
               end if
              end do   ! next idType
            end do      ! plus-minus (up-down) of Lp in a pair will be assigned later
C
C  *** for carbonyl-like Lps find the atom (iatConnect) to which the iat is connected to
C  *** and than find jat and kat using bend information
C
          else if (LpGeomType(itype).eq.3) then  ! type 3 means carbonyl-like Lps
            do ibond=1,nbonds
             if (bonds(1,ibond).eq.iat.and.
     $             (.not.ldummy(bonds(2,ibond))))then
               iatConnect=bonds(2,ibond)
               HasBeenFound=.true.
             end if
             if (bonds(2,ibond).eq.iat.and.
     $             (.not.ldummy(bonds(1,ibond))))then
               iatConnect=bonds(1,ibond)
               HasBeenFound=.true.
             end if
            end do
            if (.not.HasBeenFound) then
             write(6,*)"Cannot find an atom needed to define Lp=",idummy
             write(6,*) "Program exits"
             stop
            endif
            do ibend=1,nbends      ! added in v2.12
              jatCheck=bends(1,ibend)
              iatCheck=bends(2,ibend)
              katCheck=bends(3,ibend)
              SkipBend=.false.
              if (iat.eq.jatCheck) SkipBend=.true.
              if (iat.eq.katCheck) SkipBend=.true.
              if (iatCheck.ne.iatConnect) SkipBend=.true.
              do idType=1,ndummytypes
               if ((SkipBend.eq.0).and.
     $           atomtype(jatCheck).eq.LpBendAtomType(1,idType).and.
     $           atomtype(iatCheck).eq.LpBendAtomType(2,idType).and.
     $           atomtype(katCheck).eq.LpBendAtomType(3,idType)) then
                 jikdummy(1,idat)=jatCheck
                 jikdummy(2,idat)=iat
                 jikdummy(3,idat)=katCheck
                 jikdummy(4,idat)=idummy
                 jikdummy(5,idat)=itype
                 itypeLpCheck=itype
               endif
               if (atomtype(jatCheck).ne.atomtype(katCheck)) then  ! check the bend from another side
                if ((SkipBend .eq. 0).and.
     $            atomtype(jatCheck).eq.LpBendAtomType(3,idType).and.
     $            atomtype(iatCheck).eq.LpBendAtomType(2,idType).and.
     $            atomtype(katCheck).eq.LpBendAtomType(1,idType)) then
                 jikdummy(1,idat)=katCheck
                 jikdummy(2,idat)=iat
                 jikdummy(3,idat)=jatCheck
                 jikdummy(4,idat)=idummy
                 jikdummy(5,idat)=itype
                 itypeLpCheck=itype
                end if
               end if
              end do    ! next idType
            end do      ! plus-minus (up-down) of Lp in a pair will be assigned later
          else
            write(6,*) "LpGeomType should be 1,2,3 for this version" 
            write(6,*) "found LpGeomType",LpGeomType(itype)
            write(6,*) "Program will stop"
            stop
         end if    ! (LpGeomType(itype).le.1) then
         if (itypeLpCheck.ne.itype) then
           write(6,*) "Lp has not been identified",idat,itypeLpCheck
         end if
        end if         
      end do
      write(6,'("ndummy=",I9)')ndummy
      write(6,'("idat=",I9)') idat
C
C     for all Lp atoms 
C     1. Copy oxygen list14()
C     2. find the connected oxygen and include in the listex() 
C     for the Lp atom all atoms from the oxygen listex() except itself
C
      do idat=1,ndummy
        idummy=jikdummy(4,idat)
        iox=jikdummy(2,idat)
        listm14(idummy)=listm14(iox)
        do ineigh=1,listm14(iox)   ! copy oxygen list14 into dummy list14
          list14(ineigh,idummy)=list14(ineigh,iox)
        end do
        listmex(idummy)=listmex(iox)-1   ! 
        j=0
        do ineigh=1,listmex(iox)   ! copy oxygen listex into dummy listex except itself
          iex=listex(ineigh,iox)
          if (iex.ne.idummy) then
            j=j+1
            listex(j,idummy)=listex(ineigh,iox)
          endif
        end do
C
C  *** go through atoms below iox for which iox is excluded and included them
C      in listex(kk,idummy)
C
        do iat=1,iox
          do ineigh=1,listm14(iat)   
            if (list14(ineigh,iat).eq.iox) then 
              kk=listm14(idummy)+1
              listm14(idummy)=kk
              list14(kk,idummy)=iat
            end if
          end do
        end do
        do iat=1,iox
          do ineigh=1,listmex(iat)   
            if (listex(ineigh,iat).eq.iox) then 
              kk=listmex(idummy)+1
              listmex(idummy)=kk
              listex(kk,idummy)=iat
            end if
          end do
        end do
      end do
C 
C *** In order to observe required convention that jat=list(any_i,iat) has jat>iat 
C     modify all exclude lists
C
      do idat=1,ndummy
        idummy=jikdummy(4,idat)
        j=0
        do ineigh=1,listmex(idummy)
          iex=listex(ineigh,idummy)
          if (iex.lt.idummy) then
            kk=listmex(iex)+1
            listmex(iex)=kk
            listex(kk,iex)=idummy
           else
            j=j+1
            listex(j,idummy)=iex
          endif
        end do
        listmex(idummy)=j
      end do
C
C  *** do the same for the 1-4 exclude list
C
      do idat=1,ndummy
        idummy=jikdummy(4,idat)
        j=0
        do ineigh=1,listm14(idummy)
          iex=list14(ineigh,idummy)
          if (iex.lt.idummy) then
            kk=listm14(iex)+1
            listm14(iex)=kk
            list14(kk,iex)=idummy
           else
            j=j+1
            list14(j,idummy)=iex
          endif
        end do
        listm14(idummy)=j
      end do
C
C  *** remove duplicates from the excluded list of dummy atoms
C
      do idat=1,ndummy
        idummy=jikdummy(4,idat)
        do ineigh=1,listmex(idummy)-1
          iex=listex(ineigh,idummy)
          ipos=0
          do kk=ineigh+1,listmex(idummy)
            if (listex(kk,idummy).eq.iex) ipos=kk !duplicate is located at ipos
          end do
          if (ipos.ge.1) then ! remove the atom at ipos
            listmex(idummy)=listmex(idummy)-1
            do kk=ipos,listmex(idummy)
              listex(kk,idummy)=listex(kk+1,idummy)
            end do
          end if 
        end do
      end do
C
C
C  ***  add Lp-O-X-O-Lp to the excluded list
C
      do idat=1,ndummy
        idummy=jikdummy(4,idat)
        iox=jikdummy(2,idat)
        itmp=listmex(iox)
        do ineigh=1,itmp
          do iat=1,ndummy
            jat=jikdummy(2,iat)
            if (listex(ineigh,iox).eq.jat) then
              jdummy=jikdummy(4,iat)
C
C             *** add jdummy to idummy listex if it is not there
C
              HasBeenFound=.false.
              do i=1,listmex(idummy)
                if (listex(i,idummy).eq.jdummy) HasBeenFound=.true.
              end do
              if (.not.HasBeenFound) then
               if (idummy.lt.jdummy) then
                listmex(idummy)=listmex(idummy)+1
                listex(listmex(idummy),idummy)=jdummy
                else
                listmex(jdummy)=listmex(jdummy)+1
                listex(listmex(jdummy),jdummy)=idummy
               end if
D               write(6,*) "idummy,jdummy",idummy,jdummy
              end if
            end if
          end do  ! next iat
        end do    ! next ineigh
      end do      ! next idat
C
C  ***  add Lp-O-X-X-O-Lp to the list14
C
      do idat=1,ndummy
        idummy=jikdummy(4,idat)
        iox=jikdummy(2,idat)
        itmp=listm14(iox)
        do ineigh=1,itmp
          do iat=1,ndummy
            jat=jikdummy(2,iat)
            if (list14(ineigh,iox).eq.jat) then
              jdummy=jikdummy(4,iat)
C
C             *** add jdummy to idummy list14 if it is not there
C
              HasBeenFound=.false.
              do i=1,listm14(idummy)
                if (list14(i,idummy).eq.jdummy) HasBeenFound=.true.
              end do
              if (.not.HasBeenFound) then
               if (idummy.lt.jdummy) then
                listm14(idummy)=listm14(idummy)+1
                list14(listm14(idummy),idummy)=jdummy
                else
                listm14(jdummy)=listm14(jdummy)+1
                list14(listm14(jdummy),jdummy)=idummy
               endif
D               write(6,*) "list14:idummy,jdummy",idummy,jdummy
              end if
            end if
          end do  ! next iat
        end do    ! next ineigh
      end do      ! next idat
C
C  ***  info below is for debugging purposes
C
D     write(6,*) "after correction for Lp"
D     do iat=1,nat
D       write(6,*)listmex(iat),"listex",(listex(j,iat),j=1,listmex(iat))
D     end do
D     write(6,*) "after correction for Lp"
D     do iat=1,nat
D       write(6,*)listm14(iat),"list14",(list14(j,iat),j=1,listm14(iat))
D     end do
C           
      do idummy=1,ndummy
        HasBeenChanged(idummy)=.false.
      end do
      do idummy=1,ndummy
       if (.not.HasBeenChanged(idummy)) then
        HasBeenChanged(idummy)=.true.
        idat=jikdummy(4,idummy)  ! idat is the atom number for the dummy atom
        itype=jikdummy(5,idummy) ! a dummy type
        iat=jikdummy(2,idummy) 
        cci=cdummy(itype)
        if (LpGeomType(itype).eq.1) then ! a single extended force center (imidazole-like)
          if (abs(cci).gt.1.0d-6) then
             write(6,*) "C param should be zero for this Lp type",idat
          end if
          do jdummy=idummy+1,ndummy  ! check for mistakes
            jat=jikdummy(2,jdummy) 
            jtype=jikdummy(5,jdummy) ! a dummy atom type
            if (iat.eq.jat) then     ! two dummy force centers are connected to the same atom
              write(6,*)
     $         "Two extended force centers have the same position",
     $         idummy,jdummy
            end if
          end do
C
C ***  inverse the polarity of the lone pair by changing bend from jat-iat-kat
C ***  to kat-iat-jat
C
        else if (LpGeomType(itype).eq.2.or.LpGeomType(itype).eq.3)then ! ether-like or carbonyl-like lone pair
          if (abs(cdummy(itype)).lt.1.d-6) then  
             write(6,*) "C param should be non-zero for this Lp type",
     $                   iat,idummy
          end if
          if (abs(adummy(itype)).lt.1.d-6) then  
             write(6,*) "A param should be non-zero for this Lp type",
     $                   iat,idummy
          end if
          HasBeenFound=.false.
          do jdummy=idummy+1,ndummy  ! find a twin
            jat=jikdummy(2,jdummy) 
            jtype=jikdummy(5,jdummy) ! a dummy atom type
            if (iat.eq.jat) then     ! two dummy force centers are connected to the same atom
               if (HasBeenFound) then
                 write(6,*)"Found more than one dummy center in a pair"
     $                      ,idummy,jdummy
               end if
               HasBeenFound=.true.
               jtmp=jdummy
            end if      
          end do 
          if (HasBeenFound) then
              jikdummy(1,jtmp)=jikdummy(3,idummy)
              jikdummy(3,jtmp)=jikdummy(1,idummy)
              HasBeenChanged(jtmp)=.true.
           else
            write(6,*) "Dummy",idummy," might not have a pair"
          end if
        end if    !  (itype.eq.2.or.itype.eq.3)
       end if     !  (.not.HasBeenChanged(idummy))
      end do
      return
      end 
