
module not_sure_what_they_are_data
        implicit none
        real(8), parameter :: a_thole = 0.2d0 !! mu-mu damping constant
        integer, parameter :: ibondPF = 11
        real(8), parameter :: P2up = 50000.0d0
        integer, parameter :: Nzslice = 10
        integer, parameter :: maxnay = 900
end module not_sure_what_they_are_data

module spline_interpol_coef
      use sizes_data, only : maxpoints, maxnnb
      implicit none
      real(8),allocatable :: coeffee(:,:,:),coefffee(:,:,:),&
      coefft(:,:,:),coeffft(:,:,:),coeff4(:,:,:),coeff5(:,:,:),&
      coeff1(:,:,:),coeff2(:,:,:),coeff3(:,:,:),coeff6(:,:,:),&
      coeffft1(:,:,:)
      CONTAINS
      subroutine spline_interpol_coef_alloc()
      integer M,N
      M = maxpoints; N = maxnnb
      allocate(coeffee(6,M,N),coefffee(6,M,N))
      allocate(coefft(6,m,n),coeffft(6,m,n))
      allocate(coeff1(3,M,N),coeff2(3,M,N),coeff3(3,M,N),coeff4(3,M,N),coeff5(3,M,N),coeff6(3,M,N))
      allocate(coeffft1(6,M,N))
      coeffee=0.0d0; coefffee=0.0d0; coefft=0.0d0; coeffft=0.0d0
      coeff1=0.0d0;coeff2=0.0d0;coeff3=0.0d0;coeff4=0.0d0;coeff5=0.0d0
      coeff6=0.0d0; coeffft=0.0d0
      end subroutine spline_interpol_coef_alloc
end module spline_interpol_coef
      
module the_subroutines 
      use not_sure_what_they_are_data
      use types_module
      use math_constants
      use physical_constants
      use sizes_data
      use connectivity_data
      use threeD_Ewald_data
      use twoD_Ewald_data
      use other_data
      use the_arrays_data
      use spline_interpol_coef
      use boundaries
      implicit none

      contains
      subroutine allocate_props()
       use sizes_data, only : maxprop
       allocate(sumpr(maxprop),sumsqpr(maxprop),avgpr(maxprop))
       allocate(avgsqpr(maxprop),avgsumpr(maxprop),avgsumsqpr(maxprop))
       allocate(prop(maxprop),stdev(maxprop) )
       allocate(simpr(maxprop),simsqpr(maxprop) )
      end subroutine allocate_props

      subroutine boxes() ! (ortho version)
      implicit none
      integer subbox(maxbox,maxdim3)
      integer boxofatom(3,maxat)
      integer isub(3)
      real(8) rbig,rdimmax,rdimmin,rdim,rcheck,size(3),xij1,xij2,xij3,r2
      integer icheck,ncheck(3),idim2,ibox,iz,iy,ix,iat,kk,ich
      integer nayo,isubz,izz,izbox,isuby,iyy,iybox,isubx,ixx,j,jat
      integer jch,iex,indexex,nayos
      real(8),parameter :: safe = 5.0d0 
      real(8) cut_off, a1
! safe is an extra parameter to break symmetry of the 3D periodicity 
! if needed (for slab geometry)
      if (i_boundary_CTRL==0) then ! SLAB oriented along OZ (x(3,:))
        cut_off = rcut
        a1 = maxval(x(3,1:maxat))-minval(x(3,1:maxat))
        box(3) = max(a1+safe+cut_off,3.0d0*(cut_off+safe),box(3))
      endif
      el(:) = box(:) * 0.5d0
!C
!C     *****determine box size
!C
      rshort = ros + driftmax
      rsbox = rshort + driftmax
      rshort2 = rshort*rshort
      ros2 = ros*ros
      rsbox2 = rsbox*rsbox
      rbig = rcut
      rdimmax = rbig/2.0d0
      do kk=1,3
       rdimmin = box(kk)/nat**(1.0d0/3.0d0)
       if (rdimmax .lt. rdimmin) then
         rdim = rdimmax
        else
        icheck = 2
5       icheck = icheck + 1
        rcheck = rbig/icheck
        if (rcheck .gt. rdimmin) goto 5
        rdim = rbig/(icheck-1)
       end if
       idim(kk) = box(kk)/rdim
       if (idim(kk) .gt. maxdim) then
         write(6,*)' idim > maxdim , reassigning idim to maxdim '
         idim(kk) = maxdim
        end if
       rdim = box(kk)/idim(kk)
       ncheck(kk) = rbig/rdim + 1
       if (idim(kk) .lt. 2*ncheck(kk)+1) then
         if (mod(idim(kk),2) .eq. 0) then
           idim(kk) = idim(kk) + 1
           rdim = box(kk)/idim(kk)
         end if
         ncheck(kk) = (idim(kk)-1)/2
       end if
       size(kk) = box(kk)/idim(kk)
      end do    ! next kk
      idim2 = idim(1)*idim(2)     
      if (printout) write (6,*) 'Boxes : idim = ',idim
!C
!C     *****begin calculations
!C
      ibox = 0
      do ibox = 1,idim(1)*idim(2)*idim(3)
        listsubbox(ibox) = 0
      end do

      do iat = 1,nat
        do kk = 1,3
          isub(kk) = x(kk,iat)/size(kk) + 1
          boxofatom(kk,iat) = isub(kk)
        end do
        ibox = (isub(3) - 1)*idim2 + (isub(2) - 1)*idim(1) + isub(1)
        listsubbox(ibox) = listsubbox(ibox) + 1
        subbox(listsubbox(ibox),ibox) = iat 
      end do

      do iat = 1, nat
        ich = chain(iat)
        nayo = 0  
        nayos = 0  
!C
!C     *****determine possible neighbors from surrounding boxes
!C
        ix = boxofatom(1,iat)
        iy = boxofatom(2,iat)
        iz = boxofatom(3,iat)
        do isubz = iz-ncheck(3),iz+ncheck(3)
          izz = isubz
          if (izz .gt. idim(3)) izz = izz - idim(3)
          if (izz .lt. 1)    izz = izz + idim(3)
          izbox = (izz - 1)*idim2
          do isuby = iy-ncheck(2),iy+ncheck(2)
            iyy = isuby
            if (iyy .gt. idim(2)) iyy = iyy - idim(2)
            if (iyy .lt. 1)    iyy = iyy + idim(2)
            iybox = (iyy - 1)*idim(1)
            do isubx = ix-ncheck(1),ix+ncheck(1)
              ixx = isubx
              if (ixx .gt. idim(1)) ixx = ixx - idim(1)
              if (ixx .lt. 1)    ixx = ixx + idim(1)
              ibox = izbox + iybox + ixx
              do j = 1, listsubbox(ibox)
                jat = subbox(j,ibox)
     if (jat .gt. iat) then !goto 200
                jch = chain(jat)
!C
!C     *****check for exclusion
!C
                if (ich .eq. jch) then
                  do iex = 1,listmex(iat)
                    indexex = listex(iex,iat)
                    if (indexex .eq. jat) goto 200
                  end do
                 else
                  if (.not. inter) goto 200
                end if
!C
!C     *****calculate distance
!C
                xij1 = x(1,jat) - x(1,iat)
                xij2 = x(2,jat) - x(2,iat)
                xij3 = x(3,jat) - x(3,iat)
        if (i_boundary_CTRL == 0 ) then
              if (abs(xij1) .gt. el(1))xij1 = xij1 - dsign(box(1),xij1)
              if (abs(xij2) .gt. el(2))xij2 = xij2 - dsign(box(2),xij2)
        else
              if (abs(xij1) .gt. el(1))xij1 = xij1 - dsign(box(1),xij1)
              if (abs(xij2) .gt. el(2))xij2 = xij2 - dsign(box(2),xij2)
              if (abs(xij3) .gt. el(3))xij3 = xij3 - dsign(box(3),xij3)
        endif

                r2 = xij1*xij1 + xij2*xij2 + xij3*xij3
!C
!C     *****check cutoff
!C
                if (r2 .gt. rcut2) goto 200 
!C
!C     *****record neighbor here
!C
                if (r2 .lt. rsbox2)then
                  nayos = nayos + 1
                  lists(nayos,iat) = jat
                end if
                nayo = nayo + 1
                list(nayo,iat) = jat
200           continue
     endif
              end do
            end do
          end do
        end do
!C
!C     *****record the number of entries for this atom
!C
        listm(iat) = nayo
        listms(iat) = nayos
      end do
      return
      end subroutine boxes

      subroutine checker()
      implicit none
      integer maxn,maxnayrec,maxnayex,maxboxch,maxboxrec,maxboxex
      integer maxdimrec,maxdimex,iat,ix,iy,iz,ibox

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
      maxdimrec = idim(3) + 3
      maxdimex = idim(3) + 5
      maxboxch = 0
      do iz = 1,idim(3)
        do iy = 1,idim(2)
          do ix = 1,idim(1)
            ibox = ibox + 1
        if(listsubbox(ibox) .gt. maxboxch) maxboxch=listsubbox(ibox)
          end do
        end do
      end do
      maxboxrec = int(maxboxch*1.2)
      maxboxex = int(maxboxch*1.25)
      if (maxnay .gt. maxnayex)then
       write(6,*)'current maxnay = ',maxnay,', recommended value = ',maxnayrec
      end if
      if (maxdim .gt. maxdimex)then
        write(6,*)'current maxdim = ',maxdim,', recommended value = ',maxdimrec
      end if
      if (maxbox .gt. maxboxex)then
        write(6,*)'current maxbox = ',maxbox,', recommended value = ',maxboxrec
      end if
      write(6,*)'********** END OF WARNINGS*****************'

      end subroutine checker

      subroutine cload(cc,deltaspline)

      implicit none
      real(8) cc(6,6),deltaspline
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
      end subroutine cload


 subroutine Ewald correct 
! 123 done
! 14 to be done here
      if (lredonefour) then
       do iat = 1, nat 
        itype = atomtype(iat)
        l_distributed_charge_i=is_charge_distributed(itype)
        ci=q(itype)
        neightot = listm14(iat)
        do k  = 1, neightot
          jat = list14(k,iat)
          jtype = atomtype(jat)
          l_distributed_charge_j=is_charge_distributed(jtype)
          l4 = .not.l_distributed_charge_i
          l4 = l4.and.(.not.l_distributed_charge_j)
         if (l4) then
          cj=q(jtype)
          itypee = typee(itype,jtype)
          qq = electrostatic(itypee,1)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL) then
          if (abs(xij1).gt.el(1)) xij1 = xij1-dsign(box(1),xij1)
          if (abs(xij2).gt.el(2)) xij2 = xij2-dsign(box(2),xij2)
          else
          if (abs(xij1).gt.el(1)) xij1 = xij1-dsign(box(1),xij1)
          if (abs(xij2).gt.el(2)) xij2 = xij2-dsign(box(2),xij2)
          if (abs(xij3).gt.el(3)) xij3 = xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          zinv = 1.0d0/z
          rinv = 1.0d0/r
          r3inv = zinv*rinv
          dalphar = 1.0d0 - derfc(alphar)
!C     *****determine energy
          unbdcorrect = unbdcorrect -redfactor*qq*dalphar*rinv
!C     ****determine electrostatic field
          ffcorrect = redfactor*(dalphar*r3inv - twoalphapi*exp(-alphar*alphar)*zinv)
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          elf(1,iat) = elf(1,iat) + f1*cj
          elf(2,iat) = elf(2,iat) + f2*cj
          elf(3,iat) = elf(3,iat) + f3*cj
          elf(1,jat) = elf(1,jat) - f1*ci
          elf(2,jat) = elf(2,jat) - f2*ci
          elf(3,jat) = elf(3,jat) - f3*ci
!C     *****determine force and virial
          ffcorrect = ffcorrect*qq
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          fewald(1,iat) = fewald(1,iat) + f1
          fewald(2,iat) = fewald(2,iat) + f2
          fewald(3,iat) = fewald(3,iat) + f3
          fewald(1,jat) = fewald(1,jat) - f1
          fewald(2,jat) = fewald(2,jat) - f2
          fewald(3,jat) = fewald(3,jat) - f3

          tvirpo(1,1) = tvirpo(1,1) - f1*xij1
          tvirpo(1,2) = tvirpo(1,2) - f1*xij2
          tvirpo(1,3) = tvirpo(1,3) - f1*xij3
          tvirpo(2,1) = tvirpo(2,1) - f2*xij1
          tvirpo(2,2) = tvirpo(2,2) - f2*xij2
          tvirpo(2,3) = tvirpo(2,3) - f2*xij3
          tvirpo(3,1) = tvirpo(3,1) - f3*xij1
          tvirpo(3,2) = tvirpo(3,2) - f3*xij2
          tvirpo(3,3) = tvirpo(3,3) - f3*xij3

        else   !  l4
         write(6,*) 'FATAL ERROR in ewald_correct() at 14interactions;'
         write(6,*) 'In the same molecule there are both distributed'
         write(6,*) 'and point charges ',l_distributed_charge_i,l_distributed_charge_j
        endif   ! l4
        end do
       end do
      endif    ! (lredonefour)

      if (lredQ_mu14) then
       do iat = 1, nat-1
        itype = atomtype(iat)
        l_distributed_charge_i=is_charge_distributed(itype)
        ci=q(itype)
        neightot = listm14(iat)
        do k  = 1, neightot
          jat = list14(k,iat)
          jtype = atomtype(jat)
          l_distributed_charge_j=is_charge_distributed(jtype)
          l4 = .not.l_distributed_charge_i
          l4 = l4.and.(.not.l_distributed_charge_j)
         if (l4) then
          cj=q(jtype)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL) then
          if (abs(xij1).gt.el(1)) xij1 = xij1-dsign(box(1),xij1)
          if (abs(xij2).gt.el(2)) xij2 = xij2-dsign(box(2),xij2)
          else
          if (abs(xij1).gt.el(1)) xij1 = xij1-dsign(box(1),xij1)
          if (abs(xij2).gt.el(2)) xij2 = xij2-dsign(box(2),xij2)
          if (abs(xij3).gt.el(3)) xij3 = xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          zinv = 1.0d0/z
          rinv = 1.0d0/r
          r3inv = zinv*rinv
          dalphar = 1.0d0 - derfc(alphar)

          ffcorrect = redQmufactor*(dalphar*r3inv- twoalphapi*exp(-alphar*alphar)*zinv)
          f1 = ffcorrect*xij1
          f2 = ffcorrect*xij2
          f3 = ffcorrect*xij3
          elf(1,iat) = elf(1,iat) + f1*cj
          elf(2,iat) = elf(2,iat) + f2*cj
          elf(3,iat) = elf(3,iat) + f3*cj
          elf(1,jat) = elf(1,jat) - f1*ci
          elf(2,jat) = elf(2,jat) - f2*ci
          elf(3,jat) = elf(3,jat) - f3*ci

        else ! l4
         write(6,*) 'FATAL ERROR in ewald_correct() at lredQ_mu14'
         write(6,*) 'In the same molecule there are both distributed'
         write(6,*) 'and point charges ',l_distributed_charge_i,l_distributed_charge_j

        endif !(l4) 
        end do
       end do
      endif    ! (lredQ_mu14)

      print*, 'entered in corrected ewald with energy =',unbd*temp_cvt_en
      vir = vir + unbdcorrect
      unbd = unbd + unbdcorrect - ewself
      print*, 'corrected ewald=',unbdcorrect*temp_cvt_en, ewself*temp_cvt_en,&
      ( unbdcorrect - ewself ) * temp_cvt_en , ' > ',unbd*temp_cvt_en

      do iat = 1,nat
        do kk = 1,3
          f(kk,iat) = f(kk,iat) + fewald(kk,iat)
        end do
      end do

      end subroutine ewald_correct

      subroutine diffusion()
      implicit none
      real(8) delxup(3,maxat)
      integer iat,ich,kk
      real(8) disp,delxmax

      delxmax = 0.0d0
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
      spol = 0.00d0
      do iat = 1,nat
        do kk = 1,3
          spol = spol + sumdel(kk,iat)*sumdel(kk,iat)
        end do
      end do
      end subroutine diffusion

      double precision function dot(a,b)
      implicit none
      real(8) a(3),b(3)
      dot = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
      end   function dot

      double precision FUNCTION funct(z,innb,ifunct)

      implicit none
      integer ifunct,innb
      real(8) z
!C
!C     *****local variables
!C
      integer imap
      real(8) tolz,r,fupper,flower,pforce,pintforce,a,b,c
      real(8) eforce,edforce,eintforce,aa
      integer i,j,k
      logical l1,l2,l3,l4,l_distributed_charge_i,l_distributed_charge_j

      funct = 0.0d0
      tolz = 1.0d-8
      r = dsqrt(z)
      alpha = alphai     ! alpha should be divided by box in fort.25 <ortho>
      select case (ifunct)
      case (1)
         i = map_inverse_innb(innb)%i
         j = map_inverse_innb(innb)%j
!         print*, 'map_inverse_innb=',map_inverse_innb
         l_distributed_charge_i=is_charge_distributed(i)
         l_distributed_charge_j=is_charge_distributed(j)
         l1 = l_distributed_charge_i.and.l_distributed_charge_j
         l2 = l_distributed_charge_i.and.(.not.l_distributed_charge_j)
         l3 = .not.l_distributed_charge_i.and.l_distributed_charge_j
         l4 = (.not.l_distributed_charge_i)
         l4 = l4.and.(.not.l_distributed_charge_j)
          if (l1) then  ! gaussian gaussian
            funct = derfc(Ew_beta*r)*electrostatic(innb,1)/r
          elseif (l2) then !gaussian  puctual
            funct = derfc(Ew_gamma*r)*electrostatic(innb,1)/r
          elseif (l3) then ! puctual gaussian
            funct = derfc(Ew_gamma*r)*electrostatic(innb,1)/r
          elseif (l4) then ! punctual punctual
            funct = derfc(alpha*r)*electrostatic(innb,1)/r
          else  ! LOGICAL ERROR if hit here
           print*, 'IMPOSIBLE CASE IN interch select case l1 l2 ..l4'
             STOP
          endif
      case (2)
        if (.not. tbpol)return
        if (abs(rop - r) .lt. tolz)then
          funct = 0.0d0
          return
        end if
        fupper = pforce(r)
        flower = pforce(rop)
        funct=pintforce(rop,flower,r,fupper)*electrostatic(innb,2)
! I will not be dealing with this case (for now)
      case (3)
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
      case default
        print*, 'ERROR in funct; ifunct out of range ',ifunct
        STOP
      end select

      end function funct

      double precision FUNCTION functd(z,innb,ifunct)

      implicit none
      integer ifunct,innb
      real(8) z
      integer imap
      real(8) tolz,r,pforce,a,b,c
      real(8) eforce,edforce,eintforce,aa,bb
      logical l1,l2,l3,l4,l_distributed_charge_i,l_distributed_charge_j
      integer i,j

      functd = 0.0d0
      tolz = 1.0d-8
      r = dsqrt(z)

      select case (ifunct) 
      case (1)
         i = map_inverse_innb(innb)%i
         j = map_inverse_innb(innb)%j
         l_distributed_charge_i=is_charge_distributed(i)
         l_distributed_charge_j=is_charge_distributed(j)
         l1 = l_distributed_charge_i.and.l_distributed_charge_j
         l2 = l_distributed_charge_i.and.(.not.l_distributed_charge_j)
         l3 = .not.l_distributed_charge_i.and.l_distributed_charge_j
         l4 = (.not.l_distributed_charge_i)
         l4 = l4.and.(.not.l_distributed_charge_j)
         if (l1) then  ! gaussian gaussian
!           AA = derfc(Ew_beta*r)/(2.0d0*r**3)
!           BB = Ew_beta*dexp(-(Ew_beta*r)**2)/(r*r*rootpi)
            AA = 0.0d0
            BB = 0.0d0  ! the easiest way to fix the electrodes
          elseif (l2) then !gaussian  puctual
            AA = derfc(Ew_gamma*r)/(2.0d0*r**3)
            BB = Ew_gamma*dexp(-(Ew_gamma*r)**2)/(r*r*rootpi)
          elseif (l3) then ! puctual gaussian
            AA = derfc(Ew_gamma*r)/(2.0d0*r**3)
            BB = Ew_gamma*dexp(-(Ew_gamma*r)**2)/(r*r*rootpi)
          elseif (l4) then ! punctual punctual
            AA = derfc(alpha*r)/(2.0d0*r**3)
            BB = alpha*dexp(-(alpha*r)**2)/(r*r*rootpi)
          else  ! LOGICAL ERROR if hit here
           print*, 'IMPOSIBLE CASE IN interch select case l1 l2 ..l4'
             STOP
          endif
          functd = - electrostatic(innb,1)*(AA + BB)
      case (2)
        if (.not. tbpol)return
        functd = -pforce(r)*electrostatic(innb,2)/(2.0d0*r)
      case (3)
        imap = map(innb)
        a = nonbonded(1,imap)
        b = nonbonded(2,imap)
        c = nonbonded(3,imap)
        if (b .lt. 1.0d-6)then
          functd = -6.0d0*a/z**7 + 3.0d0*c/z**4
         else
          aa = 0.5d-4*(12.0d0/b)**12
          functd = -a*b*dexp(-b*dsqrt(z))/(2.0d0*dsqrt(z))+ 3.0d0*c/z**4 - 6.0d0*aa/z**7
        end if
      case default
       print*, 'ERROR in functd ; ifunct out of range',ifunct
       STOP
      end select

      end function functd

      double precision FUNCTION functdd(z,innb,ifunct)

      implicit none
      integer ifunct,innb
      real(8) z
      integer imap
      real(8) tolz,r,endder,pdforce,pforce,a,b,c,eff,edf
      real(8) eforce,edforce,eintforce,aa,qq
      integer i,j,k
      logical l1,l2,l3,l4,l_distributed_charge_i,l_distributed_charge_j
      

      functdd = 0.0d0
      tolz = 1.0d-8
      r = dsqrt(z)

      select case (ifunct) 
      case(1)
         qq = electrostatic(innb,1)
         i = map_inverse_innb(innb)%i
         j = map_inverse_innb(innb)%j
         l_distributed_charge_i=is_charge_distributed(i)
         l_distributed_charge_j=is_charge_distributed(j)
         l1 = l_distributed_charge_i.and.l_distributed_charge_j
         l2 = l_distributed_charge_i.and.(.not.l_distributed_charge_j)
         l3 = .not.l_distributed_charge_i.and.l_distributed_charge_j
         l4 = (.not.l_distributed_charge_i)
         l4 = l4.and.(.not.l_distributed_charge_j)
          if (l1) then  ! gaussian gaussian
!             functdd = fifi_ele(Ew_beta,r,qq)
             functdd = 0.0d0
          elseif (l2) then !gaussian  puctual
             functdd = fifi_ele(Ew_gamma,r,qq)
          elseif (l3) then ! puctual gaussian
             functdd = fifi_ele(Ew_gamma,r,qq)
          elseif (l4) then ! punctual punctual
             functdd = fifi_ele(alpha,r,qq)
          else  ! LOGICAL ERROR if hit here
           print*, 'IMPOSIBLE CASE IN interch select case l1 l2 ..l4'
             STOP
          endif

      case(2)
        if (.not. tbpol)return
        endder = (1.0d0/(4.0d0*r))*(pforce(r)/(r*r) - pdforce(r)/r)
        functdd = electrostatic(innb,2)*endder
      case(3)
        imap = map(innb)
        a = nonbonded(1,imap)
        b = nonbonded(2,imap)
        c = nonbonded(3,imap)
        if (b .lt. 1.0d-6)then
          functdd = 42.0d0*a/z**8 -12.0d0*c/z**5
         else
          aa = 0.5d-4*(12.0d0/b)**12
          functdd = 0.25d0*a*b*exp(-b*sqrt(z))*(z**(-1.5d0) + b*z**(-1))-12.0d0*c/z**5 + 42.0d0*aa/z**8
        end if
      case default
         print*, 'ERROR in functdd ; ifunct out of range',ifunct
         STOP
      end select


      CONTAINS
      function fifi_ele(alpha,r,qq) result (results)
        real(8) alpha,r, qq, results
        real(8) eff, edf, endder
        eff = derfc(alpha*r)/(r*r)+ (2.0d0*alpha/rootpi)*dexp(-(alpha*r)**2)/r
        edf = -2.0d0*derfc(alpha*r)/r**3-(4.0d0*alpha/rootpi)*exp(-(alpha*r)**2)*(1/r**2 + alpha**2)
        endder = (1.0d0/(4.0d0*r))*(eff/(r*r) - edf/r)
        results = qq*endder ! electrostatic(innb,1)
      end function fifi_ele
      end function functdd

      subroutine getkin()

      implicit none
      integer iat,kk,jj
      real(8) ratio,volinv,dndof,presold,pvir,mv2tot
      real(8) stresske(3,3),stresspo(3,3)
      real(8) stressold(3,3)
      real(8) etrunc

      SAVE presold
      SAVE stressold

      mv2tot = 0.0d0
      do iat = 1,nat
        mv2tot = mv2tot + (v(1,iat)*v(1,iat)+v(2,iat)*v(2,iat)+ v(3,iat)*v(3,iat)) * mass(iat)
      end do

      kinen = 0.5d0 * mv2tot
      temp   = mv2tot/(dble(ndof) * Red_Boltzmann_constant)
      ekin2 = 0.0d0
      do kk = 1,3
        akin2(kk) = 0.0d0
        do iat = 1,nat
          akin2(kk) = akin2(kk) + mass(iat)*v(kk,iat)*v(kk,iat)
        end do
        ekin2 = ekin2 + akin2(kk)
      end do
      if (fixtemp) then 
        ratio = sqrt(tstart/temp)
        v(:,:) = v(:,:) * ratio
        temp = tstart
      end if
!C     *****Pressure and stress tensor calculations
      if (newpress) then
        volinv =1.0d0/(box(1)*box(2)*box(3))
        dndof = dble(ndof-3)/dble(ndof)
        etrunc = esumninj*volinv
!C     *****pressure terms
        if (lprofile) then
           ptrunc=(stresstr(1,1)+stresstr(2,2)+stresstr(3,3))/3.d0 !pressure truncation is calc in interch()
         else
           ptrunc = psumninj*volinv*volinv
        endif
        if (printout) then
          write(6,*) 'isotropic correction energy', etrunc
          write(6,*) 'isotropic correction pressure',psumninj*volinv*volinv
          write(6,'("StressTr(anizotr)=",3F12.2)') stresstr(1,1),stresstr(2,2),stresstr(3,3)
        endif

        pke = dble(ndof-3)*temp*volinv
        pvir  = vir*volinv
        pres  =  pke + pvir + ptrunc 
!C
!C     *****stress terms
!C
        prtkine(:,:) = 0.0d0 ; stress(:,:) = 0.0d0
        do iat = 1,nat
          do kk = 1,3
            do jj = 1,3
              prtkine(kk,jj) = prtkine(kk,jj) + mass(iat)*v(kk,iat)*v(jj,iat)
            end do
          end do
        end do

        if (.not.lprofile) then
        stresstr(:,:) = 0.0d0
        stresstr(1,1) = ptrunc; stresstr(2,2) = ptrunc; stresstr(3,3) = ptrunc
        end if

        do kk = 1,3
          do jj = 1,3
            stresske(kk,jj) = prtkine(kk,jj)*dndof*16.3884d-3*volinv
            stresspo(kk,jj) = tvirpo(kk,jj)*6.8571d4*volinv
            stress(kk,jj) = stresske(kk,jj) + stresspo(kk,jj) + stresstr(kk,jj)
          end do
        end do
!C
!C       prt - stress without kinetic part
!C
        prt(3) = stresspo(3,3) + stresstr(3,3)
        prt(2) = stresspo(2,2) + stresstr(2,2)
        prt(1) = stresspo(1,1) + stresstr(1,1)

        presold = pres
        stressold(:,:) = stress(:,:)
      else 
        pres = presold
        stress(:,:) = stressold(:,:)
      end if

      end  subroutine getkin

      subroutine hydriva(xref,dhdxa)

      implicit none
      real(8) dhdxa(3,3,3,*),xref(3,maxat) ! shared?

      integer itype,im,icenter,icenter_m1,icenter_p1,kk,ic,icx
      real(8) avect(3),bvect(3),dpdx(3),dvdx(3),bxa(3),aminusb(3)
      real(8) vect1(3),vect2(3),vect3(3),vect4(3),vect5(3)
      real(8) p(3,3,3),vectderiv(3,2,3,3),ivdirect(3)
      real(8) dp,dv,dpi,dvi,dpi3,dvi3,totalp,totalv,fix,temph

!C
!C     *****vectderiv
!C
      data itype/2/
      data vectderiv /-1.0, 0.0, 0.0, 0.0, 0.0, 0.0,&
                      1.0, 0.0, 0.0,-1.0, 0.0, 0.0,&
                      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,&
                      0.0,-1.0, 0.0, 0.0, 0.0, 0.0,&
                      0.0, 1.0, 0.0, 0.0,-1.0, 0.0,&
                      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,&
                      0.0, 0.0,-1.0, 0.0, 0.0, 0.0,&
                      0.0, 0.0, 1.0, 0.0, 0.0,-1.0,&
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.0/
      data ivdirect / -1, 1, -1/
!C
!C     *****calculate vectors
!C
      do im = 1,numaromatic
        ivdirect(2) = idpar(im)
        icenter = iaromatic(2,im)
        icenter_m1 = iaromatic(1,im)
        icenter_p1 = iaromatic(3,im)
        do kk = 1,3
          avect(kk) = xref(kk,icenter)    - xref(kk,icenter_m1)
          bvect(kk) = xref(kk,icenter_p1) - xref(kk,icenter)
        end do
!C
!C     *****calculate bxa,aminusb,dp and dv
!C
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

        dp = sqrt(dp)
        dpi = 1.0d0/dp
        dv = sqrt(dv)
        dvi = 1.0d0/dv
        dpi3 = dpi**3
        dvi3 = dvi**3
!C
!C     *****calculate p and v vectors for each derivative
!C
        do ic = 1,3   ! each carbon
          do icx = 1,3 ! each coordinate
            vect1(1) = bvect(2)*vectderiv(3,1,ic,icx) - bvect(3)*vectderiv(2,1,ic,icx)
            vect1(2) = -(bvect(1)*vectderiv(3,1,ic,icx) - bvect(3)*vectderiv(1,1,ic,icx))
            vect1(3) = bvect(1)*vectderiv(2,1,ic,icx) - bvect(2)*vectderiv(1,1,ic,icx)
            vect2(1) = vectderiv(2,2,ic,icx)*avect(3) - vectderiv(3,2,ic,icx)*avect(2)
            vect2(2) = -(vectderiv(1,2,ic,icx)*avect(3) - vectderiv(3,2,ic,icx)*avect(1))
            vect2(3) = vectderiv(1,2,ic,icx)*avect(2) - vectderiv(2,2,ic,icx)*avect(1)
            totalp = 0.d0
            totalv = 0.d0
            vect3(1) = vect2(1) + vect1(1)
            vect5(1) = vectderiv(1,1,ic,icx) - vectderiv(1,2,ic,icx)
            totalp = totalp + bxa(1)*vect3(1)
            totalv = totalv + aminusb(1)*vect5(1)
            vect3(2) = vect2(2) + vect1(2)
            vect5(2) = vectderiv(2,1,ic,icx) - vectderiv(2,2,ic,icx)
            totalp = totalp + bxa(2)*vect3(2)
            totalv = totalv + aminusb(2)*vect5(2)
            vect3(3) = vect2(3) + vect1(3)
            vect5(3) = vectderiv(3,1,ic,icx) - vectderiv(3,2,ic,icx)
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

      end   subroutine hydriva

      subroutine initialize()

      implicit none
      integer ibond,iat,jat,idtype,itype,idef,ideftype,idbnum,kk
      integer iatdef,jatdef
      real(8) defk
      real(8) x1(3,maxat),diff,ci,rmass,pvir,volinv,dndof,mv2tot
      double precision etrunc
      logical arombond


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

      write(65,'(a72)')clinec
      write(65,'(a42)')'#                         Running Averages'
      write(65,'(a1)')'#'
      write(65,'(a39,a42)')'#     Time    Temp     Pressure     Box',&
                     '        Total E    Hamiltonian, BoxX, BoxY'
      write(65,'(a39,a42)')'#      fs       K         Atm       Ang',&
                     '        Kcal/mol    Kcal/mol    Ang    Ang'
      write(65,'(a72)')clinec
      close (65)

      el(:) = box(:) * 0.5d0
      boxini(:) = box(:)

      CALL setup()
      CALL exclude()
      CALL DoDummyInit()
      CALL boxes()
      CALL spline()
      intzs0 = int(zs(0)) - 1


      call interch()
STOP


!C
!C     *** is a bond a dummy bond?
!C
      do ibond = 1,nbonds
        iat = bonds(1,ibond)
        jat = bonds(2,ibond)
        if (ldummy(iat).or.ldummy(jat)) then
          ldummybond(ibond)=.true.
         else
          ldummybond(ibond)=.false.
        endif
     end do

      if (constrain)then
        nbcon = 0
        do ibond = 1,nbonds
          iat = bonds(1,ibond)
          jat = bonds(2,ibond)
          itype = bonds(3,ibond)
!C
!C     *****if hydrogen positions are constrained, don't include
!C            bond constraint
!C
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

          if (ldummybond(ibond)) arombond=.true.! do not constrain dummy bonds
          if (itype.eq.ibondPF) arombond=.true.! do not constrain PF6, BF4

          if (.not. arombond.and.(chain(iat).lt.iChainSurf)) then
            nbcon = nbcon + 1
            bondcon(1,nbcon) = iat
            bondcon(2,nbcon) = jat
            bondcon(3,nbcon) = itype
            massred(nbcon) = 0.5d0*mass(iat)*mass(jat)/(mass(iat)+mass(jat))
            d2(nbcon) = stretch(2,itype)*stretch(2,itype)
          end if
        end do   ! bonds

        numaromatic = 0
        do idef = 1,ndeforms
          ideftype = deforms(5,idef)
          defk = deform(ideftype)
          if ( defk .lt. 1.0d-5) then

!C
!C     *****type 1 = ca-ca-ca-*ha
!C
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

        nconst = nbcon + 3*numaromatic
        write (6,*) 'nbcon = ',nbcon
        write (6,*) 'numaromatic = ',numaromatic
        write (6,*) 'nconst = ',nconst

        CALL shake(x,x,x)
        do  iat = 1,nat
          do kk = 1,3

             x1(kk,iat) = x(kk,iat) + v(kk,iat)*delt
             if (x1(kk,iat).gt.box(kk)) x1(kk,iat)=x1(kk,iat)-box(kk)
             if (x1(kk,iat).lt.0.0) x1(kk,iat) = x1(kk,iat)+ box(kk)
          end do
        end do
        CALL shake(x,x1,x1)
        do iat = 1,nat
          do kk = 1,3
            diff = x1(kk,iat) - x(kk,iat)
            if (abs(diff).gt.el(kk)) diff=diff-dsign(box(kk),diff)
            v(kk,iat) = diff/delt
          end do
        end do
      end if
!C
!C     *****Energy block
!C
      do iat = 1,nat
        do kk = 1,3
          f(kk,iat) = 0.0d0
        end do
      end do

      unbd = 0.0d0

      CALL en2cen()
      CALL en3cen()
      CALL en4cen()
      CALL improper()
      print*, 'after en2cen en3cen en4cen improper'
!C
!C     *****self correction
!C
        call get_ewald_self_correct

        CALL DoDummyCoords()   ! update positions of dummy atoms
        print*, 'after DoDummyCoords'
        print*, 'lewald=',lewald
        call Fourier_Ewald_driver
        print*, 'reciprocal and correct'
        CALL interch()
        print*, 'after interch'
        CALL DoDummy(f)   ! update positions of dummy atoms
        print*, 'after DoDummy'

      poten = ebond + ebend + etort + eopbs + unbd
        print*, 'poten=',poten*temp_cvt_en
!C      write(6,'("ebond",F12.4)') ebond
!C      write(6,'("ebend",F12.4)') ebend
!C      write(6,'("etort",F12.4)') etort
!C      write(6,'("unbd ",F12.4)') unbd
!C
!C     *****End of energy block
!C
!C
!C     ******no. of degrees of freedom
!C
      natreal=0
      do iat=1,nat
        if (.not.ldummy(iat)) then
          natreal=natreal+1
         else
          v(1,iat)=0.0d0
          v(2,iat)=0.0d0
          v(3,iat)=0.0d0
        end if
      end do
      ndof = 3*natreal - nconst
      write(6,*)'Total number of degrees of freedom = ',ndof
!C
!C     *****calculate initial kinetic energy
!C
      mv2tot = 0.0d0
      do iat = 1,nat
        mv2tot  = mv2tot + (v(1,iat)*v(1,iat)+v(2,iat)*v(2,iat)+v(3,iat)*v(3,iat)) * mass(iat)
      end do

      kinen = 0.5d0 * mv2tot 
      write (6,*)'Initial kinetic energy = ',kinen*temp_cvt_en
      toten = kinen + poten
      temp   = mv2tot/(dble(ndof) * Red_Boltzmann_constant )
      volinv =1.0d0/(box(1)*box(2)*box(3))
      dndof = dble(ndof-3)/dble(ndof)
      etrunc = esumninj*volinv
      write(6,*) 'etrunc=',etrunc*temp_cvt_en
!C
!C     *****pressure terms
!C
      ptrunc = psumninj*volinv*volinv
      pke = dble(ndof-3)*temp*volinv
      pvir  = vir*volinv
      pres     =  pke + pvir + ptrunc
!C 
!C     *****initialize properties
!C
      do iat = 1,nat
          rmass = massinv(iat)
        do kk = 1,3
          accel(kk,iat)     = f(kk,iat)*rmass
          xn(kk,iat)    = x(kk,iat) - v(kk,iat)*delt + 0.5d0*accel(kk,iat)*delt*delt
          if (xn(kk,iat) .gt. box(kk))  xn(kk,iat) = xn(kk,iat) - box(kk)
          if (xn(kk,iat) .lt. 0.0d0)  xn(kk,iat) = xn(kk,iat) + box(kk)
          xn_1(kk,iat)  = x(kk,iat) - 2.*v(kk,iat)*delt + 2.0*accel(kk,iat)*delt*delt
          if (xn_1(kk,iat) .gt. box(kk)) xn_1(kk,iat) = xn_1(kk,iat) - box(kk)
          if (xn_1(kk,iat) .lt. 0.0) xn_1(kk,iat) = xn_1(kk,iat) + box(kk)
        end do
      end do

      end subroutine initialize

      subroutine integrator()
 
      implicit none
      integer kinterch,istep,iat,kk,irat,jj
      real(8) rmass,aa,aa2,arg2,poly,bb,scale,diff
      integer kp2up
      logical newpresstmp
      real(8), allocatable :: x2(:,:),fref(:,:),fs(:,:)

      allocate(x2(3,maxat))
      allocate(fref(3,maxat))
      allocate(fs(3,maxat))

      if (nvt) CALL nhcint()
      if (npt) CALL nptint()
      if (nve) CALL getkin()

      kount = kount + 1
!C
!C  ***  If update Surface interactions
!C
      kp2up = int(P2up/(delt*multibig)+0.5d0)
      if (lprofile) then
        if(mod(kount,kp2up) .eq. 0) CALL surftrunc()
      endif

      kinterch = 0

      do istep = 1,multibig

        do iat = 1, nat
          rmass = massinv(iat)
          do kk = 1,3
            v(kk,iat) = v(kk,iat) + 0.5d0*delt*fnow(kk,iat)*rmass
          end do
        end do
!C
!C     *****update particle positions
!C
        if (nvt .or. nve) then
          do iat = 1,nat
            do kk = 1,3
              x(kk,iat) = x(kk,iat) + v(kk,iat)*delt
            end do
          end do
        end if

        if (npt) then
          do kk = 1,3
            aa = dexp(delt*0.5d0*vlogv(kk))
            aa2 = aa*aa
            arg2 = (vlogv(kk)*delt*0.5d0)*(vlogv(kk)*delt*0.5d0)
            poly = (((e8*arg2+e6)*arg2+e4)*arg2+e2)*arg2+1.d0
            bb = aa*poly*delt
            do iat = 1,nat
              x(kk,iat) = x(kk,iat)*aa2 + v(kk,iat)*bb
            end do
            xlogv(kk) = xlogv(kk) + vlogv(kk)*delt
            scale = exp(vlogv(kk)*delt)
            box(kk) = box(kk) * scale
          end do
        end if
!C
!C     *****apply periodic boundary conditions
!C
        do iat= 1,nat
          do kk = 1,3
            if (kk==3.and.i_boundary_CTRL==0)then
            else 
            if (x(kk,iat).gt.box(kk)) x(kk,iat)=x(kk,iat)-box(kk)
            if (x(kk,iat).lt.0.0) x(kk,iat) = x(kk,iat) + box(kk)
            endif
          end do
        end do 
        CALL DoDummyCoords()   ! update positions of dummy atoms
        el(:) = box(:) * 0.5d0

        do iat = 1,nat
          do kk = 1,3
            f(kk,iat) = 0.0d0
            fshort(kk,iat) = 0.0d0
          end do
        end do

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

        CALL en2cen()
        CALL en3cen()
        CALL improper()

        do iat = 1,nat
          do kk = 1,3
            fref(kk,iat)=f(kk,iat)
            fnow(kk,iat) = fref(kk,iat)
          end do
        end do
      
         if ((mod(istep,multimed).eq.0).and. (istep.ne.multibig)) then
!C
!C     *****Energy block

           CALL en4cen()
           CALL interchs()
           CALL DoDummy(f)
!C
!C     *****End of energy block



           do iat = 1,nat
             do kk = 1,3
               fnow(kk,iat) = fref(kk,iat) + multimed*(f(kk,iat)-fref(kk,iat))
             end do
           end do
         endif   ! medium timestep 
          
         if (istep .eq. multibig)then
!C
!C     *****Energy block
!C
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

           unbd = 0.0d0

           CALL Fourier_Ewald_driver
           CALL interch()
           CALL DoDummy(fshort)
           CALL DoDummy(f)

           if (newpress) then
             vir=vir+virdummy
             do kk = 1,3
               do jj=1,3
                 tvirpo(kk,jj)=tvirpo(kk,jj)+tvirdummy(kk,jj)
               end do
             end do
           endif

           do iat = 1,nat
            do kk = 1,3
              fnow(kk,iat) = fref(kk,iat)+multimed*(fshort(kk,iat)-fref(kk,iat))
            end do
           end do

           if (mod(kount,knben) .eq. 0) CALL nbenergy()
!C
!C     *****End of energy block
!C
           if (constrain .and. newpress)then
             do iat = 1,nat
              rmass = massinv(iat)
              do kk =1 ,3
                x2(kk,iat) = x(kk,iat)+delt*v(kk,iat)+(delt*delt)*f(kk,iat)*rmass
                if (x2(kk,iat).gt.box(kk))x2(kk,iat)=x2(kk,iat)-box(kk)
                if (x2(kk,iat).lt.0.0) x2(kk,iat) = x2(kk,iat)+ box(kk)
              end do
             end do
             CALL shake(x,x2,x2)
           end if

          do iat = 1,nat
            do kk = 1,3
              fnow(kk,iat)=fnow(kk,iat)+multibig*(f(kk,iat)-fshort(kk,iat))
            end do
          end do
        end if   !(istep .eq. multibig) 

        if (constrain)then
          do iat = 1,nat
            rmass = massinv(iat)
            do kk =1 ,3
              x2(kk,iat) = x(kk,iat)+delt*v(kk,iat)+(delt*delt)*fnow(kk,iat)*rmass
              if (x2(kk,iat).gt.box(kk))x2(kk,iat)=x2(kk,iat)-box(kk)
              if (x2(kk,iat).lt. 0.0) x2(kk,iat)=x2(kk,iat)+box(kk)
            end do
          end do

          newpresstmp=newpress
          newpress=.false.
          CALL shake(x,x2,x2)
          newpress=newpresstmp

          do iat = 1,nat
            do kk = 1,3
              diff = x2(kk,iat) - x(kk,iat)
              if (abs(diff).gt.el(kk)) diff=diff-dsign(box(kk),diff)
              fnow(kk,iat) =(diff - delt*v(kk,iat))*mass(iat)/(delt*delt)
            end do
          end do
        end if

        do iat = 1, nat
          rmass = massinv(iat)
          do kk = 1,3
            v(kk,iat) = v(kk,iat) + 0.5d0*delt*fnow(kk,iat)*rmass
          end do
        end do
      end do

      do iat = 1,nat
        do kk = 1,3
          delx(kk,iat) = x(kk,iat) - xn_1(kk,iat)
          if (abs(delx(kk,iat)).gt.el(kk)) delx(kk,iat) =delx(kk,iat) - dsign(box(kk),delx(kk,iat))
          xn_1(kk,iat) = x(kk,iat)
        end do
      end do

      if (nvt) CALL nhcint()
      if (npt) CALL nptint()
      if (nve) CALL getkin()

      poten = ebond + ebend + etort + eopbs + unbd

      CALL results()

      deallocate(x2)
      deallocate(fref)
      deallocate(fs)

      end subroutine integrator

      subroutine interch()

      implicit none
      integer iat,ichain,itype,neightot,k,jat,jtype,itypee
      integer ipiece
      real(8) deltainv,z,zpiece,zpiece2,zpiece3,zpiece4,zpiece5,ff
      real(8) f1,f2,f3,ffe,xij1,xij2,xij3
      integer nneigh,iter,iflag,kk,jj,kx,ky,kz,kmag2,klimit2
      real(8) kxf,kyf,kzf
      real(8) tau,tauval
      real(8) dtau,dtauval
      real(8) rxij,ryij,rzij,pjr,pir,diff2,uind,ci,cj,r5inv
      real(8) qq,dx,dy,dz,pipj,p1,p2,p3,p4,p5,ee1,ee2,ee3,ee4,ee5
      real(8) twopifact,fouralpha2inv,factor,rkmag2,btens,akvec
      real(8) dotik,sfh,front(3),pk,cost,sint
      real(8) cossump,sinsump,cossumx,sinsumx,prefact1,prefact2
      real(8) twoalphapi,facti,factj,elfact,virfact
      real(8) r,rinv,zinv,r3inv,alphar,fact,derfc,derf,tfact
      real(8) ee6,gtaper,staper,expc1r3val,three_c1r2
      real(8) virind,tvirind(3,3),tmp
      real(8) lamda3,lamda5,dlamda3,dlamda5,c1r3,p6
      real(8) virx,virz,unbd_corr,subh,box3inv
      integer izslice, i1
      logical l_distributed_charge_i,l_distributed_charge_j,l1,l2,l3,l4
      real(8) gammar, betar
      real(8) ffx,ffy,ffz
      real(8) local_potential, En
      real(8) c1(maxcharges,maxcharges)


      real(8),allocatable :: find(:,:),px1old(:),px2old(:),px3old(:)
      real(8),allocatable :: py1old(:),py2old(:),py3old(:)
      real(8),allocatable :: pz1old(:),pz2old(:),pz3old(:)
      real(8), allocatable :: expc1r3(:,:)
      real(8),allocatable :: pxnew(:),pynew(:),pznew(:)
      real(8), allocatable :: xun(:,:),tx(:),ty(:),tz(:)
      real(8), allocatable :: costermp(:),sintermp(:),costermx(:)
      real(8), allocatable :: sintermx(:)
      real(8), allocatable :: xx(:),yy(:),zz(:)

      

      integer, allocatable :: listpol(:,:),listmpol(:)
      real(8), allocatable :: rx(:,:),ry(:,:),rz(:,:),rr(:,:),rrinv(:,:)
      allocate(listpol(maxnay,maxat))
      allocate(listmpol(maxat))
      allocate(rx(maxnay,maxat))
      allocate(ry(maxnay,maxat))
      allocate(rz(maxnay,maxat))
      allocate(rr(maxnay,maxat))
      allocate(rrinv(maxnay,maxat))
      allocate(find(3,maxat))
      allocate(px1old(maxat))
      allocate(px2old(maxat))
      allocate(px3old(maxat))
      allocate(py1old(maxat))
      allocate(py2old(maxat))
      allocate(py3old(maxat))
      allocate(pz1old(maxat))
      allocate(pz2old(maxat))
      allocate(pz3old(maxat))
      allocate(expc1r3(maxnay,maxat))
      allocate(pxnew(maxat))
      allocate(pynew(maxat))
      allocate(pznew(maxat))
      allocate(xun(3,maxat))
      allocate( tx(maxat),ty(maxat),tz(maxat))
      allocate(costermp(maxat),sintermp(maxat))
      allocate(costermx(maxat),sintermx(maxat))

      print*, 'GOT in INTERCH !!!!!!!!!!'

      unbde=0.0d0
      virind = 0.0d0
      tvirind(:,:) = 0.0d0
      deltainv = 1.0d0/deltaspline
      el(:) = box(:) * 0.5d0
      twoalphapi = 2.0d0*alpha/rootpi
      Ew_twobetapi = 2.0d0*Ew_beta/rootpi
      Ew_twogammapi = 2.0d0*Ew_gamma/rootpi

!     initialize c1 array for damping of polarization

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

!     !*****begin calculation of non-bonded and electrostatic energies
!     !     and forces

      sim_cel(1) = box(1) ; sim_cel(5) = box(2) ; sim_cel(9) = box(3)

      print*, 'start interch cycle with unbd=',unbd*temp_cvt_en
      vdw_en = 0.0d0
      
      do iat = 1, nat - 1
        nneigh = 0
        ichain = chain(iat)
        itype  = atomtype(iat)
        neightot = listm(iat)
        l_distributed_charge_i = is_charge_distributed(itype)
        allocate(xx(neightot),yy(neightot),zz(neightot),rv_sq(neightot))
        i1 = 0
        do k =  1, neightot
          i1 = i1 + 1
          jat =  list(k,iat)
          xx(i1) = x(1,jat) - x(1,iat)
          yy(i1) = x(2,jat) - x(2,iat)
          zz(i1) = x(3,jat) - x(3,iat)
        enddo
        call periodic_images(2,xx,yy,zz,sim_cel)
        rv_sq(:) = xx(:)*xx(:) + yy(:)*yy(:) + zz(:)*zz(:)
        i1 = 0

        do k =  1, neightot
          i1 = i1 + 1
          jat =  list(k,iat)
          r_squared = rv_sq(i1)
          if (r_squared .le. ro2) then !   goto 32
            Inverse_r_squared = 1.0d0/r_squared
            jtype = atomtype(jat)
            i_pair = typee(itype,jtype)
            rrr = dsqrt(r_squared)     
            ndx   = Int(rrr*rdr)
            if (ndx < 1) ndx = 1
            ppp = rrr*rdr - dble(ndx)
            vk  = vvdw(ndx,  i_pair)
            vk1 = vvdw(ndx+1,i_pair)
            vk2 = vvdw(ndx+2,i_pair)
            t1 = vk  + (vk1 - vk )*ppp
            t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
            En = (t1 + (t2-t1)*ppp*0.5d0)
            vdw_en = vdw_en + En 

            gk  = gvdw(ndx,  i_pair)
            gk1 = gvdw(ndx+1,i_pair)
            gk2 = gvdw(ndx+2,i_pair)
            t1 = gk  + (gk1 - gk )*ppp
            t2 = gk1 + (gk2 - gk1)*(ppp - 1.0d0)
            force_term = (t1 + (t2-t1)*ppp*0.5d0)
            ff1(1:3) = (force_term*Inverse_r_squared)*dr(1:3)  
            af_i_1(1:3)=af_i_1(1:3)+ff1(:)
            atom_force (jj,1:3)=atom_force(jj,1:3)-ff1(1:3)

            l_distributed_charge_j = is_charge_distributed(jtype)

          if (z .le. 1.0d0) then
            write(6,*) 'close approach',z,itype,jtype,iat,jat
            z = 1.0d0
          end if

          ipiece = int((z - zs(0))*deltainv)+ 1
          zpiece = z - zs(ipiece-1)
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece

!C     *****determine energy
!C
          En =   & 
                       +     coefft(1,ipiece,itypee) &
                       +     coefft(2,ipiece,itypee)*zpiece &
                       +     coefft(3,ipiece,itypee)*zpiece2 &
                       +     coefft(4,ipiece,itypee)*zpiece3 &
                       +     coefft(5,ipiece,itypee)*zpiece4 &
                       +     coefft(6,ipiece,itypee)*zpiece5
          local_potential = local_potential + En
if (iat==2.and.jat==7863.or.iat==1.and.jat==3320) then
print*, iat,jat,local_potential*temp_cvt_en_ALL,En*temp_cvt_en_ALL
print*, 'itype jtype=',itype,jtype
print*, 'coef=',coefft(1:6,ipiece,itypee)
!print*, zpiece
!print*, 'xyz2=',x(:,2)
!print*, 'xyz7863=',x(:,7863)
!print*, 'En=',En*temp_cvt_en_ALL
!print*, '-------'
read(*,*)
endif
!C
!C     *****determine force 
!C
          ff   =   &
                       +     coeffft(2,ipiece,itypee)  &
                       +     coeffft(3,ipiece,itypee)*zpiece &
                       +     coeffft(4,ipiece,itypee)*zpiece2 &
                       +     coeffft(5,ipiece,itypee)*zpiece3 &
                       +     coeffft(6,ipiece,itypee)*zpiece4 
!C
!C     *****insert
!C
          f1 = ff*xij1
          f2 = ff*xij2
          f3 = ff*xij3
          f(1,iat) = f(1,iat) + f1
          f(2,iat) = f(2,iat) + f2
          f(3,iat) = f(3,iat) + f3
          f(1,jat) = f(1,jat) - f1
          f(2,jat) = f(2,jat) - f2
          f(3,jat) = f(3,jat) - f3
!C
          r = dsqrt(z)
          rinv = 1.0d0/r
          zinv = rinv*rinv
          r3inv = zinv*rinv
!C
!C     *****storing distances for polarization
!C
          nneigh = nneigh + 1
          listpol(nneigh,iat) = jat
          rx(nneigh,iat) = xij1
          ry(nneigh,iat) = xij2
          rz(nneigh,iat) = xij3
          rr(nneigh,iat) = r
          expc1r3(nneigh,iat)=dexp(-c1(itype,jtype)*r**3)  ! for damped pol.
          rrinv(nneigh,iat) = rinv
!C
          l1 = l_distributed_charge_i.and.l_distributed_charge_j
          l2 = l_distributed_charge_i.and.(.not.l_distributed_charge_j)
          l3 = .not.l_distributed_charge_i.and.l_distributed_charge_j
          l4 = (.not.l_distributed_charge_i)
          l4 = l4.and.(.not.l_distributed_charge_j)
          if (l1) then  ! gaussian gaussian
            betar = Ew_beta*r
            fact = (derfc(betar)*rinv + Ew_twobetapi*dexp(-betar*betar))*zinv
          elseif (l2) then !gaussian  puctual
            gammar = Ew_gamma*r
            fact = (derfc(gammar)*rinv + Ew_twogammapi*dexp(-gammar*gammar))*zinv
          elseif (l3) then ! puctual gaussian
            gammar = Ew_gamma*r
            fact = (derfc(gammar)*rinv + Ew_twogammapi*dexp(-gammar*gammar))*zinv
          elseif (l4) then ! punctual punctual
            alphar = alpha*r
            fact = (derfc(alphar)*rinv + twoalphapi*dexp(-alphar*alphar))*zinv
          facti = fact*q(itype)
          factj = fact*q(jtype)
 
          else  ! LOGICAL ERROR if hit here
           print*, 'IMPOSIBLE CASE IN interch select case l1 l2 ..l4'
           STOP
          endif
          elf(1,iat) = elf(1,iat) - factj*xij1
          elf(2,iat) = elf(2,iat) - factj*xij2
          elf(3,iat) = elf(3,iat) - factj*xij3
          elf(1,jat) = elf(1,jat) + facti*xij1
          elf(2,jat) = elf(2,jat) + facti*xij2
          elf(3,jat) = elf(3,jat) + facti*xij3

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

!C     ***** virial update every newpress 
!C      
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
              ffe   =       &
             +            coefffee(2,ipiece,itypee)  &
             +            coefffee(3,ipiece,itypee)*zpiece &
             +            coefffee(4,ipiece,itypee)*zpiece2 &
             +            coefffee(5,ipiece,itypee)*zpiece3 &
             +            coefffee(6,ipiece,itypee)*zpiece4 
              ff = ff - ffe 
              unbde =  unbde  &
                       + coeffee(1,ipiece,itypee)  &
                       + coeffee(2,ipiece,itypee)*zpiece  &
                       + coeffee(3,ipiece,itypee)*zpiece2 &
                       + coeffee(4,ipiece,itypee)*zpiece3 &
                       + coeffee(5,ipiece,itypee)*zpiece4 &
                       + coeffee(6,ipiece,itypee)*zpiece5
          vir = vir - ff*z
          end if
32      continue
         endif ! within cut off
        end do
        deallocate(xx,yy,zz,rv_sq)
        listmpol(iat) = nneigh
      end do

      unbd = unbd + local_potential
      print*, ' local potential Q + LJ = ',local_potential*temp_cvt_en
      print*, 'real spece Q+LJ unbd=',unbd*temp_cvt_en
      print*, 'unbde=',unbde*temp_cvt_en

      if (lredonefour) then
      do iat = 1, nat 
        ichain = chain(iat)
        itype  = atomtype(iat)
        do k =  1, listm14(iat)
          jat =  list14(k,iat)
          jtype = atomtype(jat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
          zinv=1.0d0/z
          itypee = typee(itype,jtype)

          ipiece = int((z - zs(0))*deltainv)+ 1
          zpiece = z - zs(ipiece-1)
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
!C
!C     *****determine energy
!C
          unbd = unbd -       redfactor*(    &
                             coefft(1,ipiece,itypee)  &
                       +     coefft(2,ipiece,itypee)*zpiece &
                       +     coefft(3,ipiece,itypee)*zpiece2 &
                       +     coefft(4,ipiece,itypee)*zpiece3 &
                       +     coefft(5,ipiece,itypee)*zpiece4 &
                       +     coefft(6,ipiece,itypee)*zpiece5)
!C
!C     *****determine force 
!C
          ff   =   &
                       +     coeffft(2,ipiece,itypee)  &
                       +     coeffft(3,ipiece,itypee)*zpiece &
                       +     coeffft(4,ipiece,itypee)*zpiece2 &
                       +     coeffft(5,ipiece,itypee)*zpiece3 &
                       +     coeffft(6,ipiece,itypee)*zpiece4
!C
!C     *****insert
!C
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
!C
          r = dsqrt(z)
          rinv=1.0d0/r
!C
          alphar = alpha*r
          fact = (derfc(alphar)*rinv + twoalphapi*dexp(-alphar*alphar))*zinv
          fact=(-redfactor)*fact
          facti = fact*q(itype)
          factj = fact*q(jtype)
          elf(1,iat) = elf(1,iat) - factj*xij1
          elf(2,iat) = elf(2,iat) - factj*xij2
          elf(3,iat) = elf(3,iat) - factj*xij3
          elf(1,jat) = elf(1,jat) + facti*xij1
          elf(2,jat) = elf(2,jat) + facti*xij2
          elf(3,jat) = elf(3,jat) + facti*xij3
!C
!C
!C     ***** virial update every newpress 
!C      
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
              ffe   =        &
             +            coefffee(2,ipiece,itypee)  &
             +            coefffee(3,ipiece,itypee)*zpiece &
             +            coefffee(4,ipiece,itypee)*zpiece2 &
             +            coefffee(5,ipiece,itypee)*zpiece3 &
             +            coefffee(6,ipiece,itypee)*zpiece4
              ffe=(-redfactor)*ffe
              ff = ff - ffe 
              unbde =  unbde -redfactor*( &
                         coeffee(1,ipiece,itypee)  &
                       + coeffee(2,ipiece,itypee)*zpiece  &
                       + coeffee(3,ipiece,itypee)*zpiece2 &
                       + coeffee(4,ipiece,itypee)*zpiece3 &
                       + coeffee(5,ipiece,itypee)*zpiece4 &
                       + coeffee(6,ipiece,itypee)*zpiece5)
          vir = vir - ff*z
          end if
        end do
       end do
      endif
!C
      vir = vir + unbde
!C
!C  ***  Anizotropic correction
!C  ***  Truncated & Surface force, energy and virial
!C
      if (lprofile) then
       virx=0.0d0
       virz=0.0d0
       unbd_corr=0.0d0
       subh=box(3)/dble(Nzslice)
       do iat=1,nat
         itype=non_of_charge(atomtype(iat))
         izslice=int(x(3,iat)/subh+1.0d-6)+1
         unbd_corr = unbd_corr - entr(itype,izslice)
         f(3,iat) = f(3,iat)+fotr(itype,izslice)
         virx=virx-vitr(1,itype,izslice)/4.d0
         virz=virz-vitr(2,itype,izslice)/2.d0
       end do   ! next iat
       unbd=unbd+unbd_corr
       box3inv =1.0d0/(box(1)*box(2)*box(3))
       tmp=6.8571d4*box3inv
       stresstr(1,1)=virx*tmp
       stresstr(2,2)=virx*tmp
       stresstr(3,3)=virz*tmp
!C
       if (printout) then
        write(6,'("anizotropic nonbonded correction",F12.4)'),unbd_corr
        write(6,'("anizotr. virial correction,atm",F12.4)'),tmp*(2*virx+virz)/3
       endif
      endif  ! if (lprofile)
!C    
      if (.not.lpolarizable)  then
       call deallocate_arrays
       return
      endif
      if (lredQ_mu14) then
       do iat = 1, nat-1 
        itype  = atomtype(iat)
        do k =  1, listm14(iat)
          jat =  list14(k,iat)
          jtype = atomtype(jat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
          zinv=1.0d0/z
!C
          r = dsqrt(z)
          rinv=1.0d0/r
          alphar = alpha*r
          fact = (derfc(alphar)*rinv + twoalphapi*dexp(-alphar*alphar))*zinv
          fact=(-redQmufactor)*fact
          facti = fact*q(itype)
          factj = fact*q(jtype)
          elf(1,iat) = elf(1,iat) - factj*xij1
          elf(2,iat) = elf(2,iat) - factj*xij2
          elf(3,iat) = elf(3,iat) - factj*xij3
          elf(1,jat) = elf(1,jat) + facti*xij1
          elf(2,jat) = elf(2,jat) + facti*xij2
          elf(3,jat) = elf(3,jat) + facti*xij3
!C
        end do
       end do
      endif   ! (lredQ_mu14)
!C
!C     *****induced dipole prediction
!C
      do iat = 1,nat
        px(iat) = 3.0d0*(px1old(iat)-px2old(iat)) + px3old(iat)
        py(iat) = 3.0d0*(py1old(iat)-py2old(iat)) + py3old(iat)
        pz(iat) = 3.0d0*(pz1old(iat)-pz2old(iat)) + pz3old(iat)
      end do

      iter = 0
7000  continue
      iter = iter + 1
      do iat = 1,nat
        tx(iat) = 0.0d0
        ty(iat) = 0.0d0
        tz(iat) = 0.0d0
      end do
!C
      iflag = 0
      do iat = 1,nat - 1
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
!C
          pjr = 3.0d0*zinv*fact*(px(jat)*rxij+py(jat)*ryij+pz(jat)*rzij)*lamda5
          pir = 3.0d0*zinv*fact*(px(iat)*rxij+py(iat)*ryij+pz(iat)*rzij)*lamda5
          tfact = arf-fact*lamda3
!C
          tx(iat) = tx(iat) + pjr*rxij + px(jat)*tfact
          ty(iat) = ty(iat) + pjr*ryij + py(jat)*tfact
          tz(iat) = tz(iat) + pjr*rzij + pz(jat)*tfact
!C
          tx(jat) = tx(jat) + pir*rxij + px(iat)*tfact
         ty(jat) = ty(jat) + pir*ryij + py(iat)*tfact
          tz(jat) = tz(jat) + pir*rzij + pz(iat)*tfact
!C
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
!C
      if (iter.gt.100) then
        write(6,*) 'iter>100'
        write(6,*) 'polarization did not converge'
        stop
      endif 
      if (iflag .ne. 0) goto 7000
!C
      do iat = 1,nat
        px3old(iat)=px2old(iat)     
        px2old(iat)=px1old(iat)
        px1old(iat)=px(iat)
        py3old(iat)=py2old(iat)     
        py2old(iat)=py1old(iat)
        py1old(iat)=py(iat)
        pz3old(iat)=pz2old(iat)     
        pz2old(iat)=pz1old(iat)
        pz1old(iat)=pz(iat)
      end do
!C
!C     *****induced energy calculation
!C
      uind=0.0d0
      do iat = 1,nat
        uind = uind - (elf(1,iat)*px(iat) + elf(2,iat)*py(iat) + elf(3,iat)*pz(iat))
      end do
!C
      unbd = unbd + uind*0.5d0*332.08d0
!C
!C     Calculate diple moment of the box that includes induced dipole moments
!C
      if (kount.gt.1.and.lboxdip.and.(mod(kount,kboxdip).eq.0))then
        do kk=1,3
          totdipole(kk)=0.0d0
        end do
        CALL unwrap(x,xun)
        do iat=1,nat   ! first to induced dipole contribution
          itype = atomtype(iat)
          ci = q(itype)
          totdipole(1)=totdipole(1)+xun(1,iat)*ci+px(iat)
          totdipole(2)=totdipole(2)+xun(2,iat)*ci+py(iat)
          totdipole(3)=totdipole(3)+xun(3,iat)*ci+pz(iat)
        end do
      endif   ! (lboxdip .and. mod(kount,kboxdip) .eq. 0)
!C
!C     *****calculate induced forces
!C
      do iat = 1,nat
        do kk = 1,3
          find(kk,iat) = 0.0d0
        end do
      end do
!C
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
!C
          c1r3=c1(itype,jtype)*r**3    ! damped pol.
          expc1r3val=expc1r3(k,iat)            ! damped pol.
          lamda3=1.d0-expc1r3val               ! damped pol.
          lamda5=1.d0-expc1r3val*(1.d0+c1r3)   ! damped pol.
          three_c1r2=3.d0*c1(itype,jtype)*z    ! damped pol.
          dlamda3=three_c1r2*expc1r3val                ! damped pol.
          dlamda5=three_c1r2*c1r3*expc1r3val           ! damped pol.
!C
          pjr = px(jat)*rxij+py(jat)*ryij+ pz(jat)*rzij
          pir = px(iat)*rxij+py(iat)*ryij+ pz(iat)*rzij
          pipj = px(jat)*px(iat) + py(jat)*py(iat) + pz(jat)*pz(iat)
          qq = ci*pjr - cj*pir
!C
!C     *****p*grad(T)*p contribution to the force and virial
!C
          tauval = tau(r)
          dtauval = dtau(r)
          p1 = 5.0d0*zinv*pjr*pir*lamda5 - pipj*lamda3
          p2 = dtauval*rinv*(3.0d0*r5inv*pir*pjr*lamda5-pipj*r3inv*lamda3)
          p3 = 3.0d0*r5inv*tauval
          p4 = p3*p1 - p2
!C
!C         p6 is a derivative of the damped terms
!C
          p6=(3.0d0*r5inv*pir*pjr*dlamda5-pipj*r3inv*dlamda3)*rinv
!C
!C
!C          p5 = 3.0d0*r5inv*pir*pjr-pipj*r3inv
!C
!C          virind = virind - p5*(3.0d0*tauval - dtauval*r)
!C
!C     *****p*grad(E) contribution to the force and virial
!C
          alphar = alpha*r
          ee1 = derfc(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
!C          ee4 = ee2*(1.0d0+alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
!C
!C          virind = virind - 2.0d0*qq*(ee1+ee4)
          f1 = p4 + ee5 - p6*tauval
          f2 = -(ee6*ci + p3*pir*lamda5)
          f3 = ee6*cj - p3*pjr*lamda5
!C
          ffx=f1*rxij+f2*px(jat)+f3*px(iat)
          ffy=f1*ryij+f2*py(jat)+f3*py(iat)
          ffz=f1*rzij+f2*pz(jat)+f3*pz(iat)
          find(1,iat)=find(1,iat)+ffx
          find(2,iat)=find(2,iat)+ffy
          find(3,iat)=find(3,iat)+ffz
          find(1,jat)=find(1,jat)-ffx
          find(2,jat)=find(2,jat)-ffy
          find(3,jat)=find(3,jat)-ffz
!C    added in v1.2, calculation of the virial due to real part of Q-mu
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
!C
        end do
      end do
!C
!C     *****reciprocal2 part
!C
      twopifact = 2.0d0*pi/(box(1)*box(2)*box(3))
      fouralpha2inv = 1.0d0/(4.0d0*alpha*alpha)
      do kk=1,3
        front(kk) = 2.0d0*pi/box(kk)
      end do
      elfact = 2.0d0*twopifact        ! not mupliplited by front
      virfact = 4.0d0*twopifact       ! not mupliplited by front
      klimit2=(klimitx*klimitx+klimity*klimity+klimitz*klimitz)/3.0d0
!C
      do kx = 0,klimitx
        kxf=kx*front(1)
        if (kx .eq. 0) then 
          factor = 1.0d0
         else
          factor = 2.0d0
        endif
        do ky = -klimity,klimity
          kyf=ky*front(2)
          do kz = -klimitz,klimitz
            kzf=kz*front(3)
            kmag2 = kx*kx+ky*ky+kz*kz
            if (kmag2 .ge. klimit2 .or. kmag2 .eq. 0) goto 342
            rkmag2 = kxf*kxf+kyf*kyf+kzf*kzf
            akvec = factor*dexp(-rkmag2*fouralpha2inv)/rkmag2
            cossump = 0.0d0
            sinsump = 0.0d0
            cossumx = 0.0d0
            sinsumx = 0.0d0
            do iat = 1,nat
              itype = atomtype(iat)
              ci = q(itype)
              pk = (px(iat)*kxf+py(iat)*kyf+pz(iat)*kzf)
              dotik =x(1,iat)*kxf+x(2,iat)*kyf+x(3,iat)*kzf
              cost = dcos(dotik)
              sint = dsin(dotik)
              costermx(iat) = ci*cost
              sintermx(iat) = ci*sint
              costermp(iat) = pk*cost
              sintermp(iat) = pk*sint
              cossump = cossump + pk*cost
              sinsump = sinsump + pk*sint
              cossumx = cossumx + ci*cost
              sinsumx = sinsumx + ci*sint
            end do
            prefact1=akvec*elfact
            prefact2=akvec*virfact*(rkmag2*fouralpha2inv-1.0d0)
            do iat = 1,nat
              f1 = (costermp(iat)*cossumx - costermx(iat)* &
                   cossump + sintermp(iat)*sinsumx -      &
                    sintermx(iat)*sinsump)
              find(1,iat) = find(1,iat) + f1*kxf*prefact1
              find(2,iat) = find(2,iat) + f1*kyf*prefact1
              find(3,iat) = find(3,iat) + f1*kzf*prefact1
            end do
            virind = virind + prefact2*(cossumx*sinsump-sinsumx*cossump)
!C
 342       continue
          end do
        end do
      end do
!C
!C     *****correct2 part
!C
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
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
         else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          r = dsqrt(z)
          alphar = alpha*r
          zinv = 1.0d0/z
          rinv = 1.0d0/r
          r3inv = zinv*rinv
          r5inv = r3inv*zinv
!C
          pir=px(iat)*xij1+py(iat)*xij2+pz(iat)*xij3
          pjr=px(jat)*xij1+py(jat)*xij2+pz(jat)*xij3
          qq = ci*pjr - cj*pir
          alphar = alpha*r
          ee1 = -derf(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
!C          ee4 = ee2*(1.0d0+alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
!C
          f1 = ee5*xij1 - ee6*(ci*px(jat)-cj*px(iat))
          f2 = ee5*xij2 - ee6*(ci*py(jat)-cj*py(iat))
          f3 = ee5*xij3 - ee6*(ci*pz(jat)-cj*pz(iat))
!C
!C           virind = virind - 2.0d0*qq*(ee4 + ee1)  ! using virial
           virind = virind - ee5*z + ee6*qq ! using force
!C
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
!C
        end do
      end do

      if (lredonefour) then
      do iat = 1,nat 
        itype = atomtype(iat)
        ci = q(itype)
        do k = 1,listm14(iat)
          jat = list14(k,iat)
          jtype = atomtype(jat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          jtype = atomtype(jat)
          itypee = typee(itype,jtype)

          cj = q(jtype)
          r = dsqrt(z)
          rinv = 1.0d0/r
          zinv=rinv*rinv
          z = r*r
          zinv = rinv*rinv
          r3inv = zinv*rinv
          r5inv = r3inv*zinv
          fact = r3inv 
          rxij = xij1
          ryij = xij2
          rzij = xij3
          pjr = px(jat)*rxij+py(jat)*ryij+ pz(jat)*rzij
          pir = px(iat)*rxij+py(iat)*ryij+ pz(iat)*rzij
          pipj = px(jat)*px(iat) + py(jat)*py(iat) + pz(jat)*pz(iat)
          qq = ci*pjr - cj*pir
!C
!C     *****p*grad(E) contribution to the force and virial
!C
          alphar = alpha*r
          ee1 = derfc(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
!C
          f1 = ee5
          f2 =-ee6*ci 
          f3 = ee6*cj
!C
          ffx=(f1*rxij+f2*px(jat)+f3*px(iat))*(-redfactor)
          ffy=(f1*ryij+f2*py(jat)+f3*py(iat))*(-redfactor)
          ffz=(f1*rzij+f2*pz(jat)+f3*pz(iat))*(-redfactor)
          find(1,iat)=find(1,iat)+ffx
          find(2,iat)=find(2,iat)+ffy
          find(3,iat)=find(3,iat)+ffz
          find(1,jat)=find(1,jat)-ffx
          find(2,jat)=find(2,jat)-ffy
          find(3,jat)=find(3,jat)-ffz
!C    added in v1.2, calculation of the virial due to real part of Q-mu
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
!C
!C
!C     *****correct2 part
!C
          alphar = alpha*r
          ee1 = -derf(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
!C
          f1 =(ee5*xij1 - ee6*(ci*px(jat)-cj*px(iat)))*(redfactor)
          f2 =(ee5*xij2 - ee6*(ci*py(jat)-cj*py(iat)))*(redfactor)
          f3 =(ee5*xij3 - ee6*(ci*pz(jat)-cj*pz(iat)))*(redfactor)

          virind = virind-redfactor*(ee5*z - ee6*qq) ! using force
!C
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

!C
         end do
        end do
       endif
!C
      if (lredQ_mu14) then
      do iat = 1,nat - 1 
        itype = atomtype(iat)
        ci = q(itype)
        do k = 1,listm14(iat)
          jat = list14(k,iat)
          jtype = atomtype(jat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3
          jtype = atomtype(jat)
          cj = q(jtype)
!C
          r = dsqrt(z)
          rinv = 1.0d0/r
          zinv=rinv*rinv
          z = r*r
          zinv = rinv*rinv
          r3inv = zinv*rinv
          r5inv = r3inv*zinv
          rxij = xij1
          ryij = xij2
          rzij = xij3
          pjr = px(jat)*rxij+py(jat)*ryij+ pz(jat)*rzij
          pir = px(iat)*rxij+py(iat)*ryij+ pz(iat)*rzij
          pipj = px(jat)*px(iat) + py(jat)*py(iat) + pz(jat)*pz(iat)
          qq = ci*pjr - cj*pir
!C
!C     *****p*grad(E) contribution to the force and virial
!C
          alphar = alpha*r
          ee1 = derfc(alphar)*r3inv
          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
!C
          f1 = ee5
          f2 =-ee6*ci 
          f3 = ee6*cj 
!C
          ffx=(f1*rxij+f2*px(jat)+f3*px(iat))*(-redQmufactor)
          ffy=(f1*ryij+f2*py(jat)+f3*py(iat))*(-redQmufactor)
          ffz=(f1*rzij+f2*pz(jat)+f3*pz(iat))*(-redQmufactor)
          find(1,iat)=find(1,iat)+ffx
          find(2,iat)=find(2,iat)+ffy
          find(3,iat)=find(3,iat)+ffz
          find(1,jat)=find(1,jat)-ffx
          find(2,jat)=find(2,jat)-ffy
          find(3,jat)=find(3,jat)-ffz
!C    added in v1.2, calculation of the virial due to real part of Q-mu
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
!C
!C
!C     *****correct2 part
!C
          alphar = alpha*r
          ee1 = -derf(alphar)*r3inv
!C          ee2 = zinv*twoalphapi*dexp(-alphar*alphar)
!C          ee3 = ee2*(3.0d0+2.0d0*alphar*alphar)
!C    ee2 and ee3 are the same as above
          ee5 = qq*zinv*(3.0d0*ee1 + ee3)
          ee6 = ee1 + ee2
!C
          f1 =(ee5*xij1-ee6*(ci*px(jat)-cj*px(iat)))*(redQmufactor)
          f2 =(ee5*xij2-ee6*(ci*py(jat)-cj*py(iat)))*(redQmufactor)
          f3 =(ee5*xij3-ee6*(ci*pz(jat)-cj*pz(iat)))*(redQmufactor)
!C
          virind = virind-redQmufactor*(ee5*z - ee6*qq) ! using force
!C
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
!C
         end do
        end do
       end if  ! (lredQ_mu14)
!C
      do iat = 1,nat
        do kk = 1,3
          f(kk,iat) = f(kk,iat) + find(kk,iat)!*332.08d0
        end do
      end do
!C   
      if (newpress) then
        vir = vir + virind!*332.08d0
        do kk=1,3
          do jj=1,3
            tvirpo(kk,jj)=tvirpo(kk,jj)+tvirind(kk,jj) !*332.08d0
          end do
        end do
      endif
!C

      call deallocate_arrays

      contains 
      subroutine deallocate_arrays
      deallocate(costermp);deallocate(sintermp)
      deallocate(costermx);deallocate(sintermx)
      deallocate(tx); deallocate(ty);deallocate(tz)
      deallocate(xun)
      deallocate(pxnew); deallocate(pynew);deallocate(pznew)
      deallocate(expc1r3)
      deallocate(pz1old); deallocate(pz2old);deallocate(pz3old)
      deallocate(py1old); deallocate(py2old);deallocate(py3old)
      deallocate(px1old); deallocate(px2old);deallocate(px3old)
      deallocate(find)
      deallocate(rx); deallocate(ry); deallocate(rz)
      deallocate(rr); deallocate(rrinv)
      deallocate(listmpol)
      deallocate(listpol)
      end subroutine deallocate_arrays
      end subroutine interch
!C
      subroutine interchs()
!C
      implicit none
      integer iat,ichain,itype,neightot,k,jat,jtype,itypee
      integer ipiece,kk
      real(8) deltainv,z,zpiece,zpiece2,zpiece3,zpiece4,zpiece5,ff
      real(8) f1,f2,f3,ffe,xij1,xij2,xij3
!C
!C     *****begin calculations
!C
      print*, 'GOT in interCHNSSSS'
      deltainv = 1.0d0/deltaspline
      el(:) = box(:)*0.5d0
!C
!C     *****begin calculation of non-bonded and electrostatic energies
!C          and forces
!C
      do iat = 1, nat - 1
        ichain = chain(iat)
        itype  = atomtype(iat)
        neightot = listms(iat)
        do k =  1, neightot
          jat =  lists(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
!C
        if (z <= rshort2) then !if (z .gt. rshort2) goto 32

          jtype = atomtype(jat)
          itypee = typee(itype,jtype)

          if (z .le. 1.0d0) then
            write(6,*) 'close approach',z,itype,jtype,iat,jat
            z = 1.0d0
          end if

          ipiece = int((z - zs(0))*deltainv)+ 1
!C     *****determine position within piece of spline
          zpiece = z - zs(ipiece-1)
!C     *****determine powers for spline calculations
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
!C
!C     *****determine force 
!C
          ff   =   &
                       +     coeffft1(2,ipiece,itypee)  &
                       +     coeffft1(3,ipiece,itypee)*zpiece &
                       +     coeffft1(4,ipiece,itypee)*zpiece2 &
                       +     coeffft1(5,ipiece,itypee)*zpiece3 &
                       +     coeffft1(6,ipiece,itypee)*zpiece4
!C
!C     *****insert
!C
          f1 = ff*xij1
          f2 = ff*xij2
          f3 = ff*xij3
          f(1,iat) = f(1,iat) + f1
          f(2,iat) = f(2,iat) + f2
          f(3,iat) = f(3,iat) + f3
          f(1,jat) = f(1,jat) - f1
          f(2,jat) = f(2,jat) - f2
          f(3,jat) = f(3,jat) - f3
       endif ! within cut - off
!32      continue
        end do
      end do

      if (lredonefour) then
      do iat = 1, nat 
        ichain = chain(iat)
        itype  = atomtype(iat)
        do k =  1, listm14(iat)
          jat =  list14(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
          jtype = atomtype(jat)
          itypee = typee(itype,jtype)
          ipiece = int((z - zs(0))*deltainv)+ 1
          zpiece = z - zs(ipiece-1)
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
!C
!C     *****determine force 
!C
          ff   =   &
                       +     coeffft1(2,ipiece,itypee)  &
                       +     coeffft1(3,ipiece,itypee)*zpiece &
                       +     coeffft1(4,ipiece,itypee)*zpiece2 &
                       +     coeffft1(5,ipiece,itypee)*zpiece3 &
                       +     coeffft1(6,ipiece,itypee)*zpiece4 
!C
!C     *****insert
!C
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

        end do
      end do
      endif

      end subroutine interchs
      subroutine intsetup()

      implicit none
      integer i,kk,iat
      real(8) wys(5)

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

      nhcstep1 = 5
      nhcstep2 = 5
!C
!C     *****thermostat mass
!C
      qtmass(1) = dble(ndof)*gaskinkc*tstart/(qfreq**2)
!C
!C     *****barostat mass
!C
      wtmass(1) = dble(ndof + 3)*gaskinkc*tstart/(wfreq**2)
!C
!C     *****parameters in nose-hoover + andersen-hoover methods
!C
      do i = 2,10
        qtmass(i) = gaskinkc*tstart/(qfreq**2)
        wtmass(i)=dble(ndof + 3)*gaskinkc*tstart/(3.d0*wfreq**2)
      end do
!C
!C     *****parameters in the Yoshida/Suzuki integration
!C
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
!C
      do i = 1,nhcstep2
        wdt1(i) = wys(i)*stepout/dble(nhcstep1)
        wdt2(i) = wdt1(i)/2.0d0
        wdt4(i) = wdt1(i)/4.0d0
        wdt8(i) = wdt1(i)/8.0d0
      end do
!C
!C     *****parameters for yoshida suzuki integration
!C
      e2 = 1.d0/6.d0
      e4 = e2/20.d0
      e6 = e4/42.d0
      e8 = e6/72.d0
      gnkt = tstart * dble(ndof) * gaskinkc
      gn3kt = (ndof + 3)*tstart*gaskinkc
      gkt  = tstart * gaskinkc
      ondf = 1.d0 + (1.d0/dble(nat))
      onf = 1.0d0/dble(ndof)
!C
      do iat = 1,nat
        do kk = 1,3
          fnow(kk,iat) = f(kk,iat)
        end do
      end do
!C
      return
      end subroutine intsetup
      subroutine nbenergy()
!C
      implicit none
      integer iat,ichain,itype,neightot,k,jat,jtype,jchain,ipiece
      integer itypee,ifunct,innb,nfunct,ncall
      real(8) uninter,unintra,deltainv,entot
      real(8) xij1,xij2,xij3,z,zpiece,zpiece2,zpiece3,zpiece4,zpiece5
      real(8) en(3),enfintra(3,maxnnb),enfinter(3,maxnnb),enftot(3)
!C
      data nfunct /3/
      data ncall /0/
      ncall = ncall + 1
      el(:) = box(:) * 0.5d0 
      if (ncall .eq. 1)then
      deltainv = 1.0d0/deltaspline
      entot = 0.0d0
      unintra = 0.0d0
      uninter = 0.0d0
      en(:) = 0.0d0
      enfintra(:,:) = 0.0d0
      enfinter(:,:) = 0.0d0
      end if
!C
!C     *****begin calculation of non-bonded and electrostatic energies
!C          and forces
!C
      do iat = 1, nat - 1
        ichain = chain(iat)
        itype  = atomtype(iat)
        neightot = listm(iat)
        do k =  1, neightot
          jat =  list(k,iat)
          xij1 = x(1,jat) - x(1,iat)
          xij2 = x(2,jat) - x(2,iat)
          xij3 = x(3,jat) - x(3,iat)
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          else
          if(abs(xij1).gt.el(1)) xij1=xij1-dsign(box(1),xij1)
          if(abs(xij2).gt.el(2)) xij2=xij2-dsign(box(2),xij2)
          if(abs(xij3).gt.el(3)) xij3=xij3-dsign(box(3),xij3)
          endif
          z = xij1*xij1 + xij2*xij2 + xij3*xij3   
          if (z .gt. ro2) goto 332
          jtype = atomtype(jat)
          jchain = chain(jat)
          itypee = typee(itype,jtype)
          if (z .le. 1.0d0) z = 1.0d0
          ipiece = int((z - zs(0))*deltainv)+ 1
          zpiece = z - zs(ipiece-1)
          zpiece2 = zpiece*zpiece
          zpiece3 = zpiece2*zpiece
          zpiece4 = zpiece3*zpiece
          zpiece5 = zpiece4*zpiece
          do ifunct = 1,nfunct
            en(ifunct) =  &
                coeff1(ifunct,ipiece,itypee) &
              + coeff2(ifunct,ipiece,itypee)*zpiece &
              + coeff3(ifunct,ipiece,itypee)*zpiece2 &
              + coeff4(ifunct,ipiece,itypee)*zpiece3 &
              + coeff5(ifunct,ipiece,itypee)*zpiece4 &
              + coeff6(ifunct,ipiece,itypee)*zpiece5 
          end do

        en(1) = electrostatic(itypee,1)/(dsqrt(z))


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

      write(59,*)
      write(59,*)'Average nonbonded      energy = ',entot/ncall
      write(59,*)'Average intramolecular energy = ',unintra/ncall
      write(59,*)'Average intermolecular energy = ',uninter/ncall
      write(59,*)
      write(59,*)'*****************INTRA MOLECULAR SPLIT************'
      write(59,*)' Interaction      Electrostatic     Polarization   ','   LJ/Exp-6    '

      enftot(:) = 0.0d0

      do itype = 1,maxcharges
        do jtype = itype,maxcharges
          innb = typee(itype,jtype)
        write(59,259)atom_labels(itype),atom_labels(jtype),&
              (enfintra(ifunct,innb)/ncall,ifunct=1,3)
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
      write(59,*)' Interaction      Electrostatic     Polarization   ','   LJ/Exp-6    '

        enftot(:) = 0.0d0

      do itype = 1,maxcharges
        do jtype = itype,maxcharges
          innb = typee(itype,jtype)
        write(59,259)atom_labels(itype),atom_labels(jtype),&
              (enfinter(ifunct,innb)/ncall,ifunct=1,3)
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
      end subroutine nbenergy

      subroutine nhcint()
     
      implicit none
      integer i,j,inos,iat,kk
      real(8) scale,aa

      CALL getkin()

      ekin2 = 2.d0 * kinen
      scale = 1.d0
      glogs(1) = (ekin2 - gnkt)/qtmass(1)
      do i = 1,nhcstep1
        do j = 1,nhcstep2
          vlogs(nnos) = vlogs(nnos)+glogs(nnos)*wdt4(j)
          do inos = 1, nnos-1
            aa = dexp(-wdt8(j)*vlogs(nnos+1-inos)) 
            vlogs(nnos-inos) = vlogs(nnos-inos)*aa*aa &
                             + wdt4(j)*glogs(nnos-inos)*aa
          end do

          aa = dexp(-wdt2(j)*vlogs(1))
          scale = scale*aa
          glogs(1) = (scale*scale*ekin2 - gnkt)/qtmass(1)
 
          do inos = 1,nnos
            xlogs(inos) = xlogs(inos) + vlogs(inos)* wdt2(j)
          end do

          do inos = 1, nnos-1
            aa = dexp(-wdt8(j)*vlogs(inos+1))
            vlogs(inos) = vlogs(inos)*aa*aa+ wdt4(j)*glogs(inos)*aa
            glogs(inos+1) = (qtmass(inos)*vlogs(inos)*vlogs(inos)- gkt) / qtmass(inos+1)
          end do
          vlogs(nnos) = vlogs(nnos) + glogs(nnos)*wdt4(j)
        end do
      end do

      do iat = 1, nat
        do kk = 1,3
          v(kk,iat) = v(kk,iat) * scale
        end do
      end do

      end subroutine nhcint

      Subroutine nptint()
     
      implicit none
      integer inos,iat,kk,i,j
      real(8) aa,cons,scale,box3,ak(3),boxmv2,trvlogv

      cons = 1.0d0      !1.4582454d-5

      CALL getkin()

      box3 = box(1)*box(2)*box(3)
      boxmv2 = wtmass(1)*(vlogv(1)*vlogv(1)+vlogv(2)*vlogv(2)+ vlogv(3)*vlogv(3))
      glogs(1)=(ekin2+boxmv2-gn3kt)/qtmass(1)

      do kk = 1,3
        glogv(kk) = (onf*ekin2+akin2(kk)+(prt(kk)-pfix(kk))*box3*cons)/wtmass(1)
      end do
!C
!C     *****start yoshida/suzuki multiple step method
!C
      do i = 1,nhcstep1
        do j = 1,nhcstep2
          vlogs(nnos) = vlogs(nnos)+glogs(nnos)*wdt4(j)
          do inos = 1, nnos-1
            aa = dexp(-wdt8(j)*vlogs(nnos+1-inos))
            vlogs(nnos-inos) = vlogs(nnos-inos)*aa*aa+ wdt4(j)*glogs(nnos-inos)*aa
          end do

          aa = dexp(-wdt8(j)*vlogs(1))
          do kk = 1,3
            vlogv(kk)  = vlogv(kk)*aa*aa + wdt4(j)*glogv(kk)*aa
          end do
          trvlogv = onf*(vlogv(1)+vlogv(2)+vlogv(3))
          do kk = 1,3

!         update kinetic energy contribution

            aa = dexp(-wdt2(j)*(vlogs(1)+trvlogv+vlogv(kk)))
            akin2(kk) = akin2(kk)*aa*aa
            do iat = 1, nat
              v(kk,iat) = v(kk,iat) * aa
            end do
          end do
          ekin2 = 0.0d0
          do kk = 1,3
            ekin2 = ekin2 + akin2(kk)
          end do

          do kk = 1,3
            glogv(kk) = (onf*ekin2+akin2(kk)+(prt(kk) &
                        -pfix(kk))*box3*cons)/wtmass(1)
          end do
          do inos = 1,nnos
            xlogs(inos) = xlogs(inos) + vlogs(inos)* wdt2(j)
          end do

          aa = dexp(-wdt8(j)*vlogs(1))
          do kk = 1,3
            vlogv(kk)  = vlogv(kk)*aa*aa + wdt4(j)*glogv(kk)*aa
          end do
          boxmv2 = wtmass(1)*(vlogv(1)*vlogv(1)+vlogv(2)*vlogv(2)+  vlogv(3)*vlogv(3))
          glogs(1)=(ekin2+boxmv2-gn3kt)/qtmass(1)
          do  inos = 1, nnos-1
            aa =dexp(-wdt8(j)*vlogs(inos+1))
            vlogs(inos) = vlogs(inos)*aa*aa+ wdt4(j)*glogs(inos)*aa
            glogs(inos+1) = (qtmass(inos)*vlogs(inos)*vlogs(inos)- gkt) / qtmass(inos+1)
          end do
          vlogs(nnos) = vlogs(nnos) + glogs(nnos)*wdt4(j)
        end do
      end do
      return
      end subroutine nptint

      subroutine output1()
   
      implicit none
      integer inos,nfile1,nfile2,kk,iat,itype,jj,istart,n
      real(8) timeo,timev,timex,times,timef,qi
      real(8) short(3,maxat),shstress(3,3),shvel(3,maxat)
      real(8) chargeflux(3)
      real(8) timed
      character*3 label

      data nfile1 /0/
      data nfile2 /0/
      timeo = dble(koutput)*stepout
      timex = dble(kcoords)*stepout
      timev = dble(kvelocs)*stepout
      times = dble(kstress)*stepout
      timed = real(kboxdip)*stepout
      timef = dble(nsteps)*stepout
!C
!C     *****total dipole moment and charge flux (file = fort.76)
!C
      if (lboxdip .and. mod(kount,kboxdip) .eq. 0)then
        do kk = 1,3
          chargeflux(kk) = 0.0d0
        end do
        do iat=1,nat
          qi=q(atomtype(iat))
          do kk=1,3
            chargeflux(kk)=chargeflux(kk)+qi*v(kk,iat)
          end do
        end do

        open (76,file='fort.76',form='unformatted',status='old',access='append')
        write (76)timed
        write (76)totdipole,chargeflux
        close (76)
      end if
!C
!C     *****stress tensor(file = fort.78)
!C
      if (lstress .and. mod(kount,kstress) .eq. 0)then
        do kk = 1,3
          do jj = 1,3
            shstress(kk,jj) = stress(kk,jj)
          end do
        end do

        open (78,file='fort.78',form='unformatted',status='old',access='append')
!C       open (78,file='fort.78',
!C    +     form='unformatted',status='old',iostat=ios) ! RS6000(ibm)
        write (78)times
        write (78)shstress
        close (78)
      end if
!C
!C     *****coordinates(file = fort.77)
!C
      if (lcoords .and. mod(kount,kcoords) .eq. 0) then

        nfile1 = nfile1 + 1
        do iat = 1,maxat
          do kk =1,3
            short(kk,iat) = x(kk,iat)
          end do
        end do
        open (77,file='fort.77',form='unformatted',status='old',access='append')
!C        open (77,file='fort.77',
!C    +   form='unformatted',status='old',iostat=ios) ! RS6000(ibm)
        write (77) nat,timex,nfile1,box
        write (77) short
        close (77)
       end if
!C
!C     *****velocities(file = fort.79)
!C
      if (lvelocs .and. mod(kount,kvelocs) .eq. 0) then

        do iat = 1,maxat
          do kk =1,3
            shvel(kk,iat) = v(kk,iat)
          end do
        end do
        open (79,file='fort.79', form='unformatted',status='old',access='append')
!C       open (79,file='fort.79',
!C    +   form='unformatted',status='old',iostat=ios) ! RS6000(ibm)
        write (79) nat,timev
        write (79) shvel
        close (79)
      end if

!C     ***** output coordinates and velocities periodically(file = fort.66)

      if (mod(kount,koutput) .eq. 0)then
        open (167,file="jmol.out",status='unknown',err=661)
        write (167,*) nat
        write (167,*)
        do iat = 1,nat
          itype = atomtype(iat)
          if (itype .eq. 8) then
             label = 'Lp '
            else if (itype .eq. 14) then 
             label = 'Li '
            else if (itype .eq. 18 .or. itype .eq. 19) then 
             label = 'Fe '
            else
             label = atom_labels(itype)(1:1)
          end if
          write(167,*) label,(x(kk,iat),kk = 1,3) 
        end do
        close (167)
        nfile2 = nfile2 + 1
        open (66,file="coords.out",status='unknown',err=661)
        write(66,999)cline
        write(66,199)
        write(66,999)cline
        write(66,998)
        write(66,200) timeo*nfile2/1000.0d0,nint(timeo*nfile2*100.0d0/timef)
        write(66,998)
        write(66,198)
        write(66,197)
        write(66,999)cline
        write(66,998)
        do iat = 1,nat
          itype = atomtype(iat)
          write(66,220)(x(kk,iat),kk = 1,3), &
           atom_labels(itype)(1:3),atom_labels(itype)(4:5)
          write(66,221) (v(kk,iat)*unit_velocity,kk = 1,3)
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
 199  format('*           ATOM COORDINATES (Angstroms) AND ','VELOCITIES (m/sec)')
 198  format('*        X              Y              Z  Atomtype', ' Chargetype')
 197  format('*        Vx             Vy             Vz')
 200  format('*     Time = ',f15.3,' picoseconds (',i3,'% of ','the simulation run)')
 201  format('*   Box (Angstroms) ')
 202  format('*   Xlogs (Thermostat position)')
 203  format('*   Vlogs (Thermostat velocity)')
 204  format('*   Glogs (Thermostat force)')
 205  format('*   Xlogv (Barostat position)')
 206  format('*   Vlogv (Barostat velocity)')
 207  format('*   Glogv (Barostat force)')
 220  format (3f15.6,3x,a3,3x,a2)
 221  format (1x,4f15.6)

      end subroutine output1
      subroutine output2()

      implicit none
      integer iprop,nens
      character*20 ens(3),ensemble

      data ens /'Microcanonical','Canonical','Isobaric-isothermal'/

      open (70,file='fort.70',status='unknown')
!C
!C     *****compute overall sd in properties
!C
      do iprop = 1,maxprop
        if ((simsqpr(iprop) - simpr(iprop)**2) .lt. 1.0d-12)then
          stdev(iprop) = 0.0d0
         else
          stdev(iprop) = dsqrt(simsqpr(iprop) - simpr(iprop)**2)
        end if
      end do

      if (nve)ensemble = ens(1)
      if (nvt)ensemble = ens(2)
      if (npt)ensemble = ens(3)
      if (.not. lboxdip)nboxdip = 0
      if (.not. lcoords)ncoords = 0
      if (.not. lvelocs)nvelocs = 0
      if (.not. lstress)nstress = 0
!C
!C     *****write statistics to file 70
!C
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
      write(70,*)'  Alpha = ',alpha
      write(70,*)'  Klimit = ',klimitx,klimity,klimitz
      write(70,*)'Integrator : ',&
      'Multiple timestep integrator(vibrational+double cutoff)'
      write(70,*)'  Multimed = ',multimed
      write(70,*)'  Multibig = ',multibig
      write(70,*)'  Rshort = ',ros
      write(70,*)'Polarization : ',&
      'Regular Ewald'
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
      write(70,*)'Average box               = ',simpr(3),simpr(25), simpr(26)
      write(70,*)'Std dev in box            = ',stdev(3)
      write(70,*)'Std dev in Hamiltonian    = ',stdev(5)
      write(70,*)'        *****   Average stress tensor   ***** '
      write(70,255)simpr(16),simpr(17),simpr(18)
      write(70,255)simpr(19),simpr(20),simpr(21)
      write(70,255)simpr(22),simpr(23),simpr(24)
      write(70,*)
      write(70,*)clineo


 255  format (5x,3f15.4)
      return
      end subroutine output2

      subroutine parse_line(line,lstart,istart,iend)

      implicit none
!C
!C     *****shared variables
!C
      integer lstart,istart,iend
      character*144 line
!C
!C     *****local variables
!C
      character*1 value
      integer ipos
!C
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

!      return
      end subroutine parse_line

      double precision FUNCTION pdforce(r)
 
      implicit none
      real(8) r
      real(8) derfc,delta,eofr
!C
!C     *****begin force evaluation
!C
      pdforce = 0.0d0
      if (.not. tbpol)return
      if (r .gt. rscalep) then ! in scaled region
        delta = rop-rscalep
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pdforce = (2.d0/(eofr*r**5))*(-6.d0*(r-rscalep)   /delta**2 &
                           +6.d0*(r-rscalep)**2/delta**3)           &
               +(-10.d0/(eofr*r**6)                                 &
              - 2.0d0*log(epsilonp)*repsilonp/(eofr*r**7))          &
                 *(1.d0 - 3.d0*(r-rscalep)**2/delta**2              &
                       + 2.d0*(r-rscalep)**3/delta**3)
      else if (r .gt. repsilonp) then
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pdforce = (-10.d0/(eofr*r**6) - 2.0d0*log(epsilonp)*repsilonp/(eofr*r**7)) 
       else
         pdforce = (-10.d0/(r**6))
      end if
      return
      end FUNCTION pdforce
      double precision FUNCTION pforce(r)

      implicit none
      real(8) r
      real(8) delta,eofr
!C
!C     *****begin force evaluation
!C
      pforce = 0.0d0
      if (.not. tbpol)return
      if (r .gt. rscalep) then ! in scaled region
        delta = rop-rscalep
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pforce = (2.d0/(eofr*r**5))*                        &
                   (1.d0-3.d0*(r-rscalep)**2/delta**2      &
                           +2.d0*(r-rscalep)**3/delta**3)
       else if (r .gt. repsilonp) then
        eofr = epsilonp*exp(-log(epsilonp)*repsilonp/r)
        pforce = (2.0d0/(eofr*r**5))
       else
        pforce = (2.d0/(r**5))
      end if
      return
      end FUNCTION pforce

      double precision FUNCTION pintforce(xa,flower,xb,fupper)

      implicit none
      real(8) xa,flower,xb,fupper
      integer ifunct
      real(8) t(100,100)
      real(8) suma,odelx,relerror
      integer j,i,icol

      relerror = 1.0d-4
!C
!C     *****begin calculations
!C
      t(1,1)=(xb-xa)/2.0d0*(flower+fupper)
      t(2,1)=t(1,1)/2.0d0+(xb-xa)* pforce((xa+xb)/2.0d0)/2.0d0
      t(2,2)=(4.0d0*t(2,1)-t(1,1))/3.0d0

      do j=3,100
        odelx=(xb-xa)/2.0d0**(j-2)
        suma=0.0d0
        do i=1,2**(j-2)
          suma=suma+pforce(xa+(dble(i)-0.5d0)*odelx)
        end do
        t(j,1)=0.5d0*(t(j-1,1)+odelx*suma)
        do icol=2,j
          t(j,icol)=(4.0d0**(icol-1)*t(j,icol-1)-t(j-1,icol-1))/(4.0d0**(icol-1)-1.0)
        end do
        if (abs((t(j,j)-t(j-1,j-1))/t(j,j)) .le. relerror) then
          pintforce = -t(j,j)
          goto 110
        end if
      end do
      write (6,*) 'integral did not converge'
110   continue

      end FUNCTION pintforce
      subroutine read11()

      implicit none
      integer ch,iat,ibond,itbond,j,ibend,ib1,ib2,nchains
      integer imult,itemp,ich,icheck1,icheck2,itort,ideform,ib
      integer iba,itot,itest,itbend,ittort,itdefs,ibase
      integer, allocatable :: nbondsch(:),nbendsch(:),ntortsch(:),&
                             ndeformsch(:),natch(:),nbondex(:),&
                             nbendex(:)
       
!      integer nbondsch(maxnch),nbendsch(maxnch),ntortsch(maxnch),
!     +                ndeformsch(maxnch),natch(maxnch),nbondex(maxnch)
!      integer nbendex(maxnch)
      integer ibaseb,ib3,icheck3  ! for PDMS
      character*144 line
!C
!C
!C     *****read in data
!C
      open(11,file="connectivity.dat",status="old")
      itbond = 0
      itbend = 0
      ittort = 0
      itdefs = 0
      CALL read_a_line(11,line)
      read (line,*) nchains, nions
      nch = nchains + nions   ;   maxnch = nch ; iChainSurf=maxnch+1
      nat = 0
      nbonds = 0
      nbends = 0
      ntorts = 0
      ndeforms = 0
      allocate(natch(nch),nbondsch(nch),nbendsch(nch),ntortsch(nch))
      allocate(ndeformsch(nch))
      allocate(nbondex(nch),nbendex(nch))
      do ich=1,nch        ! The force field file = 1, nch
        CALL read_a_line(11,line)
        read (line,*) natch(ich),nbondsch(ich),&
          nbendsch(ich),ntortsch(ich),ndeformsch(ich)
        nat = nat + natch(ich)
        nbondex(ich) = nbonds
        nbendex(ich) = nbends
        nbonds = nbonds + nbondsch(ich)
        nbends = nbends + nbendsch(ich)
        ntorts = ntorts + ntortsch(ich)
        ndeforms = ndeforms + ndeformsch(ich)
      end do
      maxat = nat
      temp_cvt_en = temp_cvt_en_ALL / dble(nch)
      maxbonds = nbonds
      maxbends = nbends
      maxtorts = ntorts
      maxdeforms = ndeforms
      allocate(x(3,maxat),v(3,maxat),f(3,maxat))
      allocate(xn(3,maxat),xn_1(3,maxat),accel(3,maxat))
      allocate(stretches(4,-maxbonds:maxbonds),torsions(maxtorts))
      allocate(bonds(3,maxbonds),bends(6,maxbends))
      allocate(torts(10,maxtorts),deforms(8,maxdeforms))
      allocate(chain(maxat),atomtype(maxat))
      allocate(listex(40,maxat),listmex(maxat))
      allocate(list14(40,maxat),listm14(maxat))
      allocate(list(maxnay,maxat),listm(maxat),listsubbox(maxdim3))
      allocate(lists(maxnay/2,maxat),listms(maxat))
      allocate(mass(maxat),massinv(maxat),massred(maxbonds))
      allocate(d2(maxbonds),dbond(maxdeforms))
      allocate(iaromatic(4,maxdeforms),idpar(maxdeforms))
      allocate(bondcon(3,maxbonds))
      allocate(sumdel(3,maxat),delx(3,maxat))
      allocate(fewald(3,maxat),elf(3,maxat))
      allocate(fshort(3,maxat),fnow(3,maxat))
      allocate(px(maxat),py(maxat),pz(maxat))
      allocate(ldummy(maxat),ldummybond(maxbonds))
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
!C
!C     *****determine which bonds make up each bend
!C
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
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
        ibaseb = nbendex(ich)    ! PDMS
        do itort = 1,ntortsch(ich)
          CALL read_a_line(11,line)
          ittort = ittort + 1
          read (line,*) (torts(j,ittort), j=1,5)
!C
!C     *****determine which bonds make up each torsion
!C
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
              torts(8,ittort) = imult*ibond
              goto 231
            end if
          end do
          write (6,*) 'torsion:third bond not found'
231       continue
!C
!C     ***** determine which bends make up the torsion
!C
          ib1 = torts(1,ittort)
          ib2 = torts(2,ittort)
          ib3 = torts(3,ittort)
          do ib = 1,nbendsch(ich)
            ibend = ibaseb + ib
            icheck1 = bends(1,ibend)
            icheck2 = bends(2,ibend)
            icheck3 = bends(3,ibend)
            imult = 1
            if (icheck2 .eq. ib2) then
              if (ib1 .eq. icheck1 .and. ib3 .eq. icheck3) then
                torts(9,ittort) = imult*ibend
                goto 291
              end if
              if (ib1 .eq. icheck3 .and. ib3 .eq. icheck1) then
                imult = -1
                torts(9,ittort) = imult*ibend
                goto 291
              end if
            end if
          end do
          write(6,*)'torsion: first bend not found'
 291      continue
          ib1 = torts(2,ittort)
          ib2 = torts(3,ittort)
          ib3 = torts(4,ittort)
          do ib = 1,nbendsch(ich)
            ibend = ibaseb + ib
            icheck1 = bends(1,ibend)
            icheck2 = bends(2,ibend)
            icheck3 = bends(3,ibend)
            imult = 1
            if (icheck2 .eq. ib2) then
              if (ib1 .eq. icheck1 .and. ib3 .eq. icheck3) then
                torts(10,ittort) = imult*ibend
                goto 292
              end if
              if (ib1 .eq. icheck3 .and. ib3 .eq. icheck1) then
                imult = -1
                torts(10,ittort) = imult*ibend
                goto 292
              end if
            end if
          end do
          write(6,*)'torsion: second bend not found'
 292      continue
        end do
      end do

      write (6,*) 'reading deformations'
      do ich = 1,nch
        ibase = nbondex(ich)
        do ideform = 1,ndeformsch(ich)
          CALL read_a_line(11,line)
          itdefs = itdefs + 1
          read (line,*) (deforms(j,itdefs), j=1,5),idpar(itdefs)
!C
!C     *****determine which bonds make up each deformation
!C
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
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
            if (icheck1 .eq. ib1 .and.  icheck2 .eq. ib2) then
              deforms(8,itdefs) = imult*ibond
              goto 431
            end if
          end do
          write (6,*) 'deform:third bond not found'
431       continue
        end do
      end do
!C
!C     *****determine chain of each atom
!C
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
     
      close(11)
      return 
      end  subroutine read11

      subroutine read12()

      implicit none
      integer number,intert,itype,jtype,nq,iinter,jinter,icheck,jcheck
      integer inontype,jnontype,inb,irow,icol,j,innb,ibond,ibend
      integer itort,ideform,ii,n2
      real(8) a,b,c,d,remass,f1,f2,f3,f4,f5
      character*2 int
      character*3 check,label1,label2
      logical*2 l2
      character*21 label
      character*6 nnb_label
      character*144 line

      integer idummy,ndummytypes,i,NNN

      ncharge = 0
      lpolarizable=.false.
      open(12,file="ff.dat",status="old")
      CALL read_a_line(12,line)
      read (line,*) number

      intert = 0
      NNN = number*(number+1)/2
      allocate(map_inverse_innb(NNN))
      allocate(typen(number,number),typee(number,number))
      allocate(nonbonded(6,NNN),electrostatic(NNN,2))
      allocate(map(NNN))
      allocate (zs(0:maxpoints))
      maxcharges = number
      maxtypes = number
      maxnnb = NNN
      call spline_interpol_coef_alloc()
      allocate(non_of_charge(maxcharges))
      allocate(is_charge_distributed(maxcharges))
      allocate(q_reduced(maxcharges)) ; q_reduced=0.0d0
      allocate(atmasses(maxtypes),q(maxcharges),pol(maxcharges))
      allocate(atom_labels(maxcharges),non_labels(maxtypes))
      allocate(lonepairtype(maxcharges))
      allocate(entr(maxtypes,Nzslice+1))
      allocate(fotr(maxtypes,Nzslice+1))
      allocate(vitr(2,maxtypes,Nzslice+1))
      allocate(LpGeomType(maxcharges))
      allocate(typedummy(maxcharges))
      allocate(adummy(maxcharges),bdummy(maxcharges),cdummy(maxcharges))
      print*, 'A1'
      do itype = 1,number
        do jtype = itype,number
          intert = intert + 1
          typen(itype,jtype) = intert
          typen(jtype,itype) = intert
          map_inverse_innb(intert)%i = itype
          map_inverse_innb(intert)%j = jtype 
         end do
      end do
       print*, 'map_inverse_innb%i=',map_inverse_innb%i
       print*, 'map_inverse_innb%j=',map_inverse_innb%j
      number_nnb = intert
      CALL read_a_line(12,line)
      write(6,*) "Nonboned self-terms"
      do itype = 1,number
        nq = 0
        read (line,'(a3)') non_labels(itype)                 
        intert =  typen(itype,itype)
        read (line(4:144),*) a,b,c,d,remass
        write(6,'(a3,x,4F15.2,F10.4)')non_labels(itype),a,b,c,d,remass
        nonbonded(1,intert) = a*1000.0d0
        nonbonded(2,intert) = b*1000.0d0 
        nonbonded(3,intert) = c*1000.0d0
        electrostatic(intert,2) = -d*2.0d0
        atmasses(itype) = remass
!C      
!C       *******look for charges for this type
!C
        CALL read_a_line(12,line)
        if (line(1:1) .ne. ' ') then
          nq = nq + 1
          write(int,'(i2)') nq
          ncharge = ncharge +  1
          q(ncharge) = 0.0d0
          pol(ncharge) = 0.0d0
          atom_labels(ncharge) = non_labels(itype)//int
          write (6,*) 'no charge and polarizability for atom ',&
                          atom_labels(ncharge),               &
                            ' was found:  a zero charge is assumed'
          goto 134
         else
          do while (line(1:1) .eq. ' ')
            nq = nq + 1
            write(int,'(i2)') nq
            ncharge = ncharge + 1
!C
!C      added in v1.0
!C
            lonepairtype(ncharge)=.false.
            if (non_labels(itype).eq."Lp".or. non_labels(itype).eq."LP") then
              lonepairtype(ncharge)=.true.       ! added by Oleg
            endif
!C
!C     end of addition
!C
            atom_labels(ncharge) = non_labels(itype)//int
            read (line(2:144),*) q(ncharge),pol(ncharge)
            write(6,'("Q=",F12.4," pol.=",F12.4)') q(ncharge),pol(ncharge)
            if (pol(ncharge).gt.1.0e-4) lpolarizable=.true.
            non_of_charge(ncharge) = itype                   
            CALL read_a_line(12,line)
          end do
        end if
134     continue  
      end do
!C
!C     *****assign default for off diagonal
!C          a = gm   b = gm   c = gm
!C
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
!C
!C     *****read non-default nonbonded interactions
!C
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
        write(6,'(a3,x,a3,4F15.2,F10.4)') label1,label2,a,b,c,d
        nonbonded(1,intert) = a*1000.0d0
        nonbonded(2,intert) = b*1000.0d0
        nonbonded(3,intert) = c*1000.0d0
        electrostatic(intert,2) = -d*2.0d0
        CALL read_a_line(12,line)
      end do

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
      q_reduced(irow) =  q(irow)/dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
      do icol = 1,ncharge
       itype = typee(irow,icol)
       electrostatic(itype,1) = q(irow)*q(icol) / Red_Vacuum_EL_permitivity_4_Pi
      end do
      end do
!C
!C     *****valence force field
!C
      CALL read_a_line(12,line)
      read (line,*) nbondt
      maxbtypes = nbondt
      allocate(stretch(5,maxbtypes))
      allocate(linear(maxbtypes))
      linear = .false.
      write(6,*) "Bonds"
      do ibond = 1,nbondt
        CALL read_a_line(12,line)
        label = line(1:11)
        read (line(12:144),*) f1,f2,f3
        write(6,'(a10,x,5F12.3)') label,f1,f2,f3
        stretch(1,ibond) = f1
        stretch(2,ibond) = f2
        stretch(3,ibond) = f3
      end do
      CALL read_a_line(12,line)
      read (line,*) nbendt
      allocate(bend(4,nbendt))
      write(6,*) "Bends"
      do ibend = 1,nbendt
        CALL read_a_line(12,line)
        label = line(1:15)
        read (line(16:144),*) f1,f2,l2
        f3=0.0d0
        f4=0.0d0
        write(6,'(a15,x,2F12.3,l2)') label,f1,f2,l2
        bend(1,ibend) = f1
        bend(2,ibend) = f2
        bend(3,ibend) = f3
        bend(4,ibend) = f4
        linear(ibend) = l2
      end do
      CALL read_a_line(12,line)
      read (line,*) ntortt
      allocate(twist(0:maxfolds,ntortt),nprms(ntortt))
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
      allocate(deform(ndeformt))
      write(6,*) "Deformations"
      do ideform = 1,ndeformt
        CALL read_a_line(12,line)
        label = line(1:20)
        read (line(21:144),*) f1
        write(6,'(a20,x,F10.3)') label,f1
        deform(ideform) = f1
        write(6,*) label,f1
      end do
 114  format (a10,10f5.1)
      CALL read_a_line(12,line)
      read (line,*) ndummytypes
      maxdummy = ndummytypes
      allocate(jikdummy(5,maxdummy))
      write(6,*) "Dummy atoms"
      do i=1,maxcharges
        typedummy(i)=0
      end do 
      do itype=1,ndummytypes
        CALL read_a_line(12,line)
        label = line(1:20)
        read(line(21:144),*) i,adummy(itype),bdummy(itype), &
                          cdummy(itype),LpGeomType(itype)
        write(6,'(a20,I3,3F12.4,I3)')label,i,adummy(itype), &
           bdummy(itype),cdummy(itype),LpGeomType(itype)
        typedummy(i)=itype
      end do 
      write(6,*) "finished reading force field"

      close(12)

      is_charge_distributed = .false.

      end subroutine read12

      subroutine read25()

      implicit none
      character*12 integ,elect
      character*19 polter
      character*144 line
      real(8) gammma
      open(25,file="mdrun.params",status="old")

      CALL read_a_line(25,line)
      read (line,*) nve, nvt, npt,lprofile
      CALL read_a_line(25,line)
      read (line,*) rread,driftmax,delt,nsteps
      CALL read_a_line(25,line)
      read (line,*) tstart,pfix(1),pfix(2),pfix(3),nnos,qfreq,wfreq
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
      read (line(13:72),*) alphai,klimitx,klimity,klimitz,lewald
      CALL read_a_line(25,line)
      read (line,'(a12)')integ 
      read (line(13:72),*) multimed,multibig,ros
      CALL read_a_line(25,line)
      read (line,'(a19)')polter 
      read (line(20:72),*) tolpol,tapering,rtaper,epsrf
      write (6,*)"tolpol,tapering,rtaper,epsrf",tolpol,tapering,rtaper,epsrf
      read(25,*) ; read(25,*) i_boundary_CTRL, gammma
      write (6,*)"i_boundary_CTRL=",i_boundary_CTRL, gammma
      Ew_gamma = gammma*alphai/dsqrt(gammma**2+alphai**2)
      Ew_beta = gammma*alphai/dsqrt(gammma**2+2.0d0*alphai**2) 
      close(25)

! convert them in internal units.
      delt = delt / 1000.0d0 ! convert from fs to internal unit which is ps
      pfix(:) = pfix(:) * 1.0d5 / unit_pressure ! pressure enters in atm. (10^5 N/m^2)
      qfreq = 1.0d0/qfreq   ! time-scale of thermostat enters in ps
      wfreq = 1.0d0/wfreq   ! time-scale of barostat enters in ps
      
!      epsilonp = ?????? 
!      repsilonp = ?????? ASK
      end  subroutine read25

      subroutine read26()

      implicit none
      integer kk,iat,lstart,istart,iend,icharge,itype,imap,inos
      character*144 line
      character*5 check1
      character*2 int
      real(8) shift(3)
      open(26,file="coords.inp",status="old")
!C
!C
!C     *****begin read
!C
      shift(1) = 10000.d0
      shift(2) = 10000.d0
      shift(3) = 10000.d0

      ldummy(:) = .false.

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

        lstart = iend + 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:istart+2),'(a3)') check1

        lstart = iend + 1
        CALL parse_line(line,lstart,istart,iend)
        read(line(istart:iend),*) icharge
        write (int,'(i2)') icharge
        check1 = check1(1:3)//int
!C
!C       *****assign atom type
!C
        do itype = 1,maxcharges
          if (check1 .eq. atom_labels(itype)) then
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

        CALL read_a_line(26,line)
        read (line,*) (v(kk,iat),kk=1,3)
      end do
!C
!C     ***** read extended lagrangian variables
!C
      CALL read_a_line(26,line)
      read(line,*)box
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
!C
!C     *******shift positions of necessary
!C
      if (shift(1) .lt. 0.d0 .or. shift(2) .lt. 0.d0 .or. shift(3) .lt. 0.d0) then
        write (6,*) 'shifting positions'
        do iat = 1,nat
          do kk = 1,3
            x(kk,iat) = x(kk,iat) - shift(kk) + 1.d0
          end do
        end do
      end if

      close(26)
      v = v /unit_velocity  ! transform it in internal units
      end subroutine read26

      subroutine read_a_line(iunit,line)

      implicit none
!C
!C     *****shared variables
!C
      integer iunit
      character*144 line

1     read(iunit,'(a144)',end=100) line
      if (line(1:1) .eq. '*') goto 1
      return
100   continue
      write (6,*) 'end of file ',iunit,' was reached'
      stop
      end  subroutine read_a_line

      subroutine ewald_reciprocal()
      implicit none
      integer kk,jj,iat,kx,ky,kz,kmag2,klimit2,itype
      real(8) kxf,kyf,kzf
      real(8) front(3),front2(3),elfact,twopifact,fouralpha2inv
      real(8) cossum,sinsum,ci,dotik,sfh,prefact,f1,akvec,factor,rkmag2
      real(8) costerm(maxat),sinterm(maxat),recip,tmp,btens
      real(8) tvirtmp(3,3)

      twopifact = 2.0d0*pi/(box(1)*box(2)*box(3))
      fouralpha2inv = 1.0d0/(4.0d0*alpha*alpha)
      do kk=1,3
        front(kk) = 2.0d0*pi/box(kk)
        front2(kk) = front(kk)*front(kk)
      end do
      elfact = 2.0d0*twopifact ! not multiplied by front
      klimit2=(klimitx*klimitx+klimity*klimity+klimitz*klimitz)/3.0d0


      recip = 0.0d0
      fewald(:,:) = 0.0d0
      elf(:,:) = 0.0d0
      tvirtmp(:,:) = 0.0d0
      do kx = 0,klimitx
        if (kx .eq. 0) then 
           factor = 1.0d0
         else
           factor = 2.0d0
        endif
        kxf=dble(kx)*front(1)
        do ky = -klimity,klimity
          kyf=dble(ky)*front(2)
          do kz = -klimitz,klimitz
            kzf=dble(kz)*front(3)
            kmag2 = kx*kx + ky*ky + kz*kz
            if (kmag2 .gt. klimit2 .or. kmag2 .eq. 0)goto 342
            rkmag2= kxf*kxf + kyf*kyf + kzf*kzf
            btens = 2.0d0*(1.0d0+rkmag2*fouralpha2inv)/rkmag2
            akvec = factor*dexp(-rkmag2*fouralpha2inv)/rkmag2
            cossum = 0.0d0
            sinsum = 0.0d0
            do iat = 1,nat
              itype = atomtype(iat)
              ci = q(itype)
              dotik=x(1,iat)*kxf+x(2,iat)*kyf+x(3,iat)*kzf
!print*, x(:,iat),kxf,kyf,kzf
              costerm(iat) = dcos(dotik)
              sinterm(iat) = dsin(dotik)
              cossum = cossum + costerm(iat)*ci
              sinsum = sinsum + sinterm(iat)*ci
!if (iat >= 984.and.iat<=985) then 
!print*, kx,ky,kz,iat,dotik, dcos(dotik), ci, costerm(iat)*ci,cossum
!print*, iat,x(:,iat)
!read(*,*)
!endif
            end do
            sfh = (cossum*cossum + sinsum*sinsum)*akvec
            recip = recip + sfh
            prefact = akvec*elfact
            do iat = 1,nat
              tmp=prefact*(cossum*sinterm(iat)-sinsum*costerm(iat))
              elf(1,iat) = elf(1,iat)+tmp*kxf
              elf(2,iat) = elf(2,iat)+tmp*kyf
              elf(3,iat) = elf(3,iat)+tmp*kzf
              f1 = q(atomtype(iat))*tmp
              fewald(1,iat) = fewald(1,iat)+f1*kxf
              fewald(2,iat) = fewald(2,iat)+f1*kyf
              fewald(3,iat) = fewald(3,iat)+f1*kzf
            end do
            tvirtmp(1,1)=tvirtmp(1,1)+sfh*(1.0-btens*kxf*kxf)
            tvirtmp(1,2)=tvirtmp(1,2)+sfh*(-btens*kxf*kyf)
            tvirtmp(1,3)=tvirtmp(1,3)+sfh*(-btens*kxf*kzf)
            tvirtmp(2,2)=tvirtmp(2,2)+sfh*(1.0-btens*kyf*kyf)
            tvirtmp(2,3)=tvirtmp(2,3)+sfh*(-btens*kyf*kzf)
            tvirtmp(3,3)=tvirtmp(3,3)+sfh*(1.0-btens*kzf*kzf)

 342       continue
          end do
        end do
      end do
      tvirtmp(3,1)=tvirtmp(1,3)
      tvirtmp(3,2)=tvirtmp(2,3)

      recip = recip*twopifact/Red_Vacuum_EL_permitivity_4_Pi     !*332.08d0 !- ewself
!C
      do jj = 1,3
        do kk = 1,3
          tvirpo(kk,jj)=tvirpo(kk,jj)+tvirtmp(kk,jj)*twopifact/Red_Vacuum_EL_permitivity_4_Pi
        end do
      end do

       print*, 'entered in reciprocal ewald with energy =',unbd*temp_cvt_en
       vir = vir + recip
       unbd = unbd + recip
       print*, 'reciprocal energy and unbd=',recip*temp_cvt_en,unbd*temp_cvt_en
       print*, 'box=',box
read(*,*)
!C       write(6,'("recip=",F12.3)') recip
!C
      return
      end  subroutine ewald_reciprocal

      subroutine results()

      implicit none
      integer i,kk,jj,kstar,iprop,inos
      real(8) vol,ekb,eks,ekt,ept,ekk,ekp,ekc,ekm,hamilton
      SAVE kstar
      data kstar /0/

      open (65,file='fort.65',status='old',access='append')

      toten = poten + kinen

      vol = box(1)*box(2)*box(3)
      ekb = 0.5d0*wtmass(1)*(vlogv(1)*vlogv(1) +vlogv(2)*vlogv(2) + vlogv(3)*vlogv(3))
      eks = 0.5d0*qtmass(1)*vlogs(1)*vlogs(1)
      ekt = gn3kt*xlogs(1)
      ept = 1.4582454d-5*vol*(pfix(1)+pfix(2)+pfix(3))
      ekk = kinen
      ekp = poten
      ekc = 0.0d0
      ekm = 0.0d0
      do inos = 2,nnos
        ekc = ekc + 0.5d0*qtmass(inos)*vlogs(inos)*vlogs(inos)
        ekm = ekm + gkt*xlogs(inos)
      end do
      hamilton = ekb + eks + ekt + ept + ekk + ekp + ekc + ekm

      prop(1) = temp
!C      prop(2) = pres
      prop(2) = stress(3,3)
      prop(3) = box(3)
      prop(25) = box(1)
      prop(26) = box(2)
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

      if (kount .eq. kinit)then
        do i = 1,maxprop
          avgsumpr(i) = 0.0d0
        end do
      end if
!C
!C     *****calculate diffusion
!C
      CALL diffusion()
!C
!C     *****keep track of sums over last nave iterations
!C
      do i = 1,maxprop
        sumpr(i) = sumpr(i) + prop(i)
        sumsqpr(i) = sumsqpr(i) + prop(i)*prop(i)
      end do
!C
!C     *****update overall averages & output them every 'nave' iterations
!C
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
      write(65,'(f11.2,f9.2,f12.3,f10.4,2(f12.2,1X),2(f10.4,1X))')&
            dble(kount)*stepout,(avgpr(iprop),iprop=1,5),avgpr(25),avgpr(26)
      end if

      if (printout) then
        if ( kount .le. nsteps)then
          write(6,220)kount,box,pres*fp/Vol,temp,toten*temp_cvt_en
          if (lstress)then
            write(6,*)'              *****   Stress Tensor  ***** '
            write(6,'(f15.2)')dble(kount)*stepout
            do kk = 1,3
              write(6,255)(stress(kk,jj)*fp,jj=1,3)
            end do
          end if
        end if
      end if
 220  format('Kount =',i7,' Box = ',3(f8.4,1X),' P =',f11.5,' T = ',f9.3,' E = ',f11.3)
 221  format (1x,4f15.6)
 255  format(8x,3(f15.5,1X))
      close (65)

      return
      end subroutine results
      subroutine setup()

      implicit none
      integer iat,itype,jat,jtype,ii,imap,i,kk
      real(8) total,ro3,betae,beta,sumninj,value
      real(8) vcum(3)
      integer isum
      real(8) rint,rsum,rrf3,tau,w ! used for mu-mu damping func. calc
!C
!C     *****chvol stuff
!C
      if (chvol) then
       do kk=1,3
        deltbox(kk) = (boxnew(kk)-box(kk))/nsteps
       end do
      endif
!C
!C     *****check total charge of system
!C
      total = 0.0d0
      do iat = 1,nat
        itype = atomtype(iat)
        total = total + q(itype)
      end do
      write (6,*) '*****total charge = ',total
!C
!C     *****eliminate center of mass velocity
!C
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
!C
!C     *****calculate inverse masses for stochast and shake
!C
      do iat = 1,nat
        if (.not.ldummy(iat)) then
          massinv(iat) = 1.0d0/mass(iat)
         else
          massinv(iat) = 0.0d0
         endif
      end do
!C
!C     *****assign nonbonded cutoffs
!C
      ro = rread
      ro2 = ro*ro
      rcut = ro + driftmax
      rcut2 = rcut*rcut
!C
!C     *****calculate sums for ptrunc and etrunc
!C
      if (lprofile) then
        CALL surftrunc()       ! explicit calculation of truncation
      endif
!C
!C     *****calculate sums for ptrunc and etrunc
!C
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
          sumninj = sumninj + nonbonded(3,imap)
        end do
      end do
      sumninj  = sumninj/ro3
      psumninj = sumninj*beta
      esumninj = sumninj*betae
!C
!C
!C     *****Polarization stuff: mu-mu tapering function constants  
!C
!C     tauden is the denomenator for tau tapering function
!C
      if (tapering) then 
         tauden = ((ro-rtaper)**3)/6.0d0
       else     ! if no tapering set rtaper=ro, so that tau=1,dtau=0
         tauden = 1.0d0
         rtaper=ro
      end if
!C
!C     *****calculation of arf
!C
      rsum = 0.0d0
      do isum = 1,1000
        rint = rtaper+(ro-rtaper)*(0.5+dble(isum))/1000.0d0
        rsum = rsum + rint*rint*tau(rint)
      end do
      rsum = rsum*3.0d0*(ro-rtaper)/1000.0d0
      rrf3 = rsum + rtaper**3
      arf = (epsrf - 1.0d0)/(rrf3*(epsrf + 0.5d0))
!C
!C
      if (constrain) then ! bond length constraints
        do i = 1,nbondt
          stretch(2,i) = stretch(3,i)
        end do
      end if

      do iat = 1,nat
        do kk = 1,3
          sumdel(kk,iat) = 0.0d0
          delx(kk,iat)   = 0.0d0
        end do
      end do

      end subroutine setup

      subroutine shake(xref,xo,xshaken)

      implicit none
      real(8) xref(3,maxat),xo(3,maxat),xshaken(3,maxat)

      integer kk,jj,iat,jat,ibond,ih,ihyd,ihx,ic,icarb,iter,iatom,i
      integer iatom_m1,iatom_p1,icarb1,icarb2,icarb3
      real(8) g,facti,factj,c1mass,c2mass,c3mass,dv,dvi,eta,etac1,etac2
      real(8)  hmass,sigma,rijmag,dot,rijshaken,delta,etac3,boxinv(3)
      real(8) xrefun(3,maxat),xoun(3,maxat),xouno(3,maxat)
      real(8) rijref(3,maxbonds)
      real(8) force(3),fact,delt2inv
      real(8) dhdxaref(3,3,3,maxdeforms)
      real(8) dhdxa(3,3,3,maxdeforms)
      real(8) dotsuma(maxdeforms,3),denoma(maxdeforms,3)
      real(8) bxa(3),aminusb(3),avect(3),bvect(3),p(3),vect(3),xp(3,2)

      CALL unwrap(xref,xrefun)
      CALL unwrap(xo,xoun)
!C     *****store unwrapped initial coordinates if calculating pressure

      if (newpress) then
        do iat = 1,nat
          xouno(1,iat) = xoun(1,iat)
          xouno(2,iat) = xoun(2,iat)
          xouno(3,iat) = xoun(3,iat)
        end do
      end if

!C     *****calculate rijref

      do ibond = 1,nbcon
          iat = bondcon(1,ibond)
          jat = bondcon(2,ibond)
          do  kk = 1,3
            rijref(kk,ibond) = xrefun(kk,iat) - xrefun(kk,jat)
          end do
       end do
!C
!C     *****calculate dhdxref and dhdx
!C     *****aromatic hydrogens
!C
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
              dotsuma(i,ihx) = dotsuma(i,ihx) &
                                 +  massinv(icarb)* &
             (dhdxa(ihx,ic,1,i)*dhdxaref(ihx,ic,1,i) &
             +dhdxa(ihx,ic,2,i)*dhdxaref(ihx,ic,2,i) &
             +dhdxa(ihx,ic,3,i)*dhdxaref(ihx,ic,3,i))
            end do
            denoma(i,ihx) = 1.0d0/(hmass + dotsuma(i,ihx))
          end do
        end do
      end if
!C
!C     *****begin iterative application of constraints
!C
 
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
!C
!C     *****do aromatic hydrogens
!C
        do i = 1,numaromatic
!C
!C     *****calculate predicted hydrogen positions
!C
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
!C
!C     *****calculate aminusb
!C     *****calculate dv
!C
          dv = 0.d0
          do kk = 1,3
            aminusb(kk) = avect(kk) - bvect(kk)
            dv = dv + aminusb(kk)*aminusb(kk)
          end do
          dvi = dbond(i)/dsqrt(dv)
!C
!C     *****calculate  v
!C
          do kk = 1,3
            vect(kk) = aminusb(kk)*dvi
          end do
!C
!C     *****calculate hydrogen positions
!C
          do kk = 1,3
            xp(kk,1) = xoun(kk,iatom) + idpar(i)*vect(kk) 
          end do
!C
!C     *****calculate constrained hydrogen posititions
!C
          do ihx = 1,3
            eta = xp(ihx,1) - xoun(ihx,ih)
            sigma = sigma + eta*eta
            eta = eta*denoma(i,ihx)*hmass
!c            eta = eta*denoma(i,ihx)/hmass
            xoun(ihx,ih) = eta*hmass + xoun(ihx,ih)
            etac1 = eta*c1mass     
            xoun(1,icarb1) = xoun(1,icarb1)-etac1*dhdxaref(ihx,1,1,i)
            xoun(2,icarb1) = xoun(2,icarb1)-etac1*dhdxaref(ihx,1,2,i)
            xoun(3,icarb1) = xoun(3,icarb1)-etac1*dhdxaref(ihx,1,3,i)
            etac2 = eta*c2mass     
            xoun(1,icarb2) = xoun(1,icarb2)-etac2*dhdxaref(ihx,2,1,i)
            xoun(2,icarb2) = xoun(2,icarb2)-etac2*dhdxaref(ihx,2,2,i)
            xoun(3,icarb2) = xoun(3,icarb2)-etac2*dhdxaref(ihx,2,3,i)
            etac3 = eta*c3mass     
            xoun(1,icarb3) = xoun(1,icarb3)-etac3*dhdxaref(ihx,3,1,i)
            xoun(2,icarb3) = xoun(2,icarb3)-etac3*dhdxaref(ihx,3,2,i)
            xoun(3,icarb3) = xoun(3,icarb3)-etac3*dhdxaref(ihx,3,3,i)
          end do
        end do
!C
!C     *****check for convergence
!C
      sigma = sigma/nconst

! print*, 'in shake sigma tol=',sigma,tol
      if (sigma .gt. tol) goto 1
!C
!C     *****apply periodic boundary conditions to shaken unwrapped coor.
!C
      do kk = 1,3
        boxinv(kk) = 1.0d0/box(kk)
      end do
      
      if (i_boundary_CTRL==0) then
      do iat = 1,nat
        do kk = 1,2
          xshaken(kk,iat) = xoun(kk,iat)
          if (xshaken(kk,iat) .gt. box(kk)) &
             xshaken(kk,iat) = xshaken(kk,iat) - box(kk)*(int(xshaken(kk,iat)*boxinv(kk)))
          if (xshaken(kk,iat) .lt. 0.0)     &
             xshaken(kk,iat) = xshaken(kk,iat)- box(kk)*(int(xshaken(kk,iat)*boxinv(kk)) - 1)
        end do
      end do
      else
      do iat = 1,nat
        do kk = 1,3
          xshaken(kk,iat) = xoun(kk,iat)
          if (xshaken(kk,iat) .gt. box(kk)) &
             xshaken(kk,iat) = xshaken(kk,iat) - box(kk)*(int(xshaken(kk,iat)*boxinv(kk)))
          if (xshaken(kk,iat) .lt. 0.0)     &
             xshaken(kk,iat) = xshaken(kk,iat)- box(kk)*(int(xshaken(kk,iat)*boxinv(kk)) - 1)
        end do
      end do
      endif
!C
!C     *****calculate virial if position shake
!C
      if (newpress) then
        delt2inv = 1.0d0/(delt*delt)
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

      end subroutine shake

     subroutine spline()

      implicit none
      integer nfunct,ipoint,ifunct,innb,icoeff
      integer pivot(6),numpoints
      real(8) aa(6),cc(6,6),xtemp(6),z,r2min
      real(8) fs(0:maxpoints), fsd(0:maxpoints), fsdd(0:maxpoints)
      real(8) coefff2(3,maxpoints,maxnnb)
      real(8) coefff3(3,maxpoints,maxnnb)
      real(8) coefff4(3,maxpoints,maxnnb)
      real(8) coefff5(3,maxpoints,maxnnb)
      real(8) coefff6(3,maxpoints,maxnnb)
      real(8) gtaper,staper


      nfunct = 3
      r2min = 1.0d0
      numpoints = maxpoints
      deltaspline = 0.3d0
!C
!C     *****initialize independent variable vector
!C
!C
      do ipoint = 0,numpoints
        zs(ipoint) = r2min + (ipoint)*deltaspline
      end do

      CALL cload(cc,deltaspline)
      CALL gauss(cc,6,6,pivot)
!C
!C     *****determine coefficient matrix 
!C
      do ifunct = 1,nfunct
        do innb = 1,maxnnb
          do ipoint = 0,numpoints
            z = zs(ipoint)
            fs(ipoint) = funct(z,innb,ifunct)
            fsd(ipoint) = functd(z,innb,ifunct)
            fsdd(ipoint) = functdd(z,innb,ifunct)
          end do

          do ipoint = 1,numpoints

            aa(1) = fs(ipoint-1)
            aa(2) = fs(ipoint)
            aa(3) = fsd(ipoint-1)
            aa(4) = fsd(ipoint)
            aa(5) = fsdd(ipoint-1)
            aa(6) = fsdd(ipoint)
!C
!C     *****solve for coefficients
!C
            CALL gsolve(cc,6,6,aa,xtemp,pivot)

            coeff1(ifunct,ipoint,innb) = xtemp(1)
            coeff2(ifunct,ipoint,innb) = xtemp(2)
            coeff3(ifunct,ipoint,innb) = xtemp(3)
            coeff4(ifunct,ipoint,innb) = xtemp(4)
            coeff5(ifunct,ipoint,innb) = xtemp(5)
            coeff6(ifunct,ipoint,innb) = xtemp(6)

            coefff2(ifunct,ipoint,innb) = xtemp(2)*2.0d0
            coefff3(ifunct,ipoint,innb) = xtemp(3)*4.0d0
            coefff4(ifunct,ipoint,innb) = xtemp(4)*6.0d0
            coefff5(ifunct,ipoint,innb) = xtemp(5)*8.0d0
            coefff6(ifunct,ipoint,innb) = xtemp(6)*10.0d0
          end do
        end do
      end do
!C
!C     *****calculate totals
!C
      do innb = 1,maxnnb
        do ipoint = 1,numpoints
          do ifunct = 1,nfunct
            z = zs(ipoint)
            gtaper = (dsqrt(z) - (ros))/driftmax
            staper = 1.0d0+(gtaper*gtaper*(2.0d0*gtaper-3.0d0))
            coefft(1,ipoint,innb) = coefft(1,ipoint,innb) + coeff1(ifunct,ipoint,innb)
            coefft(2,ipoint,innb) = coefft(2,ipoint,innb) + coeff2(ifunct,ipoint,innb)
            coefft(3,ipoint,innb) = coefft(3,ipoint,innb) + coeff3(ifunct,ipoint,innb)
            coefft(4,ipoint,innb) = coefft(4,ipoint,innb) + coeff4(ifunct,ipoint,innb)
            coefft(5,ipoint,innb) = coefft(5,ipoint,innb) + coeff5(ifunct,ipoint,innb)
            coefft(6,ipoint,innb) = coefft(6,ipoint,innb) + coeff6(ifunct,ipoint,innb)
            coeffft(2,ipoint,innb) = coeffft(2,ipoint,innb) + coefff2(ifunct,ipoint,innb)
            coeffft(3,ipoint,innb) = coeffft(3,ipoint,innb) + coefff3(ifunct,ipoint,innb)
            coeffft(4,ipoint,innb) = coeffft(4,ipoint,innb) + coefff4(ifunct,ipoint,innb)
            coeffft(5,ipoint,innb) = coeffft(5,ipoint,innb) + coefff5(ifunct,ipoint,innb)
            coeffft(6,ipoint,innb) = coeffft(6,ipoint,innb) + coefff6(ifunct,ipoint,innb)
          end do
          if (z .lt. rshort2)then
            do icoeff = 2,6
              coeffft1(icoeff,ipoint,innb) = coeffft(icoeff,ipoint,innb)
              if (z .gt. ros2)then
             coeffft1(icoeff,ipoint,innb) = coeffft(icoeff,ipoint,innb)*staper
              end if
            end do
          end if
        end do
      end do

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

      end subroutine spline
      double precision FUNCTION tau(r)
      real(8) r
      real(8) w

      w= r-rtaper

      tau = 1.0d0
      if (r.gt.rtaper) then
        tau = tau + w*w*(w/3.0d0+0.5*(rtaper-ro))/tauden
      endif

      end FUNCTION tau

      real(8) FUNCTION dtau(r)
      real(8) r

      if (r.gt.rtaper) then
        dtau = ((r-ro)*(r-rtaper)**2)/tauden
       else
        dtau=0.0d0
      end if

      end FUNCTION dtau

      subroutine unwrap(xrf,xunrf)

      implicit none
      real(8) xrf(3,maxat),xunrf(3,maxat)
      integer iat,ibond,kk,jat
      real(8) diff
      
      el(:) = box(:)*0.5d0
      xunrf(1:3,1:nat) = xrf(1:3,1:nat)
!C
!C     *****unwrap by bonds
!C
        do ibond = 1,nbonds
          iat = bonds(1,ibond)
          jat = bonds(2,ibond)
          do kk = 1,3
            diff = xrf(kk,jat) - xrf(kk,iat)
            if (abs(diff) .gt. el(kk)) diff = diff - dsign(box(kk),diff)
            xunrf(kk,jat) = xunrf(kk,iat) + diff
          end do
        end do

      end subroutine unwrap

      subroutine surftrunc()
      implicit none
      real(8) rhoz(maxtypes,Nzslice+1)
      real(8) subh,vsub,dz,delz2,r0c2,denum
      real(8) etr,vrtr,ftr,vztr
      integer itype,iat,jtype,ii,cij,idzetta,idz,zz,iz,imap
       do itype=1,maxtypes
         do iz=1,Nzslice
           rhoz(itype,iz)=0.0d0
         enddo
       enddo 
       subh=box(3)/dble(Nzslice)
       vsub=subh*box(1)*box(2)
       do iat=1,nat
         itype=non_of_charge(atomtype(iat))
         iz=int(x(3,iat)/subh+1.0d-6)+1
         rhoz(itype,iz)=rhoz(itype,iz)+1.0d0/vsub
       enddo

       do itype=1,maxtypes
         do iz=1,Nzslice
           etr=0.0d0
           ftr=0.0d0
           vrtr=0.0d0
           vztr=0.0d0
           zz=(dble(iz)+.5d0)*subh
           do jtype=1,maxtypes
             ii = typen(itype,jtype)
             Cij=nonbonded(3,ii)
             do idzetta=-Nzslice+1,2*Nzslice
               dz=(dble(idzetta)+.5d0)*subh
               delz2=(zz-dz)**2
               if(delz2 .lt. ro2) then
                 r0c2=ro2-delz2 
               else
                 r0c2=0.0d0 
               endif
               denum=delz2+r0c2
               idz=idzetta
               if(idzetta.lt.1)idz=idzetta+Nzslice
               if(idzetta.gt.Nzslice)idz=idzetta-Nzslice
!C   ***  Trucation Energy 
               etr=etr+subh*pi*Cij*rhoz(jtype,idz)/(2.d0*denum**2)
!C   ***  Truncation Force
               ftr=ftr-subh*2.d0*pi*Cij*rhoz(jtype,idz)*(zz-dz)/denum**3
!C   ***  Truncation virial
               vrtr=vrtr+subh*4.d0*pi*Cij*rhoz(jtype,idz)*(delz2+3.d0*r0c2)/(2.d0*denum**3)
               vztr=vztr+subh*4.d0*pi*Cij*rhoz(jtype,idz)*delz2/denum**3
             enddo
           enddo
       
        entr(itype,iz)=etr*0.5d0       ! check 
        fotr(itype,iz)=ftr
        vitr(1,itype,iz)= vrtr*0.5d0
        vitr(2,itype,iz)= vztr*0.5d0
        enddo
      enddo
         
      end subroutine surftrunc

      subroutine exclude()
      implicit none
      integer i,j,iat,nex,ibond,ix,iex,icheck,iy,ineigh,itype
      integer jat,jtype,kat,k,ibend,itmp,i14a,i14b,ibond1,ibond2
      logical HasBeenUsed
      logical Found14
      do iat=1,nat
        listm14(iat)=0
        listmex(iat)=0
      end do     ! end of additions in v3.0

      do 10000, iat = 1, nat - 1  ! search tables for each atom
        nex=0  ! initialized number excluded to 0
!C
!C     *****determine all 2 center interactions with iat
!C
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
          do 2700, ibend=1,nbends
            if (bends(1,ibend) .eq. iat ) then ! exclude
              ix=bends(3,ibend)
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
!C
!C     *****do three center
!C
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
!C
!C     *****four center
!C
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
!C
!C     *****do three center
!C
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
!C
!C
!C    make a list of all 1-4 centers for reduced 1-4 interactions 
!C
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
!C
!C  remove 1-4 that are part of listex (matters for cyclic molecules) (v2.7b)
!C
                 do i=1,listmex(i14a)
                   if (listex(i,i14a).eq.i14b) HasBeenUsed=.true.
                 end do

                 if (.not.HasBeenUsed) then
                   listm14(i14a)=listm14(i14a)+1
                   list14(listm14(i14a),i14a)=i14b
                 end if
               end if 
             end do
          end do
        endif ! (lredonefour)

      return
      end subroutine exclude
!C
!C     *** DoDummy subroutine puts two dummy atoms based upon bend coordinates
!C         and distributes the force on them correctly
!C         see jcc 1999_v20_p786
!C         v2.0 ether-based and carbonyl-based Lps are treated separately
!C 
      Subroutine DoDummy(fdummy)
      implicit none
      integer idummy,idat,kk,jj,itype,jat,kat,iat
      real(8) aa,bb,cc,rid(3),rij(3),rjk(3),rik(3),rjd(3),rkd(3),factor
      real(8) rij_a_rjk(3),absrij_a_rjkinv,f1(3),Fd(3) ! rij_a_rjk  = r_ij + a*r_jk
      real(8) fdummy(3,maxat),ftmp,Fddotrid,Fddotrjk,rjkabs,rjkabs2
      real(8) tmp1,tmp2,tmp3
      real(8) rijabs2,rikabs2
      real(8) cross(3)     ! cross product of r_ij x r_ik
      real(8) crossdotFd   ! [r_ij x r_ik] . Fd
      real(8) rijdotrik    ! rij dot product rik
      real(8) crossabs
      real(8) fiout(3),fjout(3),fkout(3)
      real(8) zero
      parameter (zero=1.0d-12)

      el(:) = box(:)*0.5d0
      virdummy=0.0d0   ! virdummy and tvirdummy are common variables
      do kk=1,3
       tvirdummy(kk,1)=0.0d0
       tvirdummy(kk,2)=0.0d0
       tvirdummy(kk,3)=0.0d0
      end do

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
        do kk=1,2
          rij(kk)=x(kk,jat)-x(kk,iat)
          if (abs(rij(kk)).gt.el(kk))rij(kk)=rij(kk) - dsign(box(kk),rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)

          rjk(kk)=x(kk,kat)-x(kk,jat)
          if (abs(rjk(kk)).gt.el(kk)) rjk(kk)=rjk(kk) - dsign(box(kk),rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)

          rik(kk)=x(kk,kat)-x(kk,iat)
          if (abs(rik(kk)).gt.el(kk))  rik(kk)=rik(kk) - dsign(box(kk),rik(kk))

          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        end do
        kk = 3
        if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
          rij(kk)=x(kk,jat)-x(kk,iat)
!          if (abs(rij(kk)).gt.el(kk))
!     $        rij(kk)=rij(kk) - dsign(box(kk),rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)

          rjk(kk)=x(kk,kat)-x(kk,jat)
!          if (abs(rjk(kk)).gt.el(kk))
!     $       rjk(kk)=rjk(kk) - dsign(box(kk),rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)

          rik(kk)=x(kk,kat)-x(kk,iat)
!          if (abs(rik(kk)).gt.el(kk))
!     $       rik(kk)=rik(kk) - dsign(box(kk),rik(kk))

          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        else
          rij(kk)=x(kk,jat)-x(kk,iat)
          if (abs(rij(kk)).gt.el(kk)) &
             rij(kk)=rij(kk) - dsign(box(kk),rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)

          rjk(kk)=x(kk,kat)-x(kk,jat)
          if (abs(rjk(kk)).gt.el(kk)) &
            rjk(kk)=rjk(kk) - dsign(box(kk),rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)

          rik(kk)=x(kk,kat)-x(kk,iat)
          if (abs(rik(kk)).gt.el(kk)) &
            rik(kk)=rik(kk) - dsign(box(kk),rik(kk))

          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        endif
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

        if (LpGeomType(itype).le.2) then

!C        [rij x rik]
!C
          cross(1)= rij(2)*rik(3) - rik(2)*rij(3)
          cross(2)=-rij(1)*rik(3) + rik(1)*rij(3)
          cross(3)= rij(1)*rik(2) - rik(1)*rij(2)
          crossdotFd=cross(1)*Fd(1)+cross(2)*Fd(2)+cross(3)*Fd(3)
!C
!C       grad w.r.t iat, jat, kat
!C
          crossabs=dsqrt(rijabs2*rikabs2-rijdotrik**2)
          if (abs(crossabs).lt.zero) crossabs=zero
          tmp1=1.0/crossabs

          fiout(1)=(Fd(2)*(rik(3)-rij(3))+Fd(3)*(rij(2)-rik(2)))*tmp1
          fiout(2)=(Fd(1)*(rij(3)-rik(3))+Fd(3)*(rik(1)-rij(1)))*tmp1
          fiout(3)=(Fd(1)*(rik(2)-rij(2))+Fd(2)*(rij(1)-rik(1)))*tmp1

          fjout(1)=(Fd(2)*(-rik(3))+Fd(3)*rik(2))*tmp1
          fjout(2)=(Fd(3)*(-rik(1))+Fd(1)*rik(3))*tmp1
          fjout(3)=(Fd(1)*(-rik(2))+Fd(2)*rik(1))*tmp1

          fkout(1)=(Fd(2)*rij(3)-Fd(3)*rij(2))*tmp1
          fkout(2)=(Fd(3)*rij(1)-Fd(1)*rij(3))*tmp1
          fkout(3)=(Fd(1)*rij(2)-Fd(2)*rij(1))*tmp1

          tmp3=tmp1**3
          do kk=1,3
            fiout(kk)=fiout(kk) + crossdotFd*tmp3*(rikabs2*rij(kk) + &
            rijabs2*rik(kk) - rijdotrik*(rik(kk) + rij(kk)))
            fjout(kk)=fjout(kk) - crossdotFd*tmp3*(rikabs2*rij(kk) - &
            rijdotrik*rik(kk))
            fkout(kk)=fkout(kk) - crossdotFd*tmp3*(rijabs2*rik(kk) - &
            rijdotrik*rij(kk))
          end do

          do kk=1,2
            rid(kk)=rid(kk)+cc*cross(kk)*tmp1
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
             rkd(kk)=x(kk,idat)-x(kk,kat)
             if (abs(rkd(kk)).gt.el(kk)) &
               rkd(kk)=rkd(kk)-dsign(box(kk),rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
             if (abs(rjd(kk)).gt.el(kk)) &
               rjd(kk)=rjd(kk)-dsign(box(kk),rjd(kk))
          end do
          kk = 3
          if (i_boundary_CTRL == 0) then ! SLAB ORIENTED ALONG OZ
            rid(kk)=rid(kk)+cc*cross(kk)*tmp1
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
!            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
!            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
             rkd(kk)=x(kk,idat)-x(kk,kat)
!             if (abs(rkd(kk)).gt.el(kk))
!     $          rkd(kk)=rkd(kk)-dsign(box(kk),rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
!             if (abs(rjd(kk)).gt.el(kk))
!     $          rjd(kk)=rjd(kk)-dsign(box(kk),rjd(kk))
          else
            rid(kk)=rid(kk)+cc*cross(kk)*tmp1
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
             rkd(kk)=x(kk,idat)-x(kk,kat)
             if (abs(rkd(kk)).gt.el(kk)) &
               rkd(kk)=rkd(kk)-dsign(box(kk),rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
             if (abs(rjd(kk)).gt.el(kk)) &
               rjd(kk)=rjd(kk)-dsign(box(kk),rjd(kk))
          endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!C
!C  *** calculate forces on atoms jat,iat,kat due to lone pair
!C  *** reuse tmp1,tmp3 below with a different meaning 

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
             tvirdummy(kk,1)=tvirdummy(kk,1)-(rid(1)*tmp1+rjd(1)*tmp2+rkd(1)*tmp3)
             tvirdummy(kk,2)=tvirdummy(kk,2)-(rid(2)*tmp1+rjd(2)*tmp2+rkd(2)*tmp3)
             tvirdummy(kk,3)=tvirdummy(kk,3)-(rid(3)*tmp1+rjd(3)*tmp2+rkd(3)*tmp3)
           endif
          end do
        endif    !  (LpGeomType(itype).le.2)
!C
!C    handle lone pairs on carbonyls (LpGeomType = 3)
!C
        if (LpGeomType(itype).eq.3) then
          do kk=1,2
            rid(kk)=rid(kk)+cc*rjk(kk)/rjkabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
             rkd(kk)=x(kk,idat)-x(kk,kat)
             if (abs(rkd(kk)).gt.el(kk))  &
              rkd(kk)=rkd(kk)-dsign(box(kk),rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
             if (abs(rjd(kk)).gt.el(kk)) &
              rjd(kk)=rjd(kk)-dsign(box(kk),rjd(kk))
          end do
          kk = 3
          if (i_boundary_CTRL == 0) then
            rid(kk)=rid(kk)+cc*rjk(kk)/rjkabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
!            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
!            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
             rkd(kk)=x(kk,idat)-x(kk,kat)
!             if (abs(rkd(kk)).gt.el(kk))
!     $         rkd(kk)=rkd(kk)-dsign(box(kk),rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
!             if (abs(rjd(kk)).gt.el(kk))
!     $         rjd(kk)=rjd(kk)-dsign(box(kk),rjd(kk))
          else
            rid(kk)=rid(kk)+cc*rjk(kk)/rjkabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
             rkd(kk)=x(kk,idat)-x(kk,kat)
             if (abs(rkd(kk)).gt.el(kk)) &
              rkd(kk)=rkd(kk)-dsign(box(kk),rkd(kk))
             rjd(kk)=x(kk,idat)-x(kk,jat)
             if (abs(rjd(kk)).gt.el(kk)) &
              rjd(kk)=rjd(kk)-dsign(box(kk),rjd(kk))
          endif
!C
!C  *** calculate forces on atoms jat,iat,kat due to lone pair
!C  *** reuse tmp1,tmp3 below with a different meaning 
!C
          Fddotrjk=0.0d0
          do kk=1,3
            Fddotrjk=Fddotrjk+rjk(kk)*Fd(kk)
          end do
          do kk=1,3
           ftmp=factor*(Fd(kk) - f1(kk))
           tmp1=Fd(kk)-ftmp  
           tmp2=(1.0d0-aa)*ftmp-cc*(Fd(kk)-Fddotrjk*rjk(kk)/rjkabs2)/rjkabs
           tmp3=aa*ftmp + cc*(Fd(kk)-Fddotrjk*rjk(kk)/rjkabs2)/rjkabs
           fdummy(kk,iat)=fdummy(kk,iat)+tmp1
           fdummy(kk,jat)=fdummy(kk,jat)+tmp2
           fdummy(kk,kat)=fdummy(kk,kat)+tmp3
           if (newpress) then
             virdummy=virdummy-(rid(kk)*tmp1+rjd(kk)*tmp2+rkd(kk)*tmp3)
             tvirdummy(kk,1)=tvirdummy(kk,1)-(rid(1)*tmp1+rjd(1)*tmp2+rkd(1)*tmp3)
             tvirdummy(kk,2)=tvirdummy(kk,2)-(rid(2)*tmp1+rjd(2)*tmp2+rkd(2)*tmp3)
             tvirdummy(kk,3)=tvirdummy(kk,3)-(rid(3)*tmp1+rjd(3)*tmp2+rkd(3)*tmp3)
           endif
          end do
        end if   !  (LpGeomType(itype).eq.3)
!C       zero the force on the dummy atom after it has been distributed
        do kk=1,3
          fdummy(kk,idat)=0.0d0
        end do
      end do     ! next idummy

!D     if (newpress) then
!D      write(6,*) "virdummy=",virdummy*2.2857d+04/(box(1)*box(2)*box(3))
!D     endif
      return
      end  Subroutine DoDummy
!C
!C     *** DoDummy subroutine puts two dummy atoms based upon bend coordinates
!C         and distributes the force on them correctly
!C         see jcc 1999_v20_p786
!C 
      Subroutine DoDummyCoords()
      implicit none
      integer idummy,idat,kk,itype,jat,kat,iat
      real(8) aa,bb,cc,rid(3),rij(3),rjk(3),rik(3),factor
      real(8) rij_a_rjk(3),absrij_a_rjkinv       ! rij_a_rjk  = r_ij + a*r_jk
      real(8) rjkabs2,rjkabs
      real(8) zero
      real(8) rijabs2,rikabs2
      real(8) cross(3)        ! cross product of r_ij x r_ik
      real(8) rijdotrik    ! rij dot product rik
      real(8) crossabs
      parameter (zero=1.0d-12)

      do kk=1,3
        el(kk)=box(kk)*0.5d0
      end do
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
        do kk=1,2
          rij(kk)=x(kk,jat)-x(kk,iat)
          if (abs(rij(kk)).gt.el(kk))  &
            rij(kk)=rij(kk) - dsign(box(kk),rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)

          rjk(kk)=x(kk,kat)-x(kk,jat)
          if (abs(rjk(kk)).gt.el(kk))  &
             rjk(kk)=rjk(kk) - dsign(box(kk),rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)

          rik(kk)=x(kk,kat)-x(kk,iat)
          if(abs(rik(kk)).gt.el(kk)) &
            rik(kk)=rik(kk)-dsign(box(kk),rik(kk))

          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        end do
        kk = 3
        if (i_boundary_CTRL == 0) then
          rij(kk)=x(kk,jat)-x(kk,iat)
!          if (abs(rij(kk)).gt.el(kk))
!     $       rij(kk)=rij(kk) - dsign(box(kk),rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)

          rjk(kk)=x(kk,kat)-x(kk,jat)
!          if (abs(rjk(kk)).gt.el(kk))
!     $        rjk(kk)=rjk(kk) - dsign(box(kk),rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)

          rik(kk)=x(kk,kat)-x(kk,iat)
!          if(abs(rik(kk)).gt.el(kk))
!     $       rik(kk)=rik(kk)-dsign(box(kk),rik(kk))

          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk) 
        else
          rij(kk)=x(kk,jat)-x(kk,iat)
          if (abs(rij(kk)).gt.el(kk)) &
            rij(kk)=rij(kk) - dsign(box(kk),rij(kk))
          rijabs2=rijabs2+rij(kk)*rij(kk)

          rjk(kk)=x(kk,kat)-x(kk,jat)
          if (abs(rjk(kk)).gt.el(kk)) &
             rjk(kk)=rjk(kk) - dsign(box(kk),rjk(kk))
          rjkabs2=rjkabs2+rjk(kk)*rjk(kk)
          rij_a_rjk(kk)=rij(kk)+aa*rjk(kk)
          absrij_a_rjkinv=absrij_a_rjkinv+rij_a_rjk(kk)*rij_a_rjk(kk)

          rik(kk)=x(kk,kat)-x(kk,iat)
          if(abs(rik(kk)).gt.el(kk)) &
            rik(kk)=rik(kk)-dsign(box(kk),rik(kk))

          rikabs2=rikabs2+rik(kk)*rik(kk)
          rijdotrik=rijdotrik+rij(kk)*rik(kk)
        endif

        rjkabs=dsqrt(rjkabs2)
        crossabs=dsqrt(rijabs2*rikabs2-rijdotrik**2)
        if (abs(crossabs).lt.zero) crossabs=zero
        absrij_a_rjkinv=1.0d0/dsqrt(absrij_a_rjkinv)
        factor=-bb*absrij_a_rjkinv
        do kk=1,3
          rid(kk)=factor*rij_a_rjk(kk)
        end do
        if (LpGeomType(itype).le.2) then
!C
!C         [rij x rik]
!C
          cross(1)= rij(2)*rik(3) - rik(2)*rij(3)
          cross(2)=-rij(1)*rik(3) + rik(1)*rij(3)
          cross(3)= rij(1)*rik(2) - rik(1)*rij(2)
          do kk=1,3
            rid(kk)=rid(kk)+cc*cross(kk)/crossabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (kk==3.and.i_boundary_CTRL == 0) then
            else
            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
            endif
          end do
        end if     ! (LpGeomType(itype).le.2)

        if (LpGeomType(itype).eq.3) then
          do kk=1,3
            rid(kk)=rid(kk)+cc*rjk(kk)/rjkabs
            x(kk,idat)=x(kk,iat)+rid(kk)  ! place dummy atom
            if (kk==3.and.i_boundary_CTRL == 0) then
            else
            if (x(kk,idat).gt.box(kk)) x(kk,idat)=x(kk,idat)-box(kk)
            if (x(kk,idat).lt.0) x(kk,idat)=x(kk,idat)+box(kk)
            endif
          end do
        end if     ! (LpGeomType(itype).eq.3)
      end do       ! do idummy=1,ndummy
      return
      end  Subroutine DoDummyCoords
!C
!C *** DoDummyInit() subroutine assigns lone pairs with C>0
!C     cc=abs(cdummy) and cc=-abs(cdummy)
!C     It assings a positive c to the first dummy atom (force center)
!C     of a lone pair and a negative c to the second dummy atom of a lone pair
!C     It also modifies listex() and list14()
!C 
!C     ! IMPORTANT ! It needs to be called in initialize right after exclude()        
!C     but before boxes()
!C
      Subroutine DoDummyInit()
      implicit none
      integer nCallDummyInit
      integer ibond,ibend,ineigh,iox,iex,jtmp,j,kk,ipos
      integer idummy,jdummy,idat,iat,jat,itype,jtype,iatConnect
      real(8) cci
      logical HasBeenFound
      logical HasBeenChanged(maxdummy)
 
      data nCallDummyInit/0/
      save nCallDummyInit

      if (nCallDummyInit.ge.1) then
        write(6,*) "DoDummyInit() needs to be called only once"
        write(6,*) "Multiple calls to DoDummyInit() are found"
        return
      end if
      nCallDummyInit=nCallDummyInit+1 
!C
!C *** for all dummy atoms find the corresponding bond and then bends
!C
      idat=0
      do idummy=1,nat
        if (ldummy(idummy)) then
          idat=idat+1
          itype=typedummy(atomtype(idummy))
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
            write(6,*)"Cannot find an atom the Lp ",idummy," is connected to"
            write(6,*) "Program exits"
            stop
          end if
          if (LpGeomType(itype).le.2) then  ! 1 (1Lp) or 2 (2Lp ether-like) 
            do ibend=1,nbends
              if (bends(2,ibend).eq.iat.and.bends(1,ibend).ne.idummy.and.bends(3,ibend).ne.idummy) then
                 jikdummy(1,idat)=bends(1,ibend)
                 jikdummy(2,idat)=bends(2,ibend)
                 jikdummy(3,idat)=bends(3,ibend)
                 jikdummy(4,idat)=idummy
                 jikdummy(5,idat)=itype
               endif
            end do      ! plus-minus (up-down) of Lp in a pair will be assigned later
!C
!C  *** for carbonyl-like Lps find the atom (iatConnect) to which the iat is connected to
!C  *** and than find jat and kat using bend information
!C
          else if (LpGeomType(itype).eq.3) then  ! type 3 means carbonyl-like Lps
            do ibond=1,nbonds
             if (bonds(1,ibond).eq.iat.and.(.not.ldummy(bonds(2,ibond))))then
               iatConnect=bonds(2,ibond)
               HasBeenFound=.true.
             end if
             if (bonds(2,ibond).eq.iat.and.(.not.ldummy(bonds(1,ibond))))then
               iatConnect=bonds(1,ibond)
               HasBeenFound=.true.
             end if
            end do
            if (.not.HasBeenFound) then
             write(6,*)"Cannot find an atom needed to define Lp=",idummy
             write(6,*) "Program exits"
             stop
            endif
            do ibend=1,nbends
              if (bends(2,ibend).eq.iatConnect.and.bends(1,ibend).ne.iat.and.bends(3,ibend).ne.iat) then
                 jikdummy(1,idat)=bends(1,ibend)
                 jikdummy(2,idat)=iat
                 jikdummy(3,idat)=bends(3,ibend)
                 jikdummy(4,idat)=idummy
                 jikdummy(5,idat)=itype
              endif
            end do      ! plus-minus (up-down) of Lp in a pair will be assigned later
          else
            write(6,*) "LpGeomType should be 1,2,3 for this version" 
            write(6,*) "found LpGeomType",LpGeomType(itype)
            write(6,*) "Program will stop"
            stop
         end if    ! (LpGeomType(itype).le.1) then
        end if         
      end do
      write(6,'("ndummy=",I9)')ndummy
      write(6,'("idat=",I9)') idat
!C
!C     for all Lp atoms 
!C     1. Copy oxygen list14()
!C     2. find the connected oxygen and include in the listex() 
!C     for the Lp atom all atoms from the oxygen listex() except itself
!C
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
!C
!C  *** go through atoms below iox for which iox is excluded and included them
!C      in listex(kk,idummy)
!C
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
!C 
!C *** In order to observe required convention that jat=list(any_i,iat) has jat>iat 
!C     modify all exclude lists
!C
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
!C
!C  *** do the same for the 1-4 exclude list
!C
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
!C
!C  *** remove duplicates from the excluded list of dummy atoms
!C
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
!C
!C  ***  info below is for debugging purposes
!C
!D     write(6,*) "after correction for Lp"
!D     do iat=1,nat
!D       write(6,*)listmex(iat),"listex",(listex(j,iat),j=1,listmex(iat))
!D     end do
!D     write(6,*) "after correction for Lp"
!D     do iat=1,nat
!D       write(6,*)listm14(iat),"list14",(list14(j,iat),j=1,listm14(iat))
!D     end do
!C           
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
              write(6,*)"Two extended force centers have the same position", idummy,jdummy
            end if
          end do
!C
!C ***  inverse the polarity of the lone pair by changing bend from jat-iat-kat
!C ***  to kat-iat-jat
!C
        else if (LpGeomType(itype).eq.2.or.LpGeomType(itype).eq.3)then ! ether-like or carbonyl-like lone pair
          if (abs(cdummy(itype)).lt.1.d-6) then  
             write(6,*) "C param should be non-zero for this Lp type",iat,idummy
          end if
          if (abs(adummy(itype)).lt.1.d-6) then  
             write(6,*) "A param should be non-zero for this Lp type",iat,idummy
          end if
          HasBeenFound=.false.
          do jdummy=idummy+1,ndummy  ! find a twin
            jat=jikdummy(2,jdummy) 
            jtype=jikdummy(5,jdummy) ! a dummy atom type
            if (iat.eq.jat) then     ! two dummy force centers are connected to the same atom
               if (HasBeenFound) then
                 write(6,*)"Found more than one dummy center in a pair",idummy,jdummy
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
      end Subroutine DoDummyInit

!  !!!!!!!!!!!!!!!   ADDDED BY ME !!!!!!!! 
      subroutine Ew_2D_kis0_slowest()
      implicit none
      integer i,j,k,Natoms,itype,innb,jtype
      real(8) zij,U,ffsqrtPi_per_alpha,ff,AA,AZ,i_area
      real(8) sqrtPi_per_beta,sqrtPi_per_gamma,sqrtPi_per_alpha
      logical l_distributed_charge_i , l_distributed_charge_j
      real(8) local_potential

      Natoms = maxat   ! make sure that is correct

        sqrtPi_per_alpha= dsqrt(Pi)/alpha
        sqrtPi_per_beta = dsqrt(Pi)/Ew_beta
        sqrtPi_per_gamma = dsqrt(Pi)/Ew_gamma
        i_area = 1/(box(1)*box(2))
! Eval Fourier part for k=0
        local_potential = 0.0d0
        do i = 1, Natoms
        itype = atomtype(i)
        if (q(itype) /= 0.0d0 ) then
        l_distributed_charge_i = is_charge_distributed(itype)
        if (l_distributed_charge_i) then
        do j = i+1, Natoms
          zij = x(3,i) - x(3,j)
          jtype =  atomtype(j) 
          if (q(jtype) /= 0.0d0 ) then
          innb = typee(itype,jtype)
          l_distributed_charge_j = is_charge_distributed(jtype)
          if (l_distributed_charge_j) then   ! G G 
           AZ = Ew_beta*zij ; AA = sqrtPi_per_beta
            ff = 0.0d0   ! easiest way to fix atoms
          else                               ! G P
           AZ = Ew_gamma*zij ; AA = sqrtPi_per_gamma
           ff = derf(zij*Ew_gamma)
          endif
          U = AA*dexp(-(AZ*AZ))+zij*pi*derf(AZ)
          U = U *  (electrostatic(innb,1) * i_area)
          local_potential = local_potential + U 
          ff = ff * 2.0d0 * electrostatic(innb,1)*i_area
          f(3,i) = f(3,i) + ff
          f(3,j) = f(3,j) - ff
         endif
        enddo
        else !l_distributed_charge_i
         do j = i+1, Natoms
          zij = x(3,i) - x(3,j)
          jtype =  atomtype(j)
          if (q(jtype) /= 0.0d0 ) then
          innb = typee(itype,jtype)
          l_distributed_charge_j = is_charge_distributed(jtype)
          if (l_distributed_charge_j) then   ! P G
           AZ = Ew_gamma*zij ; AA = sqrtPi_per_gamma
           ff = derf(zij*Ew_gamma)
          else                               ! P P
           AZ = alpha*zij ; AA = sqrtPi_per_alpha
           ff = derf(zij*alpha)
          endif
          U = AA*dexp(-(AZ*AZ))+zij*pi*derf(AZ)
          U = U *  (electrostatic(innb,1) * i_area)
          local_potential = local_potential + U
          ff = ff * 2.0d0 * electrostatic(innb,1)*i_area
          f(3,i) = f(3,i) + ff
          f(3,j) = f(3,j) - ff
         endif ! q(jtype) ne 0
         enddo
        endif
        endif ! l_distributed_charge_i
        enddo
        unbd = unbd + local_potential
        print*, 'local_potential atk0 = ',local_potential*temp_cvt_en
        print*, 'unbd=',unbd*temp_cvt_en
       end subroutine Ew_2D_kis0_slowest
 
       subroutine Ew_2D_fourier_slow
        implicit none
        real(8), parameter :: factor_h_cut_off = 30.0d0 ! in units of 2Pi/Lz
        real(8) , parameter :: SAFE = 10.0d0 ! distance in Amstrom
        real(8) , parameter :: small = 0.01d0 ! a small distance (in Amstrom)
        real(8) fpi2,fa2,fct,fg2, z_limit,z_limit0
        real(8),allocatable :: GG(:,:,:),GA(:,:)
        real(8) Pi2,  angle,Sfactor_real,Sfactor_imag
        real(8) Sfactor,zi,kr,qi,zmin,zmax,t1,t2,h,fe2,rec1,rec2
        real(8) c_os , s_in, ff,Ssin,Scos, store_gg
        integer i,j,k,ih,itype, Natoms
        real(8), allocatable :: vct_cos(:),vct_sin(:), fx(:),fy(:),fz(:), enQfourier(:)
        real(8) FKK, h_integral,h_step, Lz, h_cut_off, i_area
      

        print*, 'In Fourier Ewald k > 0'
 
        Natoms = maxat   ! make sure that is correct
        allocate(vct_cos(Natoms),vct_sin(Natoms))
        allocate(GG(0:k_max_1,-k_max_2:k_max_2,-k_max_z:k_max_z))
        allocate(GA(0:k_max_1,-k_max_z:k_max_z))
        allocate(fx(Natoms),fy(Natoms),fz(Natoms))
        allocate(enQfourier(-k_max_z:k_max_z)) ; enQfourier = 0.0d0

        Pi2 = Pi*2.0d0
        fpi2 = 4.0d0*pi*pi
        fa2 = 1.0d0/(4.0d0*alpha*alpha)
        fg2 = 1.0d0/(4.0d0*Ew_gamma*Ew_gamma)
        fe2 = 1.0d0/(4.0d0*q_distrib_eta*q_distrib_eta)
        i_area = 1/(box(1)*box(2))
        zmin = minval(x(3,1:Natoms)) 
        zmax = maxval(x(3,1:Natoms))
        Lz =dabs(zmax-zmin) + SMALL !+ SAFE
        h_cut_off = factor_h_cut_off * pi2 / Lz ! + SAFE
        h_step = 2.0d0*h_cut_off/(2*k_max_z+1-1)
!print*, 'z_step andred=',z_step,z_step/(Pi2/z_limit0)
!print*, 'Lz=',Lz
        do k = 0,k_max_1
        rec1 = dble(k)/box(1)*Pi2
        do j = -k_max_2,k_max_2
        rec2 = dble(j)/box(2)*Pi2
         do ih = -k_max_z, k_max_Z
          h = h_cut_off * dble(ih) / dble(k_max_Z) 
          fct = rec1*rec1+rec2*rec2+h*h
!print*,ih, 'h hcut_off=',h,h_cut_off
!print*,k,j,ih,rec1**2+rec2**2,rec1**2,rec2**2,h**2
         GG(k,j,ih) = 1.0d0/fct*dexp(-fct*fa2)
!print*,k,j,ih,GG(k,j,ih), fct
         enddo
        enddo
        enddo

print*,'G(0 -k_max_2 -k_max_z)=', GG(0,-k_max_2,-k_max_z)

        do k = 0, k_max_1
         do ih = -k_max_z, k_max_Z
           h = dble(ih)/z_limit 
           GA(k,ih) = dexp(-(fpi2*dble(k*k)+h*h)*fg2 ) ! WRONG WRONG WRONG
         enddo
        enddo

        enQfourier = 0.0d0
        do k = 0, k_max_1
        rec1 = dble(k)/box(1)*Pi2
        FKK = dexp(-dble(k*k)*fe2)
        do j =  -k_max_2,k_max_2
        rec2 = dble(j)/box(2)*Pi2
!print*, 'max k =', dsqrt((dble(k_max_1)/box(1)*Pi2 )**2 + (dble(-k_max_2)/box(1)*Pi2)**2)/ (2*Pi/box(1))
     if (rec1**2+rec2**2 /= 0.0d0)  then
        do ih =  -k_max_z, k_max_Z
        h = h_cut_off * dble(ih) / dble(k_max_Z)
        if (k**2+j**2+ih**2 /= 0) then
!print*, ih,'h/(2Pi/limitz)=',h/(2*Pi/z_limit0)
        Sfactor_real=0.0d0
        Sfactor_imag = 0.0d0
        Scos = 0.0d0
        Ssin = 0.0d0
         do i = 1, Natoms
         if (q(atomtype(i)) /=0) then
            kr = rec1*x(1,i)+rec2*x(2,i)
            itype = atomtype(i)
            qi = q_reduced(itype)
            angle = kr+h*x(3,i)
            s_in = dsin(angle) ; c_os = dcos(angle)
            if (is_charge_distributed(itype)) then
             vct_cos(i) = FKK*c_os*qi
             vct_sin(i) = FKK*s_in*qi
             Scos = Scos + vct_cos(i)
             Ssin = Ssin + vct_sin(i)
             Sfactor_real = Sfactor_real + GA(k,ih)*vct_cos(i)
             Sfactor_imag = Sfactor_imag + GA(k,ih)*vct_sin(i)
            else
             vct_cos(i) = c_os*qi
             vct_sin(i) = s_in*qi
             Scos = Scos + vct_cos(i)
             Ssin = Ssin + vct_sin(i)
             Sfactor_real = Sfactor_real + vct_cos(i)
             Sfactor_imag = Sfactor_imag + vct_sin(i)
            endif
          endif ! (q(atomtype(i)) /=0)
         enddo
         store_gg = GG(k,j,ih)*i_area*h_step
         Sfactor = Sfactor_real*Sfactor_real + Sfactor_imag*Sfactor_imag
         enQfourier(ih) = enQfourier(ih) + Sfactor*store_gg
!print*, k,j,ih, enQfourier(ih), Sfactor,GG(k,j,ih)
         do i = 1, Natoms
           ff = store_gg*(vct_sin(i)*Scos-vct_cos(i)*Ssin)
           fx(i) = fx(i) + rec1*ff
           fy(i) = fy(i) + rec2*ff
           fz(i) = fz(i) + h*ff
         enddo
        endif ! (k**2+j**2+ih**2 /= 0)
        enddo
      endif ! k+j /= 0
        enddo
        enddo


do ih = -k_max_z,k_max_z
write(666,*) enQfourier(ih)*temp_cvt_en
enddo
       

        f(1,:) = f(1,:) + fx(:)
        f(2,:) = f(2,:) + fy(:)
        f(3,:) = f(3,:) + fz(:)        

 
        h_integral = sum(enQfourier)

PRINT*, 'h_integral = ',h_integral*temp_cvt_en
STOP


        unbd = unbd + h_integral
        print*, 'EnQFourier=',enQfourier
        print*, 'unbd=',unbd
read(*,*)
        deallocate(fx); deallocate(fy); deallocate(fz)
        deallocate(GA)
        deallocate(GG)
        deallocate(vct_cos); deallocate(vct_sin)
       end subroutine Ew_2D_fourier_slow



       subroutine Fourier_Ewald_driver
        implicit none

            if (lewald) then
             if (i_boundary_CTRL==0) then
              call Ew_2D_kis0_slowest
              call Ew_2D_fourier_slow
!             call Ew_correct_2D()
             else
              call ewald_reciprocal()
              call ewald_correct()
             endif
           endif

       end subroutine Fourier_Ewald_driver

       end module the_subroutines

        PROGRAM md
!C
!C     The ortho code has been created from the version v3.0ortho
!C
!C     polarizable md with mu-mu damping function, scaled mu-mu interactions
!C     The exclude() subroutine has been modified to handle lone pairs
!C     read25 has been modified to include redfactor and lredonefour
!C     read12 has been modified to include lonepairtype array
!C     if deform bond constant is zero, constrain the position of the atom
!C     en4cen() and improper() have been moved to the middle timestep
!C     v1.4a
!C     v1.2 dummy atoms have been added, array of ldummy is created in read26
!C     read12 has been changed
!C     dummy type is defined using the charge type in ff.dat
!C     if the atom is a dummy then bonds connected to it are not constrained
!C     bonds to dummy atoms must have zero force constant
!C     v2.0 damping of mu-mu interactions using A(Thole) a_thole has been added
!C     v2.1 1.0d-4*(12/b)**12/r^12 has been added to exp-6 potential
!C     v2.1g In order to remove slight center of mass drift in rather anisotropic
!C           systems (~0.0001 m/s after ~1 ns) due to truncation of dipole-dipole
!C           interactions and nonexact solution of induced dipole-dipole eq.s
!C           center of mass velocity of the system is set to zero every timestep
!C     v2.2 output of totdipole and chargeflux has been added
!C     v3.0ortho This code was made into orthorombic one
!C
!C     one needs to specify a bend C-O(Lp)-C to define a lone pair
!C
!C     for PDMS use correct etort=sum (1+cos) !!!
!C     List of 1-4 interactions is not made for the surface atoms
!C
!C     last modified 04/04/2004 by Oleg Borodin
!C
      use the_subroutines
      use other_data
      implicit none
      integer iat,kk
      real(8) vcum(3)
!C
!C     *****initializations
!C
      cline(1:42) = '*------------------------------------------'
      cline(43:72) = '------------------------------------------'
      clineo(1:42) = '------------------------------------------'
      clineo(43:72) =  '-------------------------------------------'
      clinec(1:42) = '#------------------------------------------'
      clinec(43:72) = '------------------------------------------'
      pi = dacos(-1.0d0)
      rootpi = dsqrt(pi)
!C
!C     *****read data files
!C
      call allocate_props
      CALL read25()        ! The control file
       write(6,*) 'read25'
      CALL read12()        ! The force field file
       write(6,*) 'read12'
      CALL read11()        ! The connectivity file
       write(6,*) 'read11'
      CALL read26()        !  "coords.inp"   - conversion required
       write(6,*) 'read26'
      CALL initialize()
      CALL intsetup()
      CALL checker()

      
      if (printout)write(6,220)kount,box,pres*(fp/box(1)/box(2)/box(3)),temp,toten*temp_cvt_en
!C
!C     ***************************************************************
!C
!C                  *******begin the dynamics loop*******
!C
!C     ***************************************************************
!C
800   continue
!C
!C     *****change volume if scaling to new value
!C
      if (chvol) then
       do kk=1,3
        boxold(kk) = box(kk)
        box(kk) = box(kk) + deltbox(kk)
        do iat = 1,nat
          x(kk,iat) = x(kk,iat)*(box(kk)/boxold(kk))
          xn(kk,iat) = xn(kk,iat)*(box(kk)/boxold(kk))
          xn_1(kk,iat) = xn_1(kk,iat)*(box(kk)/boxold(kk))
        end do
       end do
      end if
!C
!C     *****make new nblist every nlist steps
!C
      if (update) then
        CALL boxes()
        update = .false.
      endif

      CALL integrator()
!C
!C     *****eliminate center of mass velocity ADDED in md_nodamping.f by Oleg
!C
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
!C
!C     end of addition
!C
      CALL output1()

      if(kount.le.nsteps) go to 800

      CALL output2()

      stop ! THE END
!C
!C     *****formats
!C
 220  format('Kount =',i7,' Box = ',3(f8.4,1X),' P =',f11.5,' T = ',f9.3,' E = ',f11.3)

      end program md


