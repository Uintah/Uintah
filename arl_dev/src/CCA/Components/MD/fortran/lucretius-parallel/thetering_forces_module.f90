

module thetering_forces_module

contains

subroutine thetering_forces
use stresses_data, only : stress_thetering,stress
use ALL_atoms_data, only : xxx,yyy,zzz, fxx,fyy,fzz
use boundaries, only : periodic_images
use thetering_data, only : thetering
use energies_data, only : en_thetering
implicit none
integer i,j,k,N,iatom
real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz,fx,fy,fz,kx,ky,kz,x0,y0,z0
real(8),allocatable:: dx(:),dy(:),dz(:),local_pot(:)

if (thetering%N<1) RETURN

N= thetering%N
allocate(dx(N),dy(N),dz(N),local_pot(N)); local_pot=0.0d0;

do i = 1, N
 iatom=thetering%to_atom(i)
 x0 = thetering%x0(i); y0=thetering%y0(i) ; z0 = thetering%z0(i);
 dx(i) = xxx(iatom)-x0 ; dy(i) =yyy(iatom)-y0; dz(i) = zzz(iatom)-z0;
enddo

call periodic_images(dx,dy,dz)
sxx=0.0d0; syy=0.0d0; szz=0.0d0; syx=0.0d0; szx=0.0d0; szy=0.0d0; sxz=0.0d0; syz=0.0d0; sxy=0.0d0;
do i = 1, N
 iatom=thetering%to_atom(i)
 kx = thetering%kx(i);  ky = thetering%ky(i);  kz = thetering%kz(i);
 fx = -kx*dx(i) 
 fy = -ky*dy(i)
 fz = -kz*dz(i)
 local_pot(i) = (-fx*dx(i) - fy*dy(i) - fz*dz(i)) * 0.5d0;
 fxx(iatom) = fxx(iatom) + fx
 fyy(iatom) = fyy(iatom) + fy
 fzz(iatom) = fzz(iatom) + fz
 sxx = sxx + fx*dx(i) ; sxy = sxy + fx*dy(i) ; sxz = sxz + fx*dz(i)
 syx = syx + fy*dx(i) ; syy = syy + fy*dy(i) ; syz = syz + fy*dz(i)
 szx = szx + fz*dx(i) ; szy = szy + fz*dy(i) ; szz = szz + fz*dz(i)
enddo
en_thetering = sum(local_pot(1:N))
stress_thetering(1) = -sxx ; stress_thetering(2) = -syy ; stress_thetering(3) = -szz
stress_thetering(4) = -(sxx+syy+szz)/3.0d0; 
stress_thetering(5)=-sxy; stress_thetering(6) = -sxz; stress_thetering(7)=-syz
stress_thetering(8)=-syx; stress_thetering(9) = -szx; stress_thetering(10)=-szy
stress = stress + stress_thetering;
deallocate(dx,dy,dz,local_pot)
end subroutine thetering_forces

end module thetering_forces_module
