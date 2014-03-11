C
C     dimensions.h for md_damping.f v2.8
C
      real*8 stretch(3,maxbtypes),bend(2,maxbtypes)
      real*8 twist(0:maxfolds,maxbtypes),deform(maxbtypes)
      integer nprms(maxbtypes)
      real*8 nonbonded(6,maxnnb),electrostatic(maxnnb,2)
      integer typee(maxcharges,maxcharges),typen(maxtypes,maxtypes)
      integer map(maxnnb)
      character*72 cline,clineo,clinec
C
      real*8 x(3,maxat),v(3,maxat),f(3,maxat)
      real*8 xn(3,maxat),xn_1(3,maxat),accel(3,maxat)
      real*8 stretches(4,-maxbonds:maxbonds),torsions(maxtorts)
      integer bonds(3,maxbonds),bends(6,maxbends)
      integer torts(8,maxtorts),deforms(8,maxdeforms)
      integer chain(maxat),atomtype(maxat)
      integer*2 listex(40,maxat),listmex(maxat)
      integer*2 list14(25,maxat),listm14(maxat)     
      integer*2 list(maxnay,maxat),listm(maxat),listsubbox(maxdim3)
      integer*2 lists(maxnay/2,maxat),listms(maxat)
      real*8 coeffee(6,maxpoints,maxnnb),coefffee(6,maxpoints,maxnnb)
      real*8 coefft(6,maxpoints,maxnnb),coeffft(6,maxpoints,maxnnb)
      real*8 coeff1(3,maxpoints,maxnnb)
      real*8 coeff2(3,maxpoints,maxnnb)
      real*8 coeff3(3,maxpoints,maxnnb)
      real*8 coeff4(3,maxpoints,maxnnb)
      real*8 coeff5(3,maxpoints,maxnnb)
      real*8 coeff6(3,maxpoints,maxnnb)
      real*8 zs(0:maxpoints)
      real*8 mass(maxat),massinv(maxat),massred(maxbonds)
      real*8 d2(maxbonds),dbond(maxdeforms)
      integer iaromatic(4,maxdeforms),idpar(maxdeforms)
      integer bondcon(3,maxbonds)
      real*8 atmasses(maxtypes),q(maxcharges),pol(maxcharges)
      character*5 atom_labels(maxcharges)
      character*3 non_labels(maxtypes)
      logical newpress,inter,constrain,lredonefour,lredQ_mu14 
      logical lonepairtype(maxcharges)   ! added in v1.0
      logical fixtemp,chvol,printout,tbpol
      logical update,nvt,nve,npt,ex14
      logical lboxdip,lcoords,lvelocs,lstress
      real*8 redfactor,redQmufactor    ! added in v1.0
      real*8 kb,kinen
      real*8 sumdel(3,maxat),delx(3,maxat)
      real*8 sumpr(maxprop),sumsqpr(maxprop),avgpr(maxprop),
     +   avgsqpr(maxprop),avgsumpr(maxprop),avgsumsqpr(maxprop)
      real*8 prop(maxprop),stdev(maxprop)
      real*8 simpr(maxprop),simsqpr(maxprop)
      real*8 prtrunc(3,3),tvirpo(3,3),stress(3,3),prtkine(3,3)
C
      real*8 qtmass(10),wdt1(10),wdt2(10),wdt4(10),wdt8(10)
      real*8 wtmass(10),glogs(10),vlogs(10),xlogs(10)
C
      real*4 nave,nvir,nnben,noutput,ninit,nboxdip,ncoords,
     $    nvelocs,nstress 
C
      integer ncharge,number_nnb,nions,idim,nbondt,nbendt,ntortt
      integer ndeformt,nsteps,kount,kave
      integer kvir,knben,kinit,koutput,nstart,nat,nch,nbonds,nbends
      integer ntorts,ndeforms,intzs0,nconst,nbcon
      integer numaromatic,ndof,nhcstep1,nhcstep2,nnos
      real*8 temp,toten,poten,unbd,driftmax,rread,ro,ro2,rcut,rcut2
      real*8 rscale,tol,tstart,qfreq,wfreq,delt,stepout
      real*8 rop,rscalep,epsilonp,repsilonp
      real*8 pres,pintr,psumninj,pfix,pke,ptrunc,vir
      real*8 pi,gask,gaskinkc,avogadro,rootpi,delzero
      real*8 spol,totmassinv,box,boxini,boxnew,deltbox,boxold,el2
      real*8 ebond,ebend,etort,eopbs,esumninj
      real*8 unbde,deltaspline,gkt,gnkt
      real*8 gn1kt,ondf,e2,e4,e6,e8,glogv,xlogv,vlogv
      integer kboxdip,kcoords,kvelocs,kstress
C
      real*8 fewald(3,maxat),elf(3,maxat),ewself,alphai,alpha
      integer klimit
C
C
      real*8 fshort(3,maxat),fnow(3,maxat)
      real*8 coeffft1(6,maxpoints,maxnnb)
      integer multibig,multimed
      real*8 rshort,rshort2,ros,ros2,rsbox,rsbox2
C
      real*8 px(maxat),py(maxat),pz(maxat),tolpol,epsrf,rtaper,arf
      real*4 DipoleTot(3),DipolePol(3)
      real*8 tauden
      logical tapering,lpolarizable
C
C     additional declarations for dummy atoms
C
      logical ldummy(maxat),ldummybond(maxbonds)
      integer typedummy(maxcharges),ndummy,natreal
      integer jikdummy(5,maxdummy)
C     the type of geometry used to describe Lp
      integer LpGeomType(maxcharges)   
      real*8 adummy(maxcharges),bdummy(maxcharges),cdummy(maxcharges)
      real*8 tvirdummy(3,3),virdummy
      integer LpBendAtomType(3,maxcharges)  ! added in v2.13
      integer ndummytypes                   ! added in v2.13
