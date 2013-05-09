!
!     commonds for md_damping.f  v2.8
!
      common /cdefs/atom_labels,non_labels,cline,clineo,clinec
      common /charg/ncharge,number_nnb,nions
      common /props/temp,kinen,toten,poten,unbd,unbde
      common /check/idim,listsubbox
      common /typem/typee,typen,map
      common /non/nonbonded,electrostatic
      common /sysinput/driftmax,rread,ro,ro2,rcut,rcut2,rscale,
     +                  tol,tstart,qfreq,wfreq,delt,stepout
      common /polarize/rop,rscalep,epsilonp,repsilonp
      common /pressure/pres,pintr,psumninj,pfix,pke,ptrunc,vir
      common /potlst/twist,stretch,bend,deform,nprms
      common /potlst2/nbondt,nbendt,ntortt,ndeformt
      common /time1/nsteps,nave,noutput,ninit,nvir,nnben
      common /time2/kount,kave,kvir,knben,kinit,koutput
      common /time3/nboxdip,ncoords,nvelocs,nstress
      common /time4/kboxdip,kcoords,kvelocs,kstress
      common /cnsts/pi,kb,gask,gaskinkc,avogadro,rootpi,delzero
      common /diffuse/delx,sumdel,spol,totmassinv,nstart
      common /coords/x,xn,xn_1,v,f,accel
      common /con4cen/stretches,torsions
      common /connect1/bonds,bends,torts,deforms
      common /connect3/nat,nch,nbonds,nbends,ntorts,ndeforms
      common /connect4/q,pol,chain,atomtype
      common /volume/box,boxini,boxnew,deltbox,boxold,el2
      common /logic1/inter,constrain,update,tbpol,lredonefour,
     $   lonepairtype,lredQ_mu14
      common /logic2/fixtemp,chvol,printout,nve,nvt,npt,ex14,newpress
      common /logic3/lboxdip,lcoords,lvelocs,lstress
      common /ex/listex,listmex
      common /neighbour/list,listm
      common /onefourlist/ list14,listm14
      common /ener/ebond,ebend,etort,eopbs,esumninj
      common /splpar/zs,intzs0
      common /spcoeffs/coefft,coeffft,coeffee,coefffee
      common /splc/coeff1,coeff2,coeff3,coeff4,coeff5,coeff6
      common /splft/deltaspline
      common /masses/mass,massinv,massred,atmasses
      common /shakec/bondcon,iaromatic,nconst,nbcon,numaromatic,ndof
      common /shakec2/dbond,d2,idpar
      common /stats1/sumpr,sumsqpr,avgpr,avgsqpr,avgsumpr,avgsumsqpr
      common /stats2/prop,stdev,simpr,simsqpr
      common /ptensor/tvirpo,prtrunc,prtkine,stress
      common /nhctime/ nhcstep1,nhcstep2,nnos
      common /thermostat/qtmass,wdt1,wdt2,wdt4,wdt8
      common /barostat/wtmass,e2,e4,e6,e8
      common /kinetic/gkt,gnkt,gn1kt,ondf
      common /thermovar/glogs,xlogs,vlogs,glogv,xlogv,vlogv
C  
      common /ewaldcons/fewald,elf,ewself,alpha,alphai,klimit
      common /newmul/rshort,rshort2,ros,ros2,rsbox,rsbox2
      common /newmul2/fshort,coeffft1
      common /neighbours/lists,listms
      common /multis/fnow,multibig,multimed
C  
      common /pols1/px,py,pz,tolpol,epsrf,rtaper,tauden,arf,
     $    redfactor,redQmufactor
      common /poldipole/ DipoleTot,DipolePol
      common /logic4/tapering,lpolarizable
C
      common /comdummy1/ ldummy,ldummybond
      common /comdummy2/ jikdummy,ndummy,natreal
      common /comdummy3/ adummy,bdummy,cdummy,typedummy,LpGeomType
      common /comdummy4/ tvirdummy,virdummy
      common /comdummy5/ LpBendAtomType,ndummytypes   ! added in v2.13


