C    This file has been generated using SystemGenerator version 7.01
C  Parameters for unit and force center list
       integer maxnch          !Maximum number of molecules + atoms
          parameter (maxnch= 176)
       integer maxat           !Maximum number of force centers
          parameter (maxat= 8592)
C
C  Parameters for connectivity lists
       integer maxbonds        !Maximum number of bonds in system
          parameter (maxbonds= 8416)
       integer maxbends        !Maximum number of bends in system
          parameter (maxbends= 12400)
       integer maxtorts        !Maximum number of torsions in system
          parameter (maxtorts= 13776)
       integer maxdeforms      !Maximum number of deformations in system
         parameter (maxdeforms= 1) 
C
C  Parameters for force field
       integer maxtypes        !Max number of repulsion-dispersion types
          parameter (maxtypes= 10)
       integer maxcharges      !Max number of charge types
          parameter (maxcharges= 14)
       integer maxdummy        !Max number of dummy (lone pair) atoms
          parameter (maxdummy= 1680)
       integer maxbtypes       !Max of types (bonds,bends,torts)
          parameter (maxbtypes= 15)
       integer maxfolds        !Max number of terms in torsional energy
          parameter (maxfolds = 9 )
       integer maxbox
       parameter (maxbox = 50)
       integer maxnay          !Maximum neighbors per force center
          parameter (maxnay = 999)
       integer maxdim          !Maximum box subdivisions per axis
          parameter (maxdim = 50)
       integer maxdim3         !Total number of box subdivisions
          parameter (maxdim3 = maxdim*maxdim*maxdim)
C
C  Polarization related parameters
       real*8 a_thole          !Dipole-Dipole damping constant
          parameter (a_thole=0.20d0)               
C
C  Misc additional parameters as indicated (advanced)
       integer maxpoints       !Max points in interaction splines
          parameter (maxpoints = 501)
       integer maxprop         !Max number of system averaged properties
          parameter (maxprop = 30)
       integer maxnnb          !Max number of non-bonded interactions
          parameter (maxnnb = maxcharges*(maxcharges+1)/2)
       integer kmax            !Max k vectors in one direction
          parameter (kmax=6)  
