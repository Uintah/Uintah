      program massage

C     reads uintah mpm output and formats it
C     parameters are in massage.input

      implicit none

      character*20 label
      real*8 pistonvelocity
      real*8 lx,ly,lz,lpiston,luse
      real*8 ax,ay,az
      real*8 t,d1,d2,d3
      real*8 tr,d1r,d2r,d3r,tyr
      real*8 tx,ty,tz,strain
      integer outputfreq,index(20)
      real*8 property(20,100000)
      integer numentries,i,j
      real*8 pistonwork,tprev,fyprev
      real*8 fyave
      real*8 strainenergy
      real*8 initialthermal
      real*8 fractionaluminum,fractionnickel
      real*8 densityaluminum,densitynickel
      real*8 cpaluminum,cpnickel
      real*8 heatmass,temp
      real*8 volume,totalenergy
      character*12 labels(20)
      real*8 densitymaterial 
      real*8 c0
      real*8 S
      real*8 grun

      labels(1) = 'time'
      labels(2) = 'ly'
      labels(3) = 'strain y'
      labels(4) = 'Sxx'
      labels(5) = 'Szz'
      labels(6) = 'Syy'
      labels(7) = '2 X MS'
      labels(8) = 'PW'
      labels(9) = 'SE'
      labels(10) = 'TE'
      labels(11) = 'Temperature'
      labels(12) = 'KE'
      labels(13) = 'Total En'
      labels(14) = 'P'
      labels(15) = 'P MG w/o th'
      labels(16) = 'P MG'

C     *****read parameters

      open(100,file="massage.input",status="old")
      read(100,*) label, pistonvelocity
      read(100,*) label, outputfreq 
      read(100,*) label, lx          
      read(100,*) label, ly          
      read(100,*) label, lz          
      read(100,*) label, lpiston          
      read(100,*) label, fractionaluminum
      read(100,*) label, densityaluminum
      read(100,*) label, cpaluminum
      read(100,*) label, fractionnickel   
      read(100,*) label, densitynickel   
      read(100,*) label, cpnickel   
      read(100,*) label, densitymaterial 
      read(100,*) label, c0
      read(100,*) label, S
      read(100,*) label, grun
      close(100)

C     *****open data files

      open(101,file='BndyForce_xminus.dat',status="old")
      open(102,file='BndyForce_zplus.dat',status="old")
      open(103,file='BndyForce_yminus.dat',status="old")
      open(104,file='RigidReactionForce.dat',status="old")
      open(105,file='StrainEnergy.dat',status="old")
      open(106,file='ThermalEnergy.dat',status="old")
      open(107,file='KineticEnergy.dat',status="old")

C     *****read data

      pistonwork = 0.d0
      strainenergy = 0.d0
      tprev = 0.d0
      fyprev = 0.d0
      initialthermal = 1.D+66

      volume = lx*ly*lz
      heatmass = volume*densityaluminum*fractionaluminum*cpaluminum +
     +           volume*densitynickel*fractionnickel*cpnickel

      write (6,*) 'heat mass ',heatmass
      index(1) = 0
      index(2) = 0
      index(3) = 0
      index(5) = 0
      index(6) = 0
      index(7) = 0

C     *****read file 1

      numentries = 0
1     read(101,*,end=200) t,d1,d2,d3 
      index(1) = index(1) + 1
      if (mod(index(1),outputfreq) .ne. 1) goto 101
      numentries = numentries + 1
      luse = ly - pistonvelocity*t
      strain = pistonvelocity*t/ly
      ax = luse*lz
      ay = lx*lz
      az = lx*luse
      tx = d1/ax
      property(1,numentries) = t
      property(2,numentries) = luse
      property(3,numentries) = strain
      property(4,numentries) = tx
101   continue

C     ****read file 2

      read(102,*,end=200) t,d1,d2,d3 
      index(2) = index(2) + 1
      if (mod(index(2),outputfreq) .ne. 1) goto 102
      luse = ly - pistonvelocity*t
      ax = luse*lz
      ay = lx*lz
      az = lx*luse
      tz = d3/az
      property(5,numentries) = -tz
102   continue

C     ****read files 3 and 4

      read(103,*,end=200) t,d1,d2,d3 
      read(104,*,end=200) tr,d1r,d2r,d3r 
      fyave = (d2 - d2r)/2.d0
      pistonwork = pistonwork + 
     +     pistonvelocity*(t-tprev)*(fyave+fyprev)/2.d0
      fyprev = fyave
      tprev = t
      index(3) = index(3) + 1
      if (mod(index(3),outputfreq) .ne. 1) goto 103
      luse = ly - pistonvelocity*t
      ax = luse*lz
      ay = lx*lz
      az = lx*luse
      ty = fyave/ay
      property(6,numentries) = ty   
      property(8,numentries) = pistonwork
103   continue

C     ****read file 5

      read(105,*,end=200) t,d1 
      strainenergy = strainenergy + d1
      index(5) = index(5) + 1
      if (mod(index(5),outputfreq) .ne. 1) goto 105
      property(9,numentries) = strainenergy
105   continue

C     ****read file 6

      read(106,*,end=200) t,d1 
      temp = d1/heatmass
      if (d1 .lt. initialthermal) initialthermal = d1
      index(6) = index(6) + 1
      if (mod(index(6),outputfreq) .ne. 1) goto 106
      property(10,numentries) = d1 - initialthermal
      property(11,numentries) = temp                
106   continue

C     ****read file 7

      read(107,*,end=200) t,d1 
      index(7) = index(7) + 1
      if (mod(index(7),outputfreq) .ne. 1) goto 107
      property(12,numentries) = d1                  
107   continue

      goto 1 
200   continue

C     *****output results
 
      write (201,'(16A12)') (labels(j),j=1,16)
      do i = 1, numentries
        property(7,i) = property(6,i) 
     +         - (property(5,i) + property(4,i))/2.d0
        property(13,i) = property(9,i) + property(10,i) + property(12,i)
        property(14,i) = (property(4,i) + property(5,i) 
     +                     + property(6,i))/3.d0
        property(15,i) = densitymaterial*c0**2*property(3,i)
     +                     * (1.d0 - grun*property(3,i)/2.d0)
     +                     / (1.d0 - S*property(3,i))**2
        property(16,i) = property(15,i) 
     +                     + property(10,i)*grun/(lx*ly*lz)
        write (201,'(16E12.4)') (property(j,i),j=1,16)
      end do
      end

