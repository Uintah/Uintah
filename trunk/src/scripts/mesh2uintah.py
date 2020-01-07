#!/bin/env python
"""
Filename:   mesh2uintah.py

Summary:    Conversion utility to read external mesh file formats and 
            create points files for particle import into Uintah for simulations

Usage:      python mesh2uintah.py <mesh.inp> [-flags]

Input:      Mesh files from Abaqus (*.inp), Tetgen (*.ele, *.node), etc. describing 
            geometry for approximate particle domains
             (GIMP,CPDI - cuboid, hexahedron or CPTI - tetrahedron)
Output:     Uintah *.pts files created for file geometry import of particle domains
            
Revisions:  YYMMDD    Author            Comments
-------------------------------------------------------------------------------------
            150205    Brian Leavy       Abaqus import option added 
            140505    Brian Leavy       created mesh2uintah.py
"""
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os,sys
import re,operator
import warnings
import time

try:
    # Import mpi4py; if it fails, execute serially without it
    from mpi4py import MPI
    hasmpi = True
except ImportError:
    hasmpi = False
    #print "WARNING: mpi4py could not be imported.  Using single rank execution."

# python Material Point Method (pyMPM) and Uintah input file utilities
#def writeUintahPtsFile(xp,r1,r2,r3,prefix):
def writeUintahPtsFile(xp,r1,r2,r3,mats,matnum,filename):
    """
    Write a Uintah (*.PTS) file with centroids and rvectors for each hexahedron or tetrahedron particle domain 
    """
    # Uintah .pts file 

    f = open(filename,'w')
    print '\n*** Writing Uintah points file: ',filename,'***'
    if (args.rescale):
      print '    Rescaled to microns.'

    nump=len(xp)
    dim=len(xp[0])
    #print 'nump=',nump,' dim=',dim
    #print '#  centroid[x,y,z]  r1[x,y,z]  r2[x,y,z]  r3[x,y,z]' 
    for p in range(nump):
     if (mats[p]==matnum): # check material number
      if (dim==2):
        # 2D centroid < x y >  r1 < x y >  r2 < x y > 
        #print p,xp[p],r1[p],r2[p]
        #f.write('%.6e %.6e  %.6e %.6e  %.6e %.6e  %.6e %.6e \n' %(xp[p,0],xp[p,1], 
        f.write('%16.16le %16.16le  %16.16le %16.16le  %16.16le %16.16le  %16.16le %16.16le \n' %(xp[p,0],xp[p,1], 
                                                  r1[p,0],r1[p,1],r2[p,0],r2[p,1])) 
      else:
        # 3D centroid < x y z >  r1 < x y z>  r2 < x y z >  r3 < x y z > 
        #print p,xp[p],r1[p],r2[p],r3[p]
        #f.write('%.6e %.6e %.6e  %.6e %.6e %.6e  %.6e %.6e %.6e  %.6e %.6e %.6e \n' %(xp[p,0],xp[p,1],xp[p,2],r1[p,0],r1[p,1],r1[p,2], 
        f.write('%16.16le %16.16le %16.16le  %16.16le %16.16le %16.16le  %16.16le %16.16le %16.16le  %16.16le %16.16le %16.16le \n' %(xp[p,0],xp[p,1],xp[p,2],r1[p,0],r1[p,1],r1[p,2], 
                                                            r2[p,0],r2[p,1],r2[p,2],r3[p,0],r3[p,1],r3[p,2])) 
    f.close()
    #print '\nRun pfs in uintah to create bounding box in ',filename,'.0 file\n'
    return


def writeNodes(xn,offsetn,prefix):
    """
    Write a TetGen (*.NODE) file with all nodes in mesh
    """
    # TetGen .node file 
    filename=prefix+'.node'
    f = open(filename,'w')
    print '\n*** Writing Node file: ',filename,'***'
    print 'xn=',xn,' offsetn=',offsetn
    n_nodes=int(len(xn))                                                        # number of nodes
    dim = int(len(xn[0]))                                                       # dimensions
    n_attr = int(0)                                                             # number of attributes
    bndry = int(0)                                                              # boundary flag
    print '# Number of nodes, dimension, number of attributes, boundary flag'
    print '%i %i %i %i' %(n_nodes,dim,n_attr,bndry) 
    f.write('# Number of nodes, dimension, number of attributes, boundary flag\n')
    f.write('%i %i %i %i\n' %(n_nodes,dim,n_attr,bndry)) 
    print '# Node#  <x,y(,z)>' 
    f.write('# Node#  <x,y(,z)>\n') 
    # Start at a given node number depending on offset
    i = int(offsetn) 
    for n in range(n_nodes):
      if (dim==2):
        # 2D: node#  < x y > 
        print '   %i     %.6e  %.6e' %(i,xn[n,0],xn[n,1])
        f.write('   %i     %.6e  %6e\n' %(i,xn[n,0],xn[n,1]))
      else:
        # 3D: node#  < x y z >
        print '   %i    %.6e  %.6e  %.6e' %(i,xn[n,0],xn[n,1],xn[n,2])
        f.write('   %i    %.6e  %.6e  %.6e\n' %(i,xn[n,0],xn[n,1],xn[n,2]))
      i += 1
    f.close()
    return



def writeElements(ne,offsete,prefix):
    """
    Write a TetGen (*.ELE) file with all elements in mesh
    """
    # TetGen .ele file 
    filename=prefix+'.ele'
    f = open(filename,'w')
    print '\n*** Writing Element file: ',filename,'***'
    n_elements=int(len(ne))                                                     # number of elements 
    npe = int(len(ne[0]))                                                       # nodes per element 
    bndry = int(0)                                                              # boundary flag
    print '# Number of elements, nodes per element, boundary flag'
    print '%i %i %i' %(n_elements,npe,bndry) 
    f.write('# Number of elements, nodes per element, boundary flag\n')
    f.write('%i %i %i\n' %(n_elements,npe,bndry)) 
    print '# Element#  Node# n0 n1 n2 (n3)' 
    f.write('# Element#  Node# n0 n1 n2 (n3)\n') 
    # Start at a given node number depending on offset
    i = int(offsete) 
    for e in range(n_elements):
      if (npe==3):
        # 2D: element#  node# n0 n1 n2 
        print '    %i       %i     %i     %i' %(i,ne[e,0],ne[e,1],ne[e,2])
        f.write('    %i       %i     %i     %i\n' %(i,ne[e,0],ne[e,1],ne[e,2]))
      else:
        # 3D: element#  node# n0 n1 n2 n3 
        print '    %i       %i     %i     %i     %i' %(i,ne[e,0],ne[e,1],ne[e,2],ne[e,3])
        f.write('    %i       %i     %i     %i     %i\n' %(i,ne[e,0],ne[e,1],ne[e,2],ne[e,3]))
      i += 1
    f.close()
    return



def readTetgenNodes(prefix):
    """
    Read TetGen node file
    """
    # variables 
    n_nodes=0                                                                   # number of nodes
    n_attr=0                                                                    # number of atributes
    bndry=0                                                                     # boundary flag
    dim=0                                                                       # number of dimensions
    goodline=[]
    offsetn=0
    # read file and extract data
    nodename = prefix+'.node'

    print '\n*** Reading TetGen nodes file: ',nodename,'***'

    # Read NODE file
    fn = open(nodename,'r')
    with fn:
      for line in fn:
        line = line.split('#',1)[0]                                             # remove comments and blanks
        line = line.strip()
        if line:
          goodline.append(line.split())                                         # split values

      for i,val in enumerate(goodline):                                         # loop through file
        if (i==0):
          n_nodes=int(val[0])                                                   # number of nodes
          dim = int(val[1])                                                     # dimensions
          n_attr = int(val[2])                                                  # number of attributes
          bndry = int(val[3])                                                   # boundary flag
          #print '# Number of nodes, dimension, number of attributes, boundary flag'
          #print n_nodes,dim,n_attr,bndry
          inode=0
          continue
        
        if (inode<n_nodes):
          if (inode==0):
            # node list
            #print '# Node index, node coordinates'
            offsetn=int(val[0])
            nodes = np.zeros((n_nodes,dim),float)
            nodenum = np.zeros((n_nodes),int)
          
          nodenum=int(val[0])                                                   # node number 
          nx = float(val[1])                                                    # x value
          ny = float(val[2])                                                    # y value
          if (dim==3):
            nz = float(val[3])                                                  # z value
            nodes[inode]=(nx,ny,nz)                                             # node position
          else:
            nodes[inode]=(nx,ny)                                                # node position
          #print nodenum,nodes[inode]
          inode+=1
    
    fn.close()

    return nodes,offsetn 


def readTetgenElements(nodes,offsetn,prefix):
    """
    Read TetGen element file
    """
    # variables 
    n_nodes=len(nodes)
    if (n_nodes==0):
      print "\nError:  Node file missing for corresponding Element file"
      return
    dim=len(nodes[0])
    n_ele=0                                                                     # number of elements
    npe=0                                                                       # number of corners
    goodline=[]
    # Read ELE file
    elemname = prefix+'.ele'
    print '\n*** Reading TetGen element file: ',elemname,'***'
    fe = open(elemname,'r')

    # NEED TO CORRECT FOR ELEMENT NUMBERS STARTING AT 1 VERSUS ZERO
    # read file and extract data
    with fe:
      for line in fe:
        line = line.split('#',1)[0]                         # remove comments and blanks
        line = line.strip()
        if line:
          goodline.append(line.split())                     # split values

      for i,val in enumerate(goodline):                     # loop through file
        if (i==0):
          n_elements=int(val[0])                            # number of elements
          npe = int(val[1])                                 # nodes per element 
          bndry = int(val[2])                               # boundary flag
          #print '# Number of elements, number of nodes per element, boundary flag'
          #print n_elements,npe,bndry
          ielem=0
          continue

        elif (ielem<n_elements):                            # loop over elements
          if (ielem==0):
            #print '# Element index, node index'
            offsete=int(val[0])
            #print 'offsete=',offsete,' offsetn=',offsetn
            xe = np.zeros((n_elements,npe,dim),float)       # element node positions
            ne = np.zeros((n_elements,npe),float)           # element node list

          elemnum=int(val[0])                               # element number 
          for j in range(npe):                              # element nodes
            nn = int(val[j+1])-int(offsetn)                 # corrected node number
            xe[ielem,j] = nodes[nn]                         # append node positions to element positions
            ne[ielem,j] = nn+int(offsetn)                   # append node to element node list
          #print elemnum,xe[ielem]
          ielem+=1

    fe.close()
    return ne,xe,offsete



def readUintahPtsFile(ptsfile):
    """
    Read Uintah *.pts file with particle centroids and Rvectors
    Assumes centroid, r1(x,y,z) r2(x,y,z) r3(x,y,z)
    Optionally rescale if desired
    """
    # variables 
    goodline=[]

    # Read PTS file
    ptsfile=ptsfile+'.pts'
    print '\n*** Reading Uintah Pts file: ',ptsfile,'***'
    fp = open(ptsfile,'r')

    if (args.rescale):                                  # rescale for microns
      scalefactor=1.0e-6
    else:
      scalefactor=1.0

    # read file and extract data
    with fp:
      for line in fp:
        line = line.split('#',1)[0]                     # remove comments and blanks
        line = line.strip()
        if line:
          goodline.append(line.split())                 # split values

      nump=len(goodline)                                # number of particles 
      npp=4                                             # nodes per particle
      dim=3                                             # dimensions 
      bndry = 0                                         # boundary flag
      #print '# Number of particles, number of nodes per particle, boundary flag'
      #print nump,npp
      # caculate centroids and Rvectors for each particle
      xp=np.zeros((nump,dim),float)                     # particle centroids 
      tmpr1=np.zeros((nump,dim),float)                  # particle Rvectors
      tmpr2=np.zeros((nump,dim),float)
      tmpr3=np.zeros((nump,dim),float)
      for p,val in enumerate(goodline):                     # loop through file
        xp[p]=(float(val[0])*scalefactor,float(val[1])*scalefactor,float(val[2])*scalefactor)
        tmpr1[p]=(float(val[3])*scalefactor,float(val[4])*scalefactor,float(val[5])*scalefactor)
        tmpr2[p]=(float(val[6])*scalefactor,float(val[7])*scalefactor,float(val[8])*scalefactor)
        tmpr3[p]=(float(val[9])*scalefactor,float(val[10])*scalefactor,float(val[11])*scalefactor)
        
        #print 'p=',p,' r1=',tmpr1[p],' r2=',tmpr2[p],' r3=',tmpr3[p]

    fp.close()

    # Reorder Rvectors and check centroids
    #r1=np.zeros((nump,dim),float)                  # particle Rvectors
    #r2=np.zeros((nump,dim),float)
    #r3=np.zeros((nump,dim),float)
    r1,r2,r3 = fixRvectors(tmpr1,tmpr2,tmpr3)
 
    return xp,r1,r2,r3 



def writePoly(xp,conn,prefix):
    """
    Write a TetGen polygon (.POLY) file with vertices, faces and additional properties
    """
    # variables 
    n_nodes=len(xp)                                         # number of nodes
    n_facets=len(conn)                                      # number of faces
    dim=len(xp[0])                                          # number of dimensions
    npp=len(conn[0])                                        # number of corners
    n_holes=0                                               # number of holes 
    n_attr=0                                                # number of attributes
    bndry=0                                                 # boundary markers 0=false 1=true
    # TetGen .POLY header
    filename=prefix+'.poly'
    f = open('mesh/'+filename,'w')
    print '*** writing TetGen file ',filename,'***'
    # Comments follow '#' to the end of a line
    f.write('# '+filename+' TetGen input file\n')
    # Part 1 - Node list
    f.write('\n# part 1 - node list\n')
    # if <# of nodes> set to zero, TetGen uses a separate *.node file
    # <# of nodes> <dimension> <# of attributes> <boundary flag>
    f.write('# <num nodes>  <dim>  <num attributes>  <boundary flag>\n')
    f.write('%i  %i  %i  %i\n' %(n_nodes,dim,n_attr,bndry)) 
    i=0 
    for vertex in xp:
      # <point num> <x> <y> <z> [attributes] [boundary flag]
      if (dim==2):
        f.write('%i  %.6e  %.6e\n' %(i,vertex[0],vertex[1])) 
      else:
        f.write('%i  %.6e  %.6e  %.6e\n' %(i,vertex[0],vertex[1],vertex[2])) 
      i+= 1
    # Part 2 - facet list
    f.write('\n# part 2 - facet list\n')
    # <num of facets> <boundary flag>
    f.write('# <# facets>  <boundary flag>\n')
    f.write('%d  %d\n' %(n_facets,bndry))
    # for each facetNum
    # numPolygons numHoles boundaryFlag
    for face in conn:
      # <# of polygons> 
      f.write('1\n')
      # <# corners> <x> <y> <z> [attributes] [boundary flag]
      f.write('%d  %d %d %d\n'%(npp,face[0],face[1],face[2])) 
    # Part 3 - hole list
    f.write('\n# part 3 - hole list\n')
    f.write('# facets  markers\n')
    f.write('%d\n'%(n_holes))
    # <hole #> <x> <y> <z> 
	  #f.write('%d  %.6e  %.6e  %f\n' %(i,hole[0],hole[1],hole[2])) 
    # Part 4 - attribute list (Optional)
    f.write('\n# part 4 - attribute list\n')
    f.write('# region number\n')
    f.write('%d\n'%(n_attr))
    # <# > <x> <y> <z> [attributes] [boundary flag]
	  #f.write('%d  %d %d %d\n' %(npp,face[0]-1,face[1]-1,face[2]-1)) 
    f.close()
    print '\nwrote TetGen file:',filename,'\n'

    return


def calcParticles(xe):
    """
    Calculate centroid and Rvectors for particle domain elements, and fix Rvector order
    INPUT:    xe                     elements nodes 
    OUTPUT:   xp,r1,r2,r3            particle centroids,Rvectors 
    """ 
    nump=len(xe)                                                                  # number of particles 
    npp=len(xe[0])                                                                # nodes per particle
    dim=len(xe[0,0])                                                              # dimensions 
   
    # caculate centroids and Rvectors for each particle
    xp=np.zeros((nump,dim),float)                                                 # particle centroids 
    r1=np.zeros((nump,dim),float)                                                 # particle Rvectors
    r2=np.zeros((nump,dim),float)     
    r3=np.zeros((nump,dim),float)     
    rtmp=np.zeros((nump,dim),float)     
                                                      
    for p in range(nump):                                                         # loop over all particles
      # calculate particle centroids
      centroid=np.zeros((dim),float)
      for q in range(npp):
        centroid += xe[p,q]
      centroid /= float(npp)
      xp[p] = centroid
 
      # calculate Rvectors 
      if (npp==4):                                                                # CPTI (Abaqus C3D4 tetrahedron)
        r1[p]=xe[p,1]-xe[p,0]
        r2[p]=xe[p,2]-xe[p,0]
        r3[p]=xe[p,3]-xe[p,0]

      elif args.exodus:                                                           # CPDI ( Exodus Abaqus C3D8 hexahedron)
        # 0,1,2,3,4,5,6,7 
        #negrative rvectors?
        ##r1[p]=xe[p,5]-xe[p,1]
        ##r2[p]=xe[p,2]-xe[p,1]
        ##r3[p]=xe[p,0]-xe[p,1]
        #r1[p]=xe[p,1]-xe[p,0]
        #r2[p]=xe[p,3]-xe[p,0]
        #r3[p]=xe[p,4]-xe[p,0]
        # surface averages for rvectors
        xmin = (xe[p,0]+xe[p,4]+xe[p,7]+xe[p,3])/4.
        xmax = (xe[p,1]+xe[p,2]+xe[p,6]+xe[p,5])/4.
        ymin = (xe[p,0]+xe[p,1]+xe[p,5]+xe[p,4])/4.
        ymax = (xe[p,3]+xe[p,2]+xe[p,6]+xe[p,7])/4.
        zmin = (xe[p,0]+xe[p,1]+xe[p,2]+xe[p,3])/4.
        zmax = (xe[p,4]+xe[p,5]+xe[p,6]+xe[p,7])/4.
        r1[p]=xmax-xmin
        r2[p]=ymax-ymin
        r3[p]=zmax-zmin

      else:                                                                       # CPDI (Abaqus C3D8 hexahedron)
        # 4,0,3,7,5,1,2,6 for exodus 0,1,2,3,4,5,6,7
        #r1[p]=xe[p,0]-xe[p,4]
        #r2[p]=xe[p,7]-xe[p,4]
        #r3[p]=xe[p,5]-xe[p,4]
        # surface averages for rvectors
        xmin = (xe[p,4]+xe[p,5]+xe[p,6]+xe[p,7])/4.
        xmax = (xe[p,0]+xe[p,3]+xe[p,2]+xe[p,1])/4.
        ymin = (xe[p,4]+xe[p,0]+xe[p,1]+xe[p,5])/4.
        ymax = (xe[p,7]+xe[p,3]+xe[p,2]+xe[p,6])/4.
        zmin = (xe[p,4]+xe[p,0]+xe[p,3]+xe[p,7])/4.
        zmax = (xe[p,5]+xe[p,1]+xe[p,6]+xe[p,2])/4.
        r1[p]=xmax-xmin
        r2[p]=ymax-ymin
        r3[p]=zmax-zmin
      if (args.rescale):                                                          # Rescale for microns if need be
        scalefactor=1.0e-6
        xp[p]*=scalefactor
        r1[p]*=scalefactor
        r2[p]*=scalefactor 
        r3[p]*=scalefactor 

      if (args.stats):                                                            # Update statistics for particle edge lengths
        if (np.linalg.norm(r1[p]) < esize[0]):                                    # Perhaps add other edges not covered by rvectors?
          esize[0]=np.linalg.norm(r1[p])
        if (np.linalg.norm(r2[p]) < esize[0]):
          esize[0]=np.linalg.norm(r2[p])
        if (np.linalg.norm(r3[p]) < esize[0]):
          esize[0]=np.linalg.norm(r3[p])
        if (np.linalg.norm(r1[p]) > esize[1]):
          esize[1]=np.linalg.norm(r1[p])
        if (np.linalg.norm(r2[p]) < esize[1]):
          esize[1]=np.linalg.norm(r2[p])
        if (np.linalg.norm(r3[p]) < esize[1]):
          esize[1]=np.linalg.norm(r3[p])
                                                                                  # Update statistics for bounding box
        # Particle domain bounding box
        pbox = zip(*xe[p])
        if (min(pbox[0]) < bbox[0]):
          bbox[0]=min(pbox[0])
        if (max(pbox[0]) > bbox[3]):
          bbox[3]=max(pbox[0])
        if (min(pbox[1]) < bbox[1]):
          bbox[1]=min(pbox[1])
        if (max(pbox[1]) > bbox[4]):
          bbox[4]=max(pbox[1])
        if (min(pbox[2]) < bbox[2]):
          bbox[2]=min(pbox[2])
        if (max(pbox[2]) > bbox[5]):
          bbox[5]=max(pbox[2])
        # Particle centroid bounding box
        if (xp[p,0] < xpbbox[0]):
          xpbbox[0]=xp[p,0]
        if (xp[p,0] > xpbbox[3]):
          xpbbox[3]=xp[p,1]
        if (xp[p,1] < xpbbox[1]):
          xpbbox[1]=xp[p,1]
        if (xp[p,1] > xpbbox[4]):
          xpbbox[4]=xp[p,1]
        if (xp[p,2] < xpbbox[2]):
          xpbbox[2]=xp[p,2]
        if (xp[p,2] > xpbbox[5]):
          xpbbox[5]=xp[p,2]

      # check for negative Jacobian
      psize=np.zeros((dim,dim),float)                                             # particle Rvectors in columns 
      psize = ((r1[p][0],r2[p][0],r3[p][0]),(r1[p][1],r2[p][1],r3[p][1]),(r1[p][2],r2[p][2],r3[p][2]))
      determinant = np.linalg.det(psize)
      #print 'det =',determinant
      if (determinant < 0):
        rtmp[p]=r3[p]  
        r3[p]=r2[p]  
        r2[p]=rtmp[p]
        psize = ((r1[p][0],r2[p][0],r3[p][0]),(r1[p][1],r2[p][1],r3[p][1]),(r1[p][2],r2[p][2],r3[p][2]))
        determinant = np.linalg.det(psize)
        print '*** Fixed particle number =',p
        print '***                   det =',determinant
        fixed += 1
 


    return xp,r1,r2,r3 

def fixRvectors(tmpr1,tmpr2,tmpr3):
    """
    Check Rvectors for particle domain elements
    INPUT:    r1,r2,r3            original Rvectors 
    OUTPUT:   r1,r2,r3            corrected Rvectors 
    """ 
    nump=len(tmpr1)                                                               # number of particles 
    dim=len(tmpr1[0])                                                             # dimensions 
 
    # caculate centroids and Rvectors for each particle
    r1=np.zeros((nump,dim),float)                                                 # particle Rvectors
    r2=np.zeros((nump,dim),float)     
    r3=np.zeros((nump,dim),float)     
    rtmp=np.zeros((nump,dim),float)     
                                                      
    for p in range(nump):                                                         # loop over all particles
      r1[p]=tmpr1[p]
      r2[p]=tmpr2[p]
      r3[p]=tmpr3[p]
      
      # check for negative Jacobian
      psize=np.zeros((dim,dim),float)                                             # particle Rvectors in columns 
      psize = ((r1[p][0],r2[p][0],r3[p][0]),(r1[p][1],r2[p][1],r3[p][1]),(r1[p][2],r2[p][2],r3[p][2]))
      determinant = np.linalg.det(psize)
  
      if (determinant < 0):
        rtmp[p]=r3[p]  
        r3[p]=r2[p]  
        r2[p]=rtmp[p]
        psize = ((r1[p][0],r2[p][0],r3[p][0]),(r1[p][1],r2[p][1],r3[p][1]),(r1[p][2],r2[p][2],r3[p][2]))
        determinant = np.linalg.det(psize)
        print '*** Fixed particle =',p,determinant

    return r1,r2,r3 



def calcDomainCorners(xp,r1,r2,r3):
    """
    Calculate particle domain corners from centroid and Rvectors
    INPUT:    xp                        particle centroid 
             r1,r2,r3                  current Rvectors 
    OUTPUT:   xpc                       particle corners 
    """ 
    #n_particles=len(xp)                                                          # number of particles
    npp=len(xp[0])                                                                # nodes per particle 
    dim=len(xp[0,0])                                                              # dimensions
    xpc=np.zeros((npp,dim),float)                                                 # particle domain corners
    if npp==4: 
      # calculate the coordinates of 4 corners of the particle domain
      xpc[0]=(xp[0]-(r1[0]+r2[0]+r3[0])/4., xp[1]-(r1[1]+r2[1]+r3[1])/4., xp[2]-(r1[2]+r2[2]+r3[2])/4.)
      xpc[1]=(xpc[0,0]+r1[0], xpc[0,1]+r1[1], xpc[0,2]+r1[2])
      xpc[2]=(xpc[0,0]+r2[0], xpc[0,1]+r2[1], xpc[0,2]+r2[2])
      xpc[3]=(xpc[0,0]+r3[0], xpc[0,1]+r3[1], xpc[0,2]+r3[2])
    else:
      # calculate the coordinates of 8 corners of the particle domain from centroid
      xpc[0]=(xp[0]-r1[0]/2.-r2[0]/2.-r3[0]/2., xp[1]-r1[1]/2.-r2[1]/2.-r3[1]/2., xp[2]-r1[2]/2.-r2[2]/2.-r3[2]/2.)
      xpc[1]=(xp[0]+r1[0]/2.-r2[0]/2.-r3[0]/2., xp[1]+r1[1]/2.-r2[1]/2.-r3[1]/2., xp[2]+r1[2]/2.-r2[2]/2.-r3[2]/2.)
      xpc[2]=(xp[0]+r1[0]/2.+r2[0]/2.-r3[0]/2., xp[1]+r1[1]/2.+r2[1]/2.-r3[1]/2., xp[2]+r1[2]/2.+r2[2]/2.-r3[2]/2.)
      xpc[3]=(xp[0]-r1[0]/2.+r2[0]/2.-r3[0]/2., xp[1]-r1[1]/2.+r2[1]/2.-r3[1]/2., xp[2]-r1[2]/2.+r2[2]/2.-r3[2]/2.)
      xpc[4]=(xp[0]-r1[0]/2.-r2[0]/2.+r3[0]/2., xp[1]-r1[1]/2.-r2[1]/2.+r3[1]/2., xp[2]-r1[2]/2.-r2[2]/2.+r3[2]/2.)
      xpc[5]=(xp[0]+r1[0]/2.-r2[0]/2.+r3[0]/2., xp[1]+r1[1]/2.-r2[1]/2.+r3[1]/2., xp[2]+r1[2]/2.-r2[2]/2.+r3[2]/2.)
      xpc[6]=(xp[0]+r1[0]/2.+r2[0]/2.+r3[0]/2., xp[1]+r1[1]/2.+r2[1]/2.+r3[1]/2., xp[2]+r1[2]/2.+r2[2]/2.+r3[2]/2.)
      xpc[7]=(xp[0]-r1[0]/2.+r2[0]/2.+r3[0]/2., xp[1]-r1[1]/2.+r2[1]/2.+r3[1]/2., xp[2]-r1[2]/2.+r2[2]/2.+r3[2]/2.)
 
    return xpc


def getElementCorners(ne,nodes,offsetn):
    """
    Get element corners from elements, nodes and offsets 
    INPUT:    ne                        element and correpsonding nodes 
              nodes                     node list 
              offsetn                   node number offset
    OUTPUT:   xe                        element corners 
    """ 
    n_elements=len(ne)                                                            # number of elements
    npe=len(ne[0])                                                                # nodes per element 
    dim=len(nodes[0])                                                             # dimensions
    xe = np.zeros((n_elements,npe,dim),float)                                     # element node positions
    print ne,nodes,offsetn
    # look up the corner positions of the element domain
    for e in range(n_elements):
      nn = ne[e] - int(offsetn)                                                   # corrected node numbers
      for n in range(npe):                                                        # loop through nodes
        subnode = int(nn[n])                                                      
        xe[e,n]=(nodes[subnode])                                                  # get node position
        #print ' e=',e,' n=',n,' subnode=',subnode,' node=',nodes[subnode]
    return xe


def readTetgenFile(inpfile):
    """
    Read Tetgen *.node *.ele files with nodes, elements and boundary conditions
    """
    # Old mesh reading needs to be updated
    # read nodes
    nodes=[]  
    offsetn=0 # offset for node index
    nodes,offsetn = readTetgenNodes(inpfile)
    # write nodes
    #writeNodes(NODES,OFFSETN,'junk')
    # read elements
    ne=[]                                                                         # Element node list
    xe=[]                                                                         # Element node positions
    xp=[]                                                                         # Element node positions
    offsete=0                                                                     # Element number offset
    ne,xe,offsete = readTetgenElements(nodes,offsetn,inpfile)
    # write elements
    #writeElements(NE,OFFSETE,'junk')
    # read holes
    # read attributes 
    # calculate element corners without centroid or Rvector information
    #XE2 = calcElementDomainCorners(NE,NODES,OFFSETN)
    # TetGen
    xp,r1,r2,r3 = calcParticles(xe)
    return xp,r1,r2,r3 
 

def readAbaqusFile(inpfile):
    """
    Read Abaqus *.inp file with nodes, elements, materials and boundary conditions
    """
    # variables 
    nodeline=[]
    elemline=[]
    matnames=[]
    mats=[]
    nodes=False
    elements=False
    m = 0 
    npp = 0
    nump = 0
    # Read Abaqus file (expects sections in order?)
    inpfile = inpfile+".inp"
    print '\n*** Reading Abaqus mesh file: \n  ',inpfile
    fp = open(inpfile,'r')
    # read file and extract data
    with fp:
      for line in fp:

        if line.upper().startswith('*NODE'):
          #print line
          if ',' in line:
            nodetype=line.split(',',1)[1]
            #print nodetype 
          elements=False 
          nodes=True

        elif line.upper().startswith('*ELEM'):
          #print line
          nodes=False
          elements=True
          elemtype=line.split(',',1)[1]
	        # Cubit sculpt meshing creates element sets with EB prefix
          #if mat.upper().startswith('EB'):
          #  mat=mat[2:-1]
          # DREAM3D creates element sets with GRAIN prefix
          #if mat.upper().startswith('GRAIN'):
          #  mat=mat[5:-1]
          #  mat=mat.split('_')[0]
          mat=line.split(',')[2].split('=')[1]
          matnum=''.join(x for x in mat if x.isdigit())
          matnames.append(matnum)
          #print 'mat=',mat,' matnum=',matnum
          m += 1
          elemtype=elemtype.split(',')[0]
          elemtype=elemtype.split('=')[1]
          if elemtype.upper().startswith('C3D4'):         # Abaqus Tetrahedral element type (CPTI)
            interpolator = 'cpti'
            npp=4                                         # nodes per particle
          else:                                           # Abaqus Hexahedral element type C3D8 (CPDI)
            interpolator = 'cpdi'
            npp=8                                         # nodes per particle
          #print elemtype,npp
        elif line.upper().startswith('*NSET'):
          #print line
          elements=False
          nodes=False
        #elif line.upper().startswith('*ELSET'):
        elif line.upper().startswith('*SOLID SECTION, ELSET') or line.upper().startswith('*ELSET'):
	        #check number of materials with nmats += 1
          elements=False
          nodes=False
        elif line.upper().startswith('*SURF'):
          #print line
          elements=False
          nodes=False
          #elements=False
        #elif line.upper().startswith('*PART'):
        #  elements=False
        #  nodes=False
        #elif line.upper().startswith('**'):  # ending group characters
        #  elements=False
        #  nodes=False
        #else:
        #  print "nothing"

        line = line.split('*',1)[0]                       # remove headers and blanks
        line = line.strip()                               # remove extra whitespace

        if line:
          if nodes:
            line = line.split(",")                        # remove all abaqus commas in lists
            #print '   node=',line
            nodeline.append(line)                 
          elif elements:                                
            line = line.split(",")                        # remove all abaqus commas in lists
            #print '   elem=',line 
            elemline.append(line)                 
            mats.append(matnum)
          #else:                                
          #  print line 

    fp.close()
    #print "elemline=",elemline
    #print "nodeline=",nodeline
    nump=len(elemline)                                  # number of particles 
    numn=len(nodeline)                                  # number of nodes
    numm=len(matnames)                                  # number of materials
    dim=3                                               # dimensions 
    bndry = 0                                           # boundary flag
    print '# materials, particles, number of nodes per particle, nodes'
    print numm,nump,npp,numn

    # create separate files for materials
        
    #for i,el in enumerate(elemline):                   # loop through material element list
    #	print 'i=',i,' m=',mats[i],' elem=',el

    # determine nodes 
    node=np.zeros((numn,dim),float)                     # node or particle corners 
    nodeid=[]                                           # abaqus nodeids (for unique node lists for multiple materials)
    i=0
    for n,val in enumerate(nodeline):                   # loop through node list
      j=0
      nodeid.append(val[j])
      for j in range(dim):                              # loop over dimensions
        node[n,j] = float(val[j+1])                     # node locations 

    # determine elements
    elem=np.zeros((nump,npp),float)                     # element or particle domains 
    xe = np.zeros((nump,npp,dim),float)                 # element node positions
    #mxe = np.zeros((numm,nump,npp,dim),float)                   # material specific element node positions
    i=0
    for p,val in enumerate(elemline):                   # loop through element list
      m = int(mats[p])
      j=0
      for j in range(npp):                              # loop over corners 
        nid = val[j+1].strip()                          # abaqus node id
        nn = nodeid.index(nid)                          # find node number from abaqus node id
        #print 'nid,nn=',nid,nn
        elem[p,j] = nn                                  # element node index list
        xe[p,j] = node[nn]                              # element corners
      #print 'm=',m,' p=',p,' xe=',xe[p]
      #mxe[m-1] = xe[p]                                  # element corners
       
    # caculate centroids and corrected Rvectors for each particle
    #xp=np.zeros((nump,dim),float)                      # particle centroids 
    #r1=np.zeros((nump,dim),float)                      # particle Rvectors
    #r2=np.zeros((nump,dim),float)
    #r3=np.zeros((nump,dim),float)
    xp,r1,r2,r3 = calcParticles(xe)

    return xp,r1,r2,r3,mats,matnames




# MAIN Subroutine 
if __name__ == "__main__":
  str_header=  '''
  filename:   mesh2uintah.py
  
  summary:    Conversion utility to read mesh input files and create 
              Uintah points files for simulations 

  '''
  str_footer='''
  comments:
  '''

  # warnings
  warnings.simplefilter('once', UserWarning)

  # timing
  time_start = time.time()

  # MPI
  if hasmpi:
    mpicomm = MPI.COMM_WORLD
  else:
    mpicomm = None

  # argument parsing
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=str_header,
    prog='python mesh2uintah.py',
    epilog=str_footer)
  parser.add_argument(dest='inpfile',metavar='<mesh>',action='store',default=True,
    help='mesh input file prefix without extension.')
  parser.add_argument('-a','--abaqus',action='store_true',default=True,
    help='read Abaqus mesh files (*.inp) [default].')
  parser.add_argument('-e','--exodus',action='store_true',default=False,
    help='read Abaqus mesh files in ExodusII node ordering (*.inp).')
  parser.add_argument('-j','--nprocs',action='store',default=1,
    help='run with multiple processors.')
  parser.add_argument('-p','--pts',action='store_true',
    help='check Uintah points files (*.pts).')
  parser.add_argument('-r','--rescale',action='store_true',
    help='rescale Uintah points files (*.pts).')
  parser.add_argument('-s','--stats',action='store_true',
    help='Calculate statistics for bounding box and particle domain sizes.')
  parser.add_argument("-t", "--tetgen", action="store_true",
    help='read TetGen or Distmesh mesh files (*.node, *.ele *.poly *.smesh).')
  args = parser.parse_args()

  INPFILE=os.path.realpath(args.inpfile)
  
  # Clear variables 
  XP=[]                                                   # Particle centroid
  R1=[]                                                   # Particle domain Rvectors
  R2=[]
  R3=[]
  MATS=[]
  # Count of elements with reordered r-vectors
  fixed = 0
  # Bounding box for particle domains
  bbox=[1.e30, 1.e30, 1.e30,-1.e30,-1.e30,-1.e30]
  # Bounding box for particle centroids
  xpbbox=[1.e30, 1.e30, 1.e30,-1.e30,-1.e30,-1.e30]
  # Min, Max, Ave edge length of particle domains
  esize=[1.e30, -1.e30, 0.0]

  # read TetGen *.poly or pyDistMesh *.node and *.ele files with given prefix
  if (args.tetgen): 
    XP,R1,R2,R3 = readTetgenFile(INPFILE)
  elif (args.pts):
    # read Uintah *.pts file
    XP,R1,R2,R3 = readUintahPtsFile(INPFILE)
  else: 
    #if (args.abaqus): 
    # read Abaqus *.inp file
    XP,R1,R2,R3,MATS,MATNAMES = readAbaqusFile(INPFILE)
    
  # write Uintah *.pts file
  #writeUintahPtsFile(XP,R1,R2,R3,MATS,INPFILE)
  nmats=len(MATNAMES)
  for m in range(nmats):
    M = MATNAMES[m]
    if (nmats > 1):
      OUTFILE=INPFILE+'_'+M+'.pts'
    else:
      OUTFILE=INPFILE+'.pts'
    writeUintahPtsFile(XP,R1,R2,R3,MATS,M,OUTFILE)
    print '    Uintah file: '+OUTFILE+' written.'
  if (args.stats):
    print '*** Statistics ***'
    print 'Bounding Box: [xmin, ymin, zmin, xmax, ymax, zmax]'
    print ' Particle domains:',bbox
    print ' Particle centroids:',xpbbox
    esize[2]=np.average([esize[0],esize[1]])
    print 'Edge Lengths: [min, max, ave]\n',esize
  if (fixed > 0):
    print 'Number of fixed particles:',fixed

