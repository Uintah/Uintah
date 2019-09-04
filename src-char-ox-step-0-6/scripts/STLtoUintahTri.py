'''
Created on 13/10/2017

Tested on vtk version 6.1.0

Contributed by Jagir Hussan, U. of Auckland

'''

import numpy as np
import vtk  
import argparse


class STL2UintahTri(object):
    '''
    Convert a STL file into a format that could be read into to uintah tri object
    '''


    def __init__(self, stlfilename):
        '''
        :param stlfilename - Location of stl file that needs to be converted
        Load the stl file
        '''
        self.sourceMesh = stlfilename
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stlfilename)
        reader.Update()
        self.newSurf = reader.GetOutput()
        self.offset = np.zeros(3)
    def getBounds(self):
        '''
        Get the bounds of the surface mesh
        '''
        return np.array(self.newSurf.GetBounds()).reshape((3,2)).T
    
    def setOffset(self,offset):
        '''
        :param offset - a 1x3 array
        Offset the vertices by offset prior to writing the mesh to pts
        '''
        self.offset = np.array(offset)
        
    def outputTriFiles(self,filenameprefix,scaleVerticesBy=1.0):
        '''
        :param filenameprefix - Location and prefix of the files to store the vertex and face information
        :param scaleVerticesBy  - Optional parameter to scale the vertices of the mesh
        Output the vertex and face information of the triangles in the stl mesh to 
        filenameprefix.pts and filenameprefix.tri files
        '''
        numberOfPoints = self.newSurf.GetNumberOfPoints()
        numberOfCells = self.newSurf.GetNumberOfCells()
        with open('%s.pts' % filenameprefix,'w') as ser:
            for nid in range(numberOfPoints):
                pt = (np.array(self.newSurf.GetPoint(nid))+self.offset)*scaleVerticesBy
                print >>ser,pt[0],' ',pt[1],' ',pt[2]
        
        with open('%s.tri' % filenameprefix,'w') as ser:
            for eid in range(numberOfCells):
                cell = self.newSurf.GetCell(eid)
                points = cell.GetPointIds()
                ctype = cell.GetCellType()
                
                if ctype == vtk.VTK_TRIANGLE:
                    p1 = points.GetId(0)
                    p2 = points.GetId(1)
                    p3 = points.GetId(2)
                    print >>ser, p1,' ',p2,' ',p3
        print 'Number of vertices : ',numberOfPoints
        print 'Number of Faces    : ',numberOfCells
        print "Bounds ",(self.getBounds()+self.offset)*scaleVerticesBy
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--stlfile', help='location of target stl file',required=True)
    parser.add_argument('-u','--uintah', help='prefix for uintah pts and tri files',required=True)
    parser.add_argument('-z','--zerolower', help='Set the lower bounds to be 0,0,0',action='store_true')
    parser.add_argument('-s','--scale',help='Scale the vertices by this factor',type=float)
    args = vars(parser.parse_args())    

    task = STL2UintahTri(args['stlfile'])
    if args['zerolower']:
        bounds = task.getBounds()
        print 'Vertex Bounds prior to resetting the lower bounds to 0,0,0 :',bounds
        offset = np.array(-bounds[0])
        task.setOffset(offset)
    scale = 1.0
    if not args['scale'] is None:
        scale = float(args['scale'])
    task.outputTriFiles(args['uintah'],scale)
    print 'Completed'
        
