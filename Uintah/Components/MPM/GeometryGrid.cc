#include "GeometryGrid.h"
#include <string>
using std::string;
#include <fstream>
using std::ofstream;
#include <iostream>
using std::endl;

GeometryGrid::GeometryGrid(){
  
}
GeometryGrid::~GeometryGrid(){

 
}

void GeometryGrid::buildGeometryGrid(string posname, string celname,
				     const double grd[7], const double dx[4])
{
  int kz,jy, ix;
  int cnr[9];
  int ncelx,ncely,ncelz;  /* # of cells in each dir.      */
  ofstream gridcel;
  ofstream gridpos;

  gridcel.open(celname.c_str());
  gridpos.open(posname.c_str());
  
  ncelx = (int)((grd[2]-grd[1])/dx[1]+.000001);
  ncely = (int)((grd[4]-grd[3])/dx[2]+.000001);
  ncelz = (int)((grd[6]-grd[5])/dx[3]+.000001);

  gridcel << ncelx << " " << ncely << " " << ncelz << endl;
  
  for(kz=0;kz<ncelz;kz++){
    for(jy=0;jy<ncely;jy++){
      for(ix=1;ix<=ncelx;ix++){
	cnr[1] = ix + jy*(ncelx +1) + kz*(ncelx +1)*(ncely +1);
	cnr[2] = cnr[1] + 1;
	cnr[3] = cnr[1] + (ncelx +1);
	cnr[4] = cnr[3] + 1;
	cnr[5] = cnr[1] + (ncelx +1)*(ncely +1);
	cnr[6] = cnr[2] + (ncelx +1)*(ncely +1);
	cnr[7] = cnr[3] + (ncelx +1)*(ncely +1);
	cnr[8] = cnr[4] + (ncelx +1)*(ncely +1);
	gridcel << cnr[1]<< " " << cnr[2]<< " " << cnr[4]<< " " << cnr[3]<< " " << cnr[5]<< " " << cnr[6]<< " " << cnr[8]<< " " << cnr[7] << endl;;
      }
    }
  }
  
  double xv,yv,zv;

  gridpos << (ncelx+1)*(ncely+1)*(ncelz+1) << endl;
  for(kz=0;kz<=ncelz;kz++){
    for(jy=0;jy<=ncely;jy++){
      for(ix=0;ix<=ncelx;ix++) {
	xv = ((double) ix) * dx[1] + grd[1];
	yv = ((double) jy) * dx[2] + grd[3];
	zv = ((double) kz) * dx[3] + grd[5];
	gridpos << xv << " " << yv << " " << zv << endl;
      }
    }
  }


  return;
}

// $Log$
// Revision 1.1  2000/02/24 06:11:55  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:50  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:40  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.5  1999/01/26 21:53:33  campbell
// Added logging capabilities
//
