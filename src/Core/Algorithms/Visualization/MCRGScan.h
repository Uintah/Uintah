/*
 *  MCRGScan.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef MCRGScan_h
#define MCRGScan_h

#include <stdio.h>


#include <SCICore/Containers/String.h>
#include <SCICore/Thread/Time.h>
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/Datatypes/ScalarField.h> 
#include <SCICore/Datatypes/ScalarFieldRGdouble.h> 
#include <SCICore/Datatypes/ScalarFieldRGfloat.h> 
#include <SCICore/Datatypes/ScalarFieldRGint.h> 
#include <SCICore/Datatypes/ScalarFieldRGshort.h> 
#include <SCICore/Datatypes/ScalarFieldRGushort.h> 
#include <SCICore/Datatypes/ScalarFieldRGchar.h> 
#include <SCICore/Datatypes/ScalarFieldRGuchar.h> 
#include <PSECore/Datatypes/ScalarFieldPort.h>  
#include <SCICore/Thread/Thread.h>

#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>

#include <PSECommon/Algorithms/Visualization/mcube_scan.h>


namespace PSECommon {
  namespace Algorithms {
    
    using namespace SCICore::Datatypes;
    using namespace SCICore::GeomSpace;
    using namespace SCICore::Geometry;

    template <class F>
      class MCRGScan
      {
      private:
	F *field;
	int x_dim, y_dim, z_dim;
	double sx,sy,sz;

	GeomGroup *group;
	GeomTrianglesP *tri;
      public:
	MCRGScan( F *f );
	~MCRGScan();
	
	void extract( int, double);
	void extract( double, int, int, int);
	void extract( double, int, int, int, int, int ,int);
	void reset( int );
	GeomGroup *getGeom() { return group; };
      };
  

    template <class F>
      MCRGScan<F>::MCRGScan( F *field ) : field(field)
      {
	x_dim = field->grid.dim1();
	y_dim = field->grid.dim2();
	z_dim = field->grid.dim3();

	Point min, max;
	field->get_bounds( min, max );
	sx = (max.x() - min.x())/x_dim;
	sy = (max.y() - min.y())/y_dim;
	sz = (max.z() - min.z())/z_dim;
      }
    
    template <class F>
      MCRGScan<F>::~MCRGScan()
      {
      }
    
    
    template <class F>
      void MCRGScan<F>::extract( int cell, double iso )
      {
	int x = cell % x_dim;
	int y = ((cell - x )/ x_dim) % y_dim;
	int z = cell/(x_dim*y_dim) ;

	extract( iso, x, y, z, 1, 1, 1 );
      }

    template <class F>
      void MCRGScan<F>::extract( double iso, int x, int y, int z )
      {
	extract( v, x, y, z, 1, 1, 1 );
      }

    template <class F>
      void MCRGScan<F>::extract( double iso, int x, int y, int z,
			     int dx, int dy, int dz)
      {
	double val[8];
	val[0]=field->grid(x,    y,    z   ) - iso;
	val[1]=field->grid(x+dx, y,    z   ) - iso;
	val[2]=field->grid(x+dx, y+dy, z   ) - iso;
	val[3]=field->grid(x,    y+dy, z   ) - iso;
	val[4]=field->grid(x,    y,    z+dz) - iso;
	val[5]=field->grid(x+dx, y,    z+dz) - iso;
	val[6]=field->grid(x+dx, y+dy, z+dz) - iso;
	val[7]=field->grid(x,    y+dy, z+dz) - iso;

	int mask=0;
	int idx;
	for(idx=0;idx<8;idx++){
	  if(val[idx]<0)
	    mask|=1<<idx;
	}
	if (mask==0 || mask==255) {
	  printf("Cell is %s\n", mask ? "full" : "empry");
	  return;
	}
	
	double x0 = x*sx;
	double x1 = (x+dx)*sx;
	double y0 = y*sy;
	double y1 = (y+dy)*sy;
	double z0 = z*sz;
	double z1 = (z+dz)*sz;

	Point vp[8];
	vp[0]=Point(x0, y0, z0);
	vp[1]=Point(x1, y0, z0);
	vp[2]=Point(x1, y1, z0);
	vp[3]=Point(x0, y1, z0);
	vp[4]=Point(x0, y0, z1);
	vp[5]=Point(x1, y0, z1);
	vp[6]=Point(x1, y1, z1);
	vp[7]=Point(x0, y1, z1);
      
	TriangleCase *tcase=&tri_case[mask];
	int *vertex = tcase->vertex;

	Point q[12];

	// interpolate and project vertices
	int v = 0;
	for (int t=0; t<tcase->n; t++) {
	  int id = vertex[v++];
	  for ( ; id != -1; id=vertex[v++] ) {
	    int v1 = edge_table[id][0];
	    int v2 = edge_table[id][1];
	    q[id] = Interpolate(vp[v1], vp[v2], val[v1]/(val[v1]-val[v2]));
	  }
	}
      
	v = 0;
	for ( int t=0; t<tcase->n; t++) {
	  int v0 = vertex[v++];
	  int v1 = vertex[v++];
	  int v2 = vertex[v++];
	
	  for (; v2 != -1; v1=v2,v2=vertex[v++]) {
	    tri->add(q[v0], q[v1], q[v2]);
	  }
	
	}
      }

    template <class F>
      void MCRGScan<F>::reset( int n )
      {
	group = new GeomGroup;
	tri = new GeomTrianglesP;
	tri->reserve_clear(n*3);
	group->add(tri);
      }

     
  }  // namespace Algorithms
}  // namespace PSECommon

#endif
