/*
 *  MCUG.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef MCUG_h
#define MCUG_h

#include <stdio.h>


#include <Core/Containers/String.h>
#include <Core/Thread/Time.h>
#include <Dataflow/Network/Module.h> 
#include <Core/Datatypes/ScalarFieldUG.h> 

#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/Pt.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>

#include <Packages/Yarden/Core/Algorithms/Visualization/mcube_scan.h>


namespace Yarden {
    
using namespace SCIRun;

class MCUG 
{
private:
  ScalarFieldUG *field;
	
  GeomGroup *group;

public:
  MCUG( ScalarFieldUG *f );
  virtual ~MCUG();
	
  void extract( int, double);
  void reset( int );
  GeomGroup *getGeom() { return group; };
};
  
    
inline
MCUG::MCUG( ScalarFieldUG *field ) : field(field)
{
}

inline
MCUG::~MCUG()
{
}
    

inline
void MCUG::extract( int cell, double v )
{
  GeomTrianglesP *tmp = scinew GeomTrianglesP;
	
  Array1<double> &data = field->data;

  int *n = field->mesh->elems[cell]->n;

  const Point &n0 = field->mesh->point(n[0]);
  const Point &n1 = field->mesh->point(n[1]);
  const Point &n2 = field->mesh->point(n[2]);
  const Point &n3 = field->mesh->point(n[3]);
	
  double v0 = data[n[0]];
  double v1 = data[n[1]];
  double v2 = data[n[2]];
  double v3 = data[n[3]];
	
  if ( v < v1 ) {
    /* one triangle */
    Point p1(Interpolate( n0, n3, (v-v0)/(v3-v0)));
    Point p2(Interpolate( n0, n2, (v-v0)/(v2-v0)));
    Point p3(Interpolate( n0, n1, (v-v0)/(v1-v0)));
	  
    tmp->add( p1, p2, p3 );
  }
  else if ( v < v2 ) {
    /* two triangle */
    Point p1(Interpolate( n0, n3, (v-v0)/(v3-v0)));
    Point p2(Interpolate( n0, n2, (v-v0)/(v2-v0)));
    Point p3(Interpolate( n1, n3, (v-v1)/(v3-v1)));
	  
    tmp->add( p1, p2, p3 );
	  
    Point p4(Interpolate( n1, n2, (v-v1)/(v2-v1)));
    tmp->add( p2, p3, p4 );
	  
  }
  else {
    /* one triangle */
	  
    Point p1 = Interpolate( n0, n3, (v-v0)/(v3-v0));
    Point p2 = Interpolate( n1, n3, (v-v1)/(v3-v1));
    Point p3 = Interpolate( n2, n3, (v-v2)/(v3-v2));
	  
    tmp->add( p1, p2, p3 );
  }
	
  group->add(tmp );
}


inline
void MCUG::reset( int n )
{
  group = new GeomGroup;
}

     
} // End namespace Yarden

#endif
