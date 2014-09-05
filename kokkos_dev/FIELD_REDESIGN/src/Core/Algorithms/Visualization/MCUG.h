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


#include <SCICore/Containers/String.h>
#include <SCICore/Thread/Time.h>
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/Datatypes/ScalarFieldUG.h> 

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
	
	Array1<NodeHandle> &nodes = field->mesh->nodes;
	Array1<double> &data = field->data;

	int *n = field->mesh->elems[cell]->n;


	Point &n0 = nodes[n[0]]->p;
	Point &n1 = nodes[n[1]]->p;
	Point &n2 = nodes[n[2]]->p;
	Point &n3 = nodes[n[3]]->p;
	
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

     
  }  // namespace Algorithms
}  // namespace PSECommon

#endif
