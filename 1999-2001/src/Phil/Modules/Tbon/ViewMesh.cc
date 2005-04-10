//static char *id="@(#) $Id$";

/* ViewMesh.cc
   Display grid points for unstructured meshes

   Philip Sutton
   July 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include <SCICore/Util/NotFinished.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>

#include <strings.h>
#include <iostream>

namespace Phil {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace std;

struct UGpoint {
  float x, y, z;
};
struct UGtetra {
   int v[4];
};


class ViewMesh : public Module {
public:
  // Functions required by Module inheritance
  ViewMesh(const clString& id);
  virtual ~ViewMesh();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

protected:
private:
  TCLint representation;
  TCLdouble radius;
  TCLstring geomfilename;

  int npts, ntets;
  UGpoint* points;
  UGtetra* tets;

  int* matrix;

  GeometryOPort* geomout;
  
  void update();
}; // class ViewMesh

// More required stuff...
extern "C" Module* make_ViewMesh(const clString& id){
  return new ViewMesh(id);
}

// Constructor
ViewMesh::ViewMesh(const clString& id)
  : Module("ViewMesh", id, Filter), representation("representation",id,this), 
  radius("radius",id,this), geomfilename("geomfilename",id,this)
{
  representation.set(0);
  radius.set(0.0);
  matrix = 0;

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
ViewMesh::~ViewMesh() {
}

// Execute - main guts
void
ViewMesh::execute() {
  int i;
  
  geomout->delAll();
  FILE* geom = fopen( geomfilename.get()(), "r" );
  if( !geom ) {
    cerr << "Error opening .htvol file " << geomfilename.get()() << endl;
    return;
  }

  // read header
  char dummy1[80];
  char dummy2[80];
  fscanf( geom, "%s %s\n", dummy1, dummy2 );
  if( strcmp( dummy1, "HTVolumeBrick" ) || strcmp( dummy2, "file" ) ) {
    cerr << "Error: file " << geomfilename.get()() 
	 << " is not an HTVolumeBrick file" << endl;
    return;
  }
  fscanf( geom, "%d %d\n", &npts, &ntets );
  cout << "npts = " << npts << " ntets = " << ntets << endl;

  // read points
  cout << "reading points" << endl;
  points = new UGpoint[npts];
  for( i = 0; i < npts; i++ ) {
    fread( &points[i].x, sizeof(float), 1, geom );
    fread( &points[i].y, sizeof(float), 1, geom );
    fread( &points[i].z, sizeof(float), 1, geom );
    fseek( geom, sizeof(float), SEEK_CUR );
  }
  
  // read tets
  cout << "reading tets" << endl;
  tets = new UGtetra[ntets];
  for( i = 0; i < ntets; i++ ) {
    fread( tets[i].v, sizeof(int), 4, geom );
  }

  // fill in connectivity matrix
  cout << "making lines" << endl;
  GeomLines* lines = new GeomLines();
  for( i = 0; i < ntets; i++ ) {
    int p1, p2;
    p1 = tets[i].v[0]; p2 = tets[i].v[1];
    lines->add( Point( points[p1].x, points[p1].y, points[p1].z ),
		Point( points[p2].x, points[p2].y, points[p2].z ) );
    lines->add( Point( points[p2].x, points[p2].y, points[p2].z ),
		Point( points[p1].x, points[p1].y, points[p1].z ) );

    p1 = tets[i].v[0]; p2 = tets[i].v[2];
    lines->add( Point( points[p1].x, points[p1].y, points[p1].z ),
		Point( points[p2].x, points[p2].y, points[p2].z ) );
    lines->add( Point( points[p2].x, points[p2].y, points[p2].z ),
		Point( points[p1].x, points[p1].y, points[p1].z ) );

    p1 = tets[i].v[0]; p2 = tets[i].v[3];
    lines->add( Point( points[p1].x, points[p1].y, points[p1].z ),
		Point( points[p2].x, points[p2].y, points[p2].z ) );
    lines->add( Point( points[p2].x, points[p2].y, points[p2].z ),
		Point( points[p1].x, points[p1].y, points[p1].z ) );

    p1 = tets[i].v[1]; p2 = tets[i].v[2];
    lines->add( Point( points[p1].x, points[p1].y, points[p1].z ),
		Point( points[p2].x, points[p2].y, points[p2].z ) );
    lines->add( Point( points[p2].x, points[p2].y, points[p2].z ),
		Point( points[p1].x, points[p1].y, points[p1].z ) );

    p1 = tets[i].v[1]; p2 = tets[i].v[3];
    lines->add( Point( points[p1].x, points[p1].y, points[p1].z ),
		Point( points[p2].x, points[p2].y, points[p2].z ) );
    lines->add( Point( points[p2].x, points[p2].y, points[p2].z ),
		Point( points[p1].x, points[p1].y, points[p1].z ) );

    p1 = tets[i].v[2]; p2 = tets[i].v[3];
    lines->add( Point( points[p1].x, points[p1].y, points[p1].z ),
		Point( points[p2].x, points[p2].y, points[p2].z ) );
    lines->add( Point( points[p2].x, points[p2].y, points[p2].z ),
		Point( points[p1].x, points[p1].y, points[p1].z ) );

  }
  GeomMaterial* matl = new GeomMaterial( lines, 
					 new Material( Color(0,0,0),
						       Color(0,0.2,0.5),
						       Color(1,1,1),
						       20.0 ) 
					 );
  geomout->addObj( matl, "Mesh" );
  geomout->flush();


  fclose(geom);

} // execute

void
ViewMesh::tcl_command(TCLArgs& args, void* userdata) {
  if( args[1] == "update" ) {
    if( matrix != 0 )
      update();
  } else {
    Module::tcl_command( args, userdata );
  }
}

void
ViewMesh::update() {
  int i, j;
  reset_vars();

  cout << "drawing tets" << endl;
  if( representation.get() == 0 ) {
    // lines
    GeomLines* lines = new GeomLines();
    for( i = 0; i < npts; i++ ) {
      for( j = i; j < npts; j++ ) {
	if( matrix[i*npts + j] ) {
	  lines->add( Point( points[i].x, points[i].y, points[i].z ),
		      Point( points[j].x, points[j].y, points[j].z ) );
	  matrix[i*npts+j] = 0;
	}
      }
    }
    GeomMaterial* matl = new GeomMaterial( lines, 
					   new Material( Color(0,0,0),
							 Color(0,0.2,0.5),
							 Color(1,1,1),
							 20.0 ) 
					   );
    geomout->addObj( matl, "Mesh" );

  } else {
    // cylinders
    GeomGroup* group = new GeomGroup();
    for( i = 0; i < npts; i++ ) {
      for( j = i; j < npts; j++ ) {
	//	if( matrix[i][j] > 0 ) {
	if( matrix[i*npts + j] ) {
	  group->add( new GeomCylinder( Point( points[i].x, 
					       points[i].y, points[i].z ),
					Point( points[j].x, points[j].y, 
					       points[j].z ),
					radius.get() ) );
	  matrix[i*npts+j] = 0;
	}
      }
    }
    geomout->addObj( group, "Mesh" );
  }
  cout << "done" << endl;
  geomout->flush();
}


} // End namespace Modules
} // End namespace Phil

//
// $Log$
// Revision 1.3  2000/03/17 09:28:12  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  2000/02/04 22:16:48  psutton
// fixed ID problem
//
// Revision 1.1  2000/02/04 21:16:47  psutton
// initial revision
//
//
