/*
 *  MeshBoundary.cc:  Unfinished modules
 *
 *  Written by:
 *   Peter-Pike Sloan and David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::GeomSpace;

class MeshBoundary : public Module {
    MeshIPort* inport;
    GeometryOPort* outport;
    SurfaceOPort* osurf;
public:
    MeshBoundary(const clString& id);
    virtual ~MeshBoundary();
    virtual void execute();
};

Module* make_MeshBoundary(const clString& id)
{
    return scinew MeshBoundary(id);
}

MeshBoundary::MeshBoundary(const clString& id)
: Module("MeshBoundary", id, Filter)
{
   // Create the input port
    inport=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);
    outport=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(outport);
    osurf=scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurf);
}

MeshBoundary::~MeshBoundary()
{
}

void MeshBoundary::execute()
{
    MeshHandle mesh;
    if (!inport->get(mesh))
	return;


    // now actualy look at this stuff...

    GeomTrianglesP *tris= scinew GeomTrianglesP;
    mesh->compute_face_neighbors();

    int facei[8] = {0,1,2,3,0,1,2,3};
    Array1<int> bdryNodes(mesh->nodes.size());
    bdryNodes.initialize(0);

    int i;
    for(i=0;i<mesh->elems.size();i++) {
      Element *teste = mesh->elems[i];
      if (teste) {
	for(int j=0;j<4;j++) {
	  if (teste->faces[j] == -1) {
	      bdryNodes[teste->n[facei[j+1]]] = 1;
	      bdryNodes[teste->n[facei[j+2]]] = 1;
	      bdryNodes[teste->n[facei[j+3]]] = 1;
	      tris->add(mesh->nodes[teste->n[facei[j+1]]]->p,
			mesh->nodes[teste->n[facei[j+2]]]->p,
			mesh->nodes[teste->n[facei[j+3]]]->p);
	  }
	}
      }
    }
    TriSurface *ts = new TriSurface;
    Array1<int> nodeMap(bdryNodes.size());
    nodeMap.initialize(-1);
    int count=0;
    for (i=0; i<mesh->nodes.size(); i++) {
	if (bdryNodes[i]) {
	    nodeMap[i] = count;
	    count++;
	    ts->points.add(mesh->nodes[i]->p);
	}
    }
    for (i=0; i<mesh->elems.size(); i++) {
	Element *teste = mesh->elems[i];
	if (teste) {
	    for(int j=0;j<4;j++) {
		if (teste->faces[j] == -1) {
		    int n1=nodeMap[teste->n[facei[j+1]]];
		    int n2=nodeMap[teste->n[facei[j+2]]];
		    int n3=nodeMap[teste->n[facei[j+3]]];
		    if (n1 == -1 || n2 == -1 || n3 == -1) {
			cerr << "ERROR in MeshBoundary!\n";
		    }
		    ts->elements.add(new TSElement(n1,n2,n3));
		}
	    }
	}
    }
    outport->addObj(tris,"Boundary Triangles");
    SurfaceHandle tsHandle(ts);
    osurf->send(tsHandle);
}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.2  1999/10/07 02:08:19  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/05 01:15:27  dmw
// added all of the old SCIRun mesh modules
//
