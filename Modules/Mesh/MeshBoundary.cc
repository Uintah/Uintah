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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Triangles.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

using sci::MeshHandle;
using sci::Element;

class MeshBoundary : public Module {
    MeshIPort* inport;
    GeometryOPort* outport;
public:
    MeshBoundary(const clString& id);
    MeshBoundary(const MeshBoundary&, int deep);
    virtual ~MeshBoundary();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_MeshBoundary(const clString& id)
{
    return scinew MeshBoundary(id);
}
};

MeshBoundary::MeshBoundary(const clString& id)
: Module("MeshBoundary", id, Filter)
{
   // Create the input port
    inport=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inport);
    outport=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(outport);
}

MeshBoundary::MeshBoundary(const MeshBoundary& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("MeshBoundary::MeshBoundary");
}

MeshBoundary::~MeshBoundary()
{
}

Module* MeshBoundary::clone(int deep)
{
    return scinew MeshBoundary(*this, deep);
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
    for(int i=0;i<mesh->elems.size();i++) {
      Element *teste = mesh->elems[i];
      if (teste) {
	for(int j=0;j<4;j++) {
	  if (teste->faces[j] == -1) {
	      tris->add(mesh->nodes[teste->n[facei[j+1]]]->p,
			mesh->nodes[teste->n[facei[j+2]]]->p,
			mesh->nodes[teste->n[facei[j+3]]]->p);
	  }
	}
      }
    }
    outport->addObj(tris,"Boundary Triangles");
}
