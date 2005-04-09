/*
 *  Delaunay.cc:  Delaunay Triangulation in 3D
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

class Delaunay : public Module {
    MeshIPort* iport;
    MeshOPort* oport;
public:
    Delaunay(const clString& id);
    Delaunay(const Delaunay&, int deep);
    virtual ~Delaunay();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_Delaunay(const clString& id)
{
    return new Delaunay(id);
}

static RegisterModule db1("Mesh", "Delaunay", make_Delaunay);

Delaunay::Delaunay(const clString& id)
: Module("Delaunay", id, Filter)
{
    iport=new MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new MeshOPort(this, "Delaunay Mesh", MeshIPort::Atomic);
    add_oport(oport);
}

Delaunay::Delaunay(const Delaunay& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Delaunay::Delaunay");
}

Delaunay::~Delaunay()
{
}

Module* Delaunay::clone(int deep)
{
    return new Delaunay(*this, deep);
}

void Delaunay::execute()
{
    MeshHandle mesh_handle;
    if(!iport->get(mesh_handle))
	return;

    // Get our own copy of the mesh...
    mesh_handle.detach();
    Mesh* mesh=mesh_handle.get_rep();
    mesh->elems.remove_all();

    int nnodes=mesh->nodes.size();
    BBox bbox;
    for(int i=0;i<nnodes;i++)
	bbox.extend(mesh->nodes[i]->p);

    double epsilon=1.e-4;

    // Extend by max-(eps, eps, eps) and min+(eps, eps, eps) to
    // avoid thin/degenerate bounds
    bbox.extend(bbox.max()-Vector(epsilon, epsilon, epsilon));
    bbox.extend(bbox.min()+Vector(epsilon, epsilon, epsilon));

    // Make the bbox square...
    Point center(bbox.center());
    double le=1.0001*bbox.longest_edge();
    Vector diag(le, le, le);
    Point bmin(center-diag/2.);
    Point bmax(center+diag/2.);

    // Make the initial mesh with a tetra which encloses the bounding
    // box.  The first point is at the minimum point.  The other 3
    // have one of the coordinates at bmin+diagonal*3.
    mesh->nodes.add(new Node(bmin));
    mesh->nodes.add(new Node(bmin+Vector(le*3, 0, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, le*3, 0)));
    mesh->nodes.add(new Node(bmin+Vector(0, 0, le*3)));

    Element* el=new Element(mesh, nnodes+0, nnodes+1, nnodes+2, nnodes+3);
    el->orient();
    el->faces[0]=el->faces[1]=el->faces[2]=el->faces[3]=-1;
    mesh->elems.add(el);

    for(int node=0;node<nnodes;node++){
	// Add this node...
	update_progress(node, nnodes);

	if(!mesh->insert_delaunay(node)){
	    error("Mesher upset - point outside of domain...");
	    return;
	}
//	if(!mesh->elems[0])
//	    mesh->pack_elems();
    }
    mesh->pack_elems();
    mesh->compute_neighbors();
    oport->send(mesh);
}
