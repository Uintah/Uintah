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
    double le=bbox.longest_edge();
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

    Element* el=new Element(mesh, 0, 1, 2, 3);
    el->orient();
    mesh->elems.add(el);

    for(i=0;i<nnodes;i++){
	// Add this node...
	Point p(mesh->nodes[i]->p);

	// Find which element this node is in
	int in_element;
	if(!mesh->locate(p, in_element)){
	    error("Mesher upset - point outside of domain...");
	    return;
	}

	Array1<int> to_remove;
	to_remove.add(in_element);

	// Find it's neighbors...
	// We might be able to fix this loop to make it
	// O(N) instead of O(n^2) - use a Queue
	int i=0;
	while(i<to_remove.size()){
	    // See if the neighbor should also be removed...
#if 0
	    Element* e=mesh->elems[to_remove[i]];
	    for(int j=0;j<4;j++){
		int skip=0;
		int neighbor=e->faces[j];
		for(int ii=0;ii<to_remove.size();i++){
		    if(neighbor==to_remove[ii])
#endif
	}
    }
}
