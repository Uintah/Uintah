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
    MeshHandle mesh;
    if(!iport->get(mesh))
	return;

    // Get our own copy of the mesh...
    mesh.detach();
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

    // Make the initial mesh with 
    int bbox_node_start=nnodes;
    Point bmin(bbox.min());
    Point bmax(bbox.max());
    for(i=0;i<2;i++)
	for(int j=0;j<2;j++)
	    for(int k=0;k<2;k++)
		mesh->nodes.add(new Node(Point(i?bmax.x():bmin.x(),
					       j?bmax.y():bmin.y(),
					       k?bmax.z():bmin.z())));
    NOT_FINISHED("Delaunay::execute");
}
