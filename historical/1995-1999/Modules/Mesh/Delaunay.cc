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
#include <Datatypes/GeometryPort.h>
#include <Datatypes/MeshPort.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

using sci::Mesh;
using sci::MeshHandle;
using sci::Node;
using sci::Element;

class Delaunay : public Module {
    MeshIPort* iport;
    MeshOPort* oport;
    GeometryOPort* ogeom;
    TCLint nnodes;
    TCLint cleanup;
public:
    Delaunay(const clString& id);
    Delaunay(const Delaunay&, int deep);
    virtual ~Delaunay();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_Delaunay(const clString& id)
{
    return scinew Delaunay(id);
}
}

Delaunay::Delaunay(const clString& id)
: Module("Delaunay", id, Filter), nnodes("nnodes", id, this),
  cleanup("cleanup", id, this)
{
    iport=scinew MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=scinew MeshOPort(this, "Delaunay Mesh", MeshIPort::Atomic);
    add_oport(oport);
    ogeom=scinew GeometryOPort(this, "Animation", GeometryIPort::Atomic);
    add_oport(ogeom);
}

Delaunay::Delaunay(const Delaunay& copy, int deep)
: Module(copy, deep), nnodes("nnodes", id, this),
  cleanup("cleanup", id, this)
{
    NOT_FINISHED("Delaunay::Delaunay");
}

Delaunay::~Delaunay()
{
}

Module* Delaunay::clone(int deep)
{
    return scinew Delaunay(*this, deep);
}

void Delaunay::execute()
{
    MeshHandle mesh_handle;
    if(!iport->get(mesh_handle))
	return;
    ogeom->delAll();

    // Get our own copy of the mesh...
    mesh_handle.detach();
    mesh_handle->detach_nodes();
    Mesh* mesh=mesh_handle.get_rep();
    mesh->elems.remove_all();
    mesh->compute_neighbors();

    int nn=mesh->nodes.size();
    BBox bbox;
    int i;
    for(i=0;i<nn;i++)
	bbox.extend(mesh->nodes[i]->p);

    double epsilon=.1*bbox.longest_edge();

    // Extend by max-(eps, eps, eps) and min+(eps, eps, eps) to
    // avoid thin/degenerate bounds
    Point max(bbox.max()+Vector(epsilon, epsilon, epsilon));
    Point min(bbox.min()-Vector(epsilon, epsilon, epsilon));

    mesh->nodes.add(new Node(Point(min.x(), min.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), min.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), min.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), min.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), max.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), max.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), max.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), max.y(), max.z())));

    Element* el1=new Element(mesh, nn+0, nn+1, nn+4, nn+3);
    Element* el2=new Element(mesh, nn+2, nn+1, nn+3, nn+6);
    Element* el3=new Element(mesh, nn+7, nn+3, nn+6, nn+4);
    Element* el4=new Element(mesh, nn+5, nn+6, nn+4, nn+1);
    Element* el5=new Element(mesh, nn+1, nn+3, nn+4, nn+6);
    el1->faces[0]=4; el1->faces[1]=-1; el1->faces[2]=-1; el1->faces[3]=-1;
    el2->faces[0]=4; el2->faces[1]=-1; el2->faces[2]=-1; el2->faces[3]=-1;
    el3->faces[0]=4; el3->faces[1]=-1; el3->faces[2]=-1; el3->faces[3]=-1;
    el4->faces[0]=4; el4->faces[1]=-1; el4->faces[2]=-1; el4->faces[3]=-1;
    el5->faces[0]=2; el5->faces[1]=3; el5->faces[2]=1; el5->faces[3]=0;
    el1->orient();
    el2->orient();
    el3->orient();
    el4->orient();
    el5->orient();
    mesh->elems.add(el1);
    mesh->elems.add(el2);
    mesh->elems.add(el3);
    mesh->elems.add(el4);
    mesh->elems.add(el5);

    int onn=nn;
    GeometryOPort* aport=ogeom;
    if(ogeom->nconnections() == 0)
	aport=0;
    nn=nnodes.get();
    if(nn==0 || nn > onn)nn=onn;
    for(int node=0;node<nn;node++){
	// Add this node...
	update_progress(node, nn);

	if(!mesh->insert_delaunay(node, aport)){
	    error("Mesher upset - point outside of domain...");
	    return;
	}
	// Every 1000 nodes, cleanup the elems array...
	if(node%1000 == 0){
cerr << "Adding node " << node << endl;
	    mesh->pack_elems();
	}
    }
    mesh->compute_neighbors();
    if(cleanup.get()){
	mesh->remove_delaunay(onn, 0);
	mesh->remove_delaunay(onn+1, 0);
	mesh->remove_delaunay(onn+2, 0);
	mesh->remove_delaunay(onn+3, 0);
	mesh->remove_delaunay(onn+4, 0);
	mesh->remove_delaunay(onn+5, 0);
	mesh->remove_delaunay(onn+6, 0);
	mesh->remove_delaunay(onn+7, 0);
    }
    mesh->pack_all();
    double vol=0;
    cerr << "There are " << mesh->elems.size() << " elements" << endl;
    for(i=0;i<mesh->elems.size();i++){
	vol+=mesh->elems[i]->volume();
    }
    cerr << "Total volume: " << vol << endl;
    oport->send(mesh);
}
