/*
 *  SetBCNode.cc:  SetBCNode Triangulation in 3D
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Math/MusilRNG.h>
#include <TCL/TCLvar.h>
#include <fstream.h>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
// #include <sys/prctl.h>

using sci::Element;
using sci::Mesh;
using sci::MeshHandle;
using sci::NodeHandle;

class SetBCNode : public Module {
    MeshIPort* miport;
    MeshOPort* moport;
    SurfaceIPort* surfport;
    int bcidx;
public:
    SetBCNode(const clString& id);
    SetBCNode(const SetBCNode&, int deep);
    virtual ~SetBCNode();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SetBCNode(const clString& id)
{
    return scinew SetBCNode(id);
}
}

SetBCNode::SetBCNode(const clString& id)
: Module("SetBCNode", id, Filter)
{
    miport=scinew MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(miport);
    surfport=scinew SurfaceIPort(this, "Added Surface", SurfaceIPort::Atomic);
    add_iport(surfport);

    // Create the output port
    moport=scinew MeshOPort(this, "SetBCNode Mesh", MeshIPort::Atomic);
    add_oport(moport);
    bcidx=-1;
}

SetBCNode::SetBCNode(const SetBCNode& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SetBCNode::SetBCNode");
}

SetBCNode::~SetBCNode()
{
}

Module* SetBCNode::clone(int deep)
{
    return scinew SetBCNode(*this, deep);
}

void SetBCNode::execute()
{
    MeshHandle mh;
    if(!miport->get(mh))
	return;
    SurfaceHandle sh;
    if (!surfport->get(sh))
	return;
    Surface *s=sh.get_rep();
    PointSurface *ps=dynamic_cast<PointSurface* >(s);
    if (!ps) {
	cerr << "Error - input surface wasn't a PointSurface.\n";
	return;
    }
    Array1<NodeHandle> nh;
    ps->get_surfnodes(nh);
    if (nh.size() != 1) {
	cerr << "Error - should just be one surface node in a PointSurface.\n";
	return;
    }
    Point p(nh[0]->p);

    if (bcidx != -1) {
	if(mh->nodes.size() > bcidx)
	    if (mh->nodes[bcidx]->bc) {
		delete(mh->nodes[bcidx]->bc);
		mh->nodes[bcidx]->bc=0;
	    }
    }

    int found=0;
    double dist;
    int idx;
    for (int i=0; i<mh->nodes.size(); i++) {
	if (mh->nodes[i]->bc) continue;
	double dd=(mh->nodes[i]->p-p).length();
	if (!found || dd<dist) {
	    dist=dd;
	    idx=i;
	    found=1;
	}
    }
    if (!found) {
	cerr << "Error - couldn't find any unused nodes!\n";
	return;
    }

    mh->nodes[idx]->bc = new DirichletBC(0, nh[0]->bc->value);
    cerr << "Set node "<<idx<<" to have Dirichlet BC value "<<nh[0]->bc->value<<"\n";
    bcidx=idx;
    moport->send(mh);
}
