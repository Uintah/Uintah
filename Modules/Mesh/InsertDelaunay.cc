/*
 *  InsertDelaunay.cc:  InsertDelaunay Triangulation in 3D
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Math/MusilRNG.h>
#include <TCL/TCLvar.h>

class InsertDelaunay : public Module {
    MeshIPort* iport;
    MeshOPort* oport;
    Array1<SurfaceIPort*> surfports;
public:
    InsertDelaunay(const clString& id);
    InsertDelaunay(const InsertDelaunay&, int deep);
    virtual ~InsertDelaunay();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void connection(ConnectionMode mode, int which_port, int);
};

extern "C" {
Module* make_InsertDelaunay(const clString& id)
{
    return scinew InsertDelaunay(id);
}
};

InsertDelaunay::InsertDelaunay(const clString& id)
: Module("InsertDelaunay", id, Filter)
{
    iport=scinew MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(iport);
    surfports.add(scinew SurfaceIPort(this, "Added Surface", SurfaceIPort::Atomic));
    add_iport(surfports[0]);

    // Create the output port
    oport=scinew MeshOPort(this, "InsertDelaunay Mesh", MeshIPort::Atomic);
    add_oport(oport);
}

InsertDelaunay::InsertDelaunay(const InsertDelaunay& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("InsertDelaunay::InsertDelaunay");
}

InsertDelaunay::~InsertDelaunay()
{
}

Module* InsertDelaunay::clone(int deep)
{
    return scinew InsertDelaunay(*this, deep);
}

void InsertDelaunay::execute()
{
    MeshHandle mesh_handle;
    if(!iport->get(mesh_handle))
	return;
    Array1<SurfaceHandle> surfs(surfports.size()-1);
    for(int i=0;i<surfs.size();i++){
	if(!surfports[i]->get(surfs[i]))
	   return;
    }

    // Get our own copy of the mesh...
    update_progress(0, 6);
    mesh_handle.detach();
    update_progress(1, 6);
    mesh_handle->detach_nodes();
    Mesh* mesh=mesh_handle.get_rep();
    update_progress(3, 6);
    mesh->compute_neighbors();

    // Insert the points...
    int nsurfs=surfs.size();
    Array1<Point> points;
    int isurf;
    for(isurf=0;isurf<nsurfs;isurf++){
	points.remove_all();
	surfs[isurf]->get_surfpoints(points);
	int npoints=points.size();
	for(int i=0;i<npoints;i++){
	    mesh->insert_delaunay(points[i]);
	    update_progress(i, npoints);
	}
    }
    mesh->compute_neighbors();

    // Go through the mesh and remove all points that are
    // inside any of the inserted surfaces.
    int nnodes=mesh->nodes.size();
    int ngone=0;
    int ntodo=2*nnodes;
    int ndone=0;
    for(isurf=0;isurf<nsurfs;isurf++){
	if(surfs[isurf]->closed){
	    for(int i=0;i<nnodes;i++){
		update_progress(ndone, ntodo);
		if(mesh->nodes[i].get_rep() && 
		   surfs[isurf]->inside(mesh->nodes[i]->p)){
		    // Remove this node...
		    mesh->remove_delaunay(i, 0);
		    ngone++;
		}
	    }
	} else {
	    ntodo-=nnodes;
	}
    }
    mesh->pack_all();
    oport->send(mesh);
}

void InsertDelaunay::connection(ConnectionMode mode, int which_port, int)
{
    if(which_port > 0){
	if(mode==Disconnected){
	    remove_iport(which_port);
	    surfports.remove(which_port-1);
	} else {
	    SurfaceIPort* p=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
	    surfports.add(p);
	    add_iport(p);
	}
    }
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<SurfaceIPort*>;
template class Array1<SurfaceHandle>;

#endif
