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

#include <Tester/RigorousTest.h>
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
#include <fstream.h>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/prctl.h>

class InsertDelaunay : public Module {
    MeshIPort* iport;
    MeshOPort* oport;
    Array1<SurfaceIPort*> surfports;
    int mesh_generation;
    MeshHandle last_mesh;
    Array1<int> surf_generations;
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
}

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

    // This is somewhat of a hack in order to keep from redoing the
    // mesh generation when only the boundary conditions change....
    int same=1;
    if(last_mesh.get_rep() == 0){
	cerr << "Not same because no last_mesh\n";
	same=0;
    }
    if(mesh_handle->generation != mesh_generation){
	cerr << "Not same because input mesh is different\n";
	same=0;
    }
    if(surf_generations.size() == surfs.size()){
	for(int i=0;i<surfs.size();i++){
	    if(surfs[i]->generation != surf_generations[i]){
		cerr << "Not same because surface " << i << " is different\n";
		same=0;
		break;
	    }
	}
    } else {
	cerr << "Not same because we have a different number of surfaces\n";
	same=0;
    }
    surf_generations.resize(surfs.size());
    for(i=0;i<surfs.size();i++){
	surf_generations[i]=surfs[i]->generation;
    }
    mesh_generation=mesh_handle->generation;
    
    update_progress(0, 6);
    mesh_handle.detach();
    update_progress(1, 6);
    mesh_handle->detach_nodes();

    if(same){
	cerr << "SAME!\n";
	int nelems=last_mesh->elems.size();
	mesh_handle->remove_all_elements();
	mesh_handle->elems.resize(nelems);
	for(int i=0;i<nelems;i++){
	    mesh_handle->elems[i]=new Element(*last_mesh->elems[i], mesh_handle.get_rep());
	}
	int isurf;
	int nsurfs=surfs.size();
	Array1<NodeHandle> newnodes;
	for(isurf=0;isurf<nsurfs;isurf++){
	    newnodes.remove_all();
	    surfs[isurf]->get_surfnodes(newnodes);
	    int npoints=newnodes.size();
	    for(int i=0;i<npoints;i++){
		mesh_handle->nodes.add(newnodes[i]);
	    }
	}
    } else {
	// Get our own copy of the mesh...
	Mesh* mesh=mesh_handle.get_rep();
	update_progress(2, 6);

	// Insert the points...
	int nsurfs=surfs.size();
	Array1<NodeHandle> newnodes;
	int isurf;
	for(isurf=0;isurf<nsurfs;isurf++){
	    newnodes.remove_all();
	    surfs[isurf]->get_surfnodes(newnodes);
	    int npoints=newnodes.size();
	    int nt=nsurfs*npoints;
	    for(int i=0;i<npoints;i++){
		mesh->nodes.add(newnodes[i]);
		mesh->insert_delaunay(mesh->nodes.size()-1);
		if(i%100 == 0)
		    update_progress(nt+isurf*npoints+i, 2*nt);
	    }
	}
    }
    last_mesh=mesh_handle;
    mesh_handle->compute_neighbors();

#if 0
    // Go through the mesh and remove all points that are
    // inside any of the inserted surfaces.
    int nelems=mesh_handle->elems.size();
    int ngone=0;
    int ntodo=2*nelems;
    int ndone=0;
    int nsurfs=surfs.size();
    for(i=0;i<nelems;i++){
	Element* e=mesh_handle->elems[i];
	if(e){
	    Point p=e->centroid();
	    for(int isurf=0;isurf<nsurfs;isurf++){
		if(surfs[isurf]->closed && surfs[isurf]->inside(p)){
		    delete e;
		    mesh_handle->elems[i]=0;
		    break;
		}
	    }
	}
	update_progress(i+nelems, 3*nelems);
    }
    mesh_handle->pack_all();
    int nnodes=mesh_handle->nodes.size();
    for(i=0;i<nnodes;i++){
	if(mesh_handle->nodes[i]->elems.size() == 0){
	    cerr << "Node " << i << " was orphaned" << endl;
	    mesh_handle->nodes[i]=0;
	}
	update_progress(i+2*nnodes, 3*nnodes);
    }
#endif
#if 0
    for(int isurf=0;isurf<nsurfs;isurf++){
	if(surfs[isurf]->closed){
	    for(int i=0;i<nnodes;i++){
		update_progress(ndone, ntodo);
		if(mesh_handle->nodes[i].get_rep() && 
		   surfs[isurf]->inside(mesh_handle->nodes[i]->p)){
		    // Remove this node...
		    cerr << "removing node " << i << endl;
		    mesh_handle->remove_delaunay(i, 0);
		    ngone++;
		}
	    }
	} else {
	    ntodo-=nnodes;
	}
    }
#endif
    update_progress(6, 6);
    mesh_handle->pack_all();
    cerr << "There are now " << mesh_handle->elems.size() << " elements" << endl;
    oport->send(mesh_handle);
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
