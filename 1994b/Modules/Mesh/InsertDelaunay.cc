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
#include <Dataflow/ModuleList.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
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
};

static Module* make_InsertDelaunay(const clString& id)
{
    return new InsertDelaunay(id);
}

static RegisterModule db1("Mesh", "InsertDelaunay", make_InsertDelaunay);

InsertDelaunay::InsertDelaunay(const clString& id)
: Module("InsertDelaunay", id, Filter), surfports(4)
{
    iport=new MeshIPort(this, "Input Mesh", MeshIPort::Atomic);
    add_iport(iport);
    surfports[0]=new SurfaceIPort(this, "Added Surface", SurfaceIPort::Atomic);
    add_iport(surfports[0]);
    surfports[1]=new SurfaceIPort(this, "Added Surface", SurfaceIPort::Atomic);
    add_iport(surfports[1]);
    surfports[2]=new SurfaceIPort(this, "Added Surface", SurfaceIPort::Atomic);
    add_iport(surfports[2]);
    surfports[3]=new SurfaceIPort(this, "Added Surface", SurfaceIPort::Atomic);
    add_iport(surfports[3]);

    // Create the output port
    oport=new MeshOPort(this, "InsertDelaunay Mesh", MeshIPort::Atomic);
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
    return new InsertDelaunay(*this, deep);
}

void InsertDelaunay::execute()
{
    MeshHandle mesh_handle;
    if(!iport->get(mesh_handle))
	return;
    Array1<SurfaceHandle> surfs(surfports.size());
    for(int i=0;i<surfports.size();i++){
	if(!surfports[i]->get(surfs[i]))
	   return;
    }

    // Get our own copy of the mesh...
    mesh_handle.detach();
    Mesh* mesh=mesh_handle.get_rep();

    // Insert the points...
    int nsurfs=surfports.size();
    Array1<Point> points;
    for(int isurf=0;isurf<nsurfs;isurf++){
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
	    for(i=0;i<nnodes;i++){
		update_progress(ndone, ntodo);
		if(mesh->nodes[i] && surfs[isurf]->inside(mesh->nodes[i]->p)){
		    // Remove this node...
		    mesh->remove_delaunay(i, 0);
		    ngone++;
		}
	    }
	} else {
	    ntodo-=nnodes;
	}
    }
    mesh->compute_neighbors();
    mesh->pack_nodes();
    mesh->pack_elems();
    oport->send(mesh);
}
