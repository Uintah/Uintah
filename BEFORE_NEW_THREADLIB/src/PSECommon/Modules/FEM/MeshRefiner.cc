//static char *id="@(#) $Id$";

/*
 *  MeshRefiner.cc: Evaluate the error in a finite element solution
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/BooleanPort.h>
#include <PSECore/Datatypes/IntervalPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class MeshRefiner : public Module {
    ScalarFieldIPort* upbound_field;
    ScalarFieldIPort* lowbound_field;
    IntervalIPort* interval_iport;
    MeshOPort* mesh_oport;
    sciBooleanOPort* cond_oport;
public:
    MeshRefiner(const clString& id);
    virtual ~MeshRefiner();
    virtual void execute();
};

Module* make_MeshRefiner(const clString& id) {
  return new MeshRefiner(id);
}

MeshRefiner::MeshRefiner(const clString& id)
: Module("MeshRefiner", id, Filter)
{
    lowbound_field=new ScalarFieldIPort(this, "Lower bound",
					ScalarFieldIPort::Atomic);
    add_iport(lowbound_field);
    upbound_field=new ScalarFieldIPort(this, "Upper bound",
				       ScalarFieldIPort::Atomic);
    add_iport(upbound_field);
    interval_iport=new IntervalIPort(this, "Error interval",
				     IntervalIPort::Atomic);
    add_iport(interval_iport);

    // Create the output port
    mesh_oport=new MeshOPort(this, "Refined mesh",
			     MeshIPort::Atomic);
    add_oport(mesh_oport);
    cond_oport=new sciBooleanOPort(this, "Stopping condition",
				   sciBooleanIPort::Atomic);
    add_oport(cond_oport);
}

MeshRefiner::~MeshRefiner()
{
}

void MeshRefiner::execute()
{
  cerr << "MeshRefiner executing...\n";
    ScalarFieldHandle lowf;
    if(!lowbound_field->get(lowf))
        return;
    ScalarFieldHandle upf;
    if(!upbound_field->get(upf))
        return;
    IntervalHandle interval;
    if(!interval_iport->get(interval))
        return;
    ScalarFieldUG* lowfug=lowf->getUG();
    if(!lowfug){
	error("ComposeError can't deal with this field");
	return;
    }
    ScalarFieldUG* upfug=upf->getUG();
    if(!upfug){
	error("ComposeError can't deal with this field");
	return;
    }
    if(upfug->mesh.get_rep() != lowfug->mesh.get_rep()){
        error("Two different meshes...\n");
	return;
    }

    MeshHandle mesh(upfug->mesh);
    //double* low=&lowfug->data[0];
    double* up=&upfug->data[0];
    double upper_bound=interval->high;
    Array1<Point> new_points;
    int nelems=mesh->elems.size();
    int i;
    for(i=0;i<nelems;i++){
        if(up[i] > upper_bound){
	    new_points.add(mesh->elems[i]->centroid());
	}
	if(i%10000 == 0)
	  update_progress(i, 2*nelems);
    }
    if(new_points.size() == 0){
        mesh_oport->send(mesh);
	cond_oport->send(new sciBoolean(1));
	return;
    }

    mesh.detach();
    mesh->detach_nodes();
    mesh->compute_face_neighbors();
    cond_oport->send(new sciBoolean(0));
    cerr << "There are " << new_points.size() << " new points\n";
    for(i=0;i<new_points.size();i++){
        if(!mesh->insert_delaunay(new_points[i]))
	    cerr << "Error inserting point: " << new_points[i] << endl;
	if(i%1000 == 0)
	  update_progress(new_points.size()+i, 2*new_points.size());
    }
    mesh->pack_all();
    cerr << "There are now " << mesh->nodes.size() << " nodes and " << mesh->elems.size() << " elements\n";
    mesh_oport->send(mesh);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5  1999/08/25 03:47:45  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:42  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:37  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:26  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:40  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:48  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
