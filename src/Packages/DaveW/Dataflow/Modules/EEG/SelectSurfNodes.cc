//static char *id="@(#) $Id$";

/*
 *  SelectSurfNodes.cc:  Select a set of nodes from a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1998
 *
 *  Copyright (C) 1998 SCI Group
 *
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <PSECore/Widgets/PointWidget.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Util/NotFinished.h>

#include <iostream.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::GeomSpace;

class SelectSurfNodes : public Module {
    SurfaceIPort* iport_surf;
    SurfaceOPort* oport_surf;
    GeomPts* geomPts;
    GeometryOPort* ogeom;
    GeomPick* gpk;
    int sel_geom_idx;
    int igen;
    Array1<int> sel_idx;
    Array1<Point> ipts;
    Array1<Point> opts;
    SurfaceHandle oSurfHdl;
    CrowdMonitor widget_lock;
    TCLstring method;
    TCLdouble sphereSize;
    MaterialHandle orange;		// for spheres
    clString tcl_msg;
    int pts_changed;
    int add_idx;
    int del_idx;
public:
    SelectSurfNodes(const clString& id);
    virtual ~SelectSurfNodes();
    virtual void execute();
    virtual void geom_pick(GeomPick*, void*);
    virtual void tcl_command(TCLArgs&, void*);
};

Module* make_SelectSurfNodes(const clString& id)
{
    return new SelectSurfNodes(id);
}

static clString module_name("SelectSurfNodes");

SelectSurfNodes::SelectSurfNodes(const clString& id)
: Module("SelectSurfNodes", id, Filter), method("method", id, this),
  sel_geom_idx(0), pts_changed(0), sphereSize("sphereSize", id, this), igen(0)
{
    // Create the input ports
    iport_surf=new SurfaceIPort(this, "ISurf", SurfaceIPort::Atomic);
    add_iport(iport_surf);
    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    oport_surf=new SurfaceOPort(this, "OSurf", SurfaceIPort::Atomic);
    add_oport(oport_surf);
    orange = scinew Material(Color(.2,.2,.2), Color(.8,.4,0),
			   Color(.5,.5,.5), 20);
}

SelectSurfNodes::~SelectSurfNodes()
{
}

void SelectSurfNodes::execute()
{
    SurfaceHandle iScalpHdl;
    SurfaceHandle iSurfHdl;

    if(!iport_surf->get(iSurfHdl))
	return;
    if (!(iSurfHdl.get_rep())) return;
    if (iSurfHdl->generation != igen) {
	tcl_msg = "clear";
	igen = iSurfHdl->generation;
    }
    cerr << "tcl_msg = "<<tcl_msg<<"\n";
    if (tcl_msg != "") {
	clString tcl_msg_copy(tcl_msg);
	tcl_msg = "";	// note: this is not thread-safe -- want a locking
                    	// mailbox here, instead.
	if (tcl_msg_copy == "clear") {
	    sel_idx.resize(0);
	    ipts.resize(0);
	    opts.resize(0);
	    ogeom->delAll();
	    sel_geom_idx=0;
	    Array1<NodeHandle> nh;
	    iSurfHdl->get_surfnodes(nh);
	    GeomPts* gpts=scinew GeomPts(nh.size());
	    for (int s=0; s<nh.size(); s++) {
		ipts.add(nh[s]->p);
		gpts->add(nh[s]->p);
	    }
	    gpk=scinew GeomPick(gpts, this);
	    gpts->pickable=1;
	    gpk->drawOnlyOnPick=0;
	    ogeom->addObj(gpk, "Pickable surface nodes");
	    pts_changed = 1;
	} else if (tcl_msg_copy == "loadlist") {
	    cerr << "Haven't add support for tcl list indexing yet.\n";
	} else if (tcl_msg_copy == "addnode") {
	    cerr << "Attempting to add node "<<add_idx<<"\n";
	    int add_idx_copy(add_idx);
	    int i;
	    for (i=0; i<sel_idx.size(); i++) {
		if (sel_idx[i] == add_idx_copy) break;
	    }
	    if (i != sel_idx.size()) {
		cerr << "Node "<<add_idx<<" has already been added.\n";
	    } else {
		cerr << "Added node "<<add_idx<<"\n";
		sel_idx.add(add_idx_copy);
		opts.add(ipts[add_idx_copy]);
		pts_changed = 1;
	    }
	} else if (tcl_msg_copy == "delnode") {
	    cerr << "Attempting to add node "<<del_idx<<"\n";
	    int del_idx_copy(del_idx);
	    int i;
	    for (i=0; i<sel_idx.size(); i++) {
		if (sel_idx[i] == del_idx_copy) break;
	    }
	    if (i == sel_idx.size()) {
		cerr << "Node "<<del_idx<<" hasn't been added.\n";
	    } else { 
		cerr << "Deleted node "<<del_idx<<"\n";
		sel_idx.remove(i);
		opts.remove(i);
		pts_changed = 1;
	    }
	} else {
	    cerr << "Unrecognized msg in SelectSurfNodes.  No action taken.\n";
	}
    }
    if (!pts_changed) return;

    if (sel_geom_idx) ogeom->delObj(sel_geom_idx);
    double sphsize = sphereSize.get();
    GeomGroup *gg = new GeomGroup;
    for (int i=0; i<sel_idx.size(); i++) {
	GeomObj *go;
	GeomMaterial *gm;
	go = scinew GeomSphere(ipts[sel_idx[i]], sphsize);
	gm = scinew GeomMaterial(go, orange);
	gg->add(gm);
    }
    sel_geom_idx=ogeom->addObj(gg, "Selected Nodes");
    if (pts_changed) {
	TriSurface *ts = new TriSurface;
	ts->points = opts;
	oSurfHdl = ts;
	pts_changed = 0;
    }
    oport_surf->send(oSurfHdl);
}	

void SelectSurfNodes::geom_pick(GeomPick* pick, void* cb)
{
    int *val((int *)cb);
    if (cb && (*val != -1234)) {	
	int picknode = *val;
	reset_vars();
	cerr <<"NODE "<<picknode<<" was picked.\n";
	cerr << "method.get() == "<<method.get()<<"\n";
	if (method.get() == "addnode") {
	    add_idx = picknode;
	    tcl_msg = "addnode";
	} else if (method.get() == "delnode") {
	    del_idx = picknode;
	    tcl_msg = "delnode";
	}
	want_to_execute();
    }
}


void SelectSurfNodes::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2) {
            args.error("SelectSurfNodes needs a minor command");
            return;
    }
    if(args[1] == "clear") {
	    tcl_msg = "clear";
            want_to_execute();
    } else {
            Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.1  1999/08/24 06:23:03  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:14  dmw
// Added and updated DaveW Datatypes/Modules
//
//
