/*
 *  TopoSurfToGeom.cc:  Convert a topological surface into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/TopoSurfTree.h>
#include <Datatypes/SurfacePort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Group.h>
#include <Geom/Pt.h>
#include <Geom/Line.h>
#include <Geom/Triangles.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class TopoSurfToGeom : public Module {
    SurfaceIPort* isurface;
    GeometryOPort* ogeom;

    Array1<MaterialHandle> c;

    TCLstring mode;

    void surf_to_geom(const SurfaceHandle&, GeomGroup*);
public:
    TopoSurfToGeom(const clString& id);
    TopoSurfToGeom(const TopoSurfToGeom&, int deep);
    virtual ~TopoSurfToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_TopoSurfToGeom(const clString& id)
{
    return scinew TopoSurfToGeom(id);
}
}

TopoSurfToGeom::TopoSurfToGeom(const clString& id)
: Module("TopoSurfToGeom", id, Filter), mode("mode", id, this)
{
    // Create the input port
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    c.resize(7);
    c[0]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
    c[2]=scinew Material(Color(.2,.2,.2),Color(.1,.1,.7),Color(.5,.5,.5),20);
    c[3]=scinew Material(Color(.2,.2,.2),Color(.7,.7,.1),Color(.5,.5,.5),20);
    c[4]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.7),Color(.5,.5,.5),20);
    c[5]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.7),Color(.5,.5,.5),20);
    c[6]=scinew Material(Color(.2,.2,.2),Color(.6,.6,.6),Color(.5,.5,.5),20);
}

TopoSurfToGeom::TopoSurfToGeom(const TopoSurfToGeom&copy, int deep)
: Module(copy, deep), mode("mode", id, this)
{
    NOT_FINISHED("TopoSurfToGeom::TopoSurfToGeom");
}

TopoSurfToGeom::~TopoSurfToGeom()
{
	ogeom->delAll();
}

Module* TopoSurfToGeom::clone(int deep)
{
    return scinew TopoSurfToGeom(*this, deep);
}

MaterialHandle outmatl(new Material(Color(0,0,0), Color(0,0,0),
				    Color(0,0,0), 0));
MaterialHandle black(new Material(Color(.2,.2,.2), Color(.2,.2,.2), 
				  Color(.5,.5,.5), 30));
Color blackClr(.2,.2,.2);

void TopoSurfToGeom::execute()
{
    SurfaceHandle surf;
    update_state(NeedData);
    reset_vars();
    if (!isurface->get(surf)){
	ogeom->delAll();
	return;
    }
    if (!surf.get_rep()) return;
    update_state(JustStarted);
    reset_vars();

    Array1<GeomTriangles* > Patches;
    Array1<GeomLines* > Wires;
    GeomPts *Junctions = 0;

    TopoSurfTree* topo=surf->getTopoSurfTree();
    if (!topo) {
	SurfTree *st=surf->getSurfTree();
	if (st) topo=st->toTopoSurfTree();
    }
    if (!topo) {
	cerr << "Error - can only deal with Topo surfaces.\n";
	return;
    }

    reset_vars();
    if (mode.get() == "patches") {
	Patches.resize(topo->patches.size());
	for (int i=0; i<Patches.size(); i++)
	    Patches[i] = scinew GeomTriangles;
	for (i=0; i<topo->patches.size(); i++)
	    for (int j=0; j<topo->patches[i].size(); j++) {
		int elIdx=topo->patches[i][j];
		int i1=topo->faces[elIdx]->i1;
		int i2=topo->faces[elIdx]->i2;
		int i3=topo->faces[elIdx]->i3;
		if (topo->patchesOrient[i][0][j])	// 0 is arbitrary (1)
		    Patches[i]->add(topo->nodes[i1], 
				    topo->nodes[i2],
				    topo->nodes[i3]);
		else	
		    Patches[i]->add(topo->nodes[i1], 
				    topo->nodes[i3],
				    topo->nodes[i2]);
	    }	
    } else if (mode.get() == "wires") {
	Wires.resize(topo->wires.size());
	for (int i=0; i<Wires.size(); i++)
	    Wires[i] = scinew GeomLines;
	for (i=0; i<topo->wires.size(); i++)
	    for (int j=0; j<topo->wires[j].size(); j++) {
		int lnIdx=topo->wires[i][j];
		int i1=topo->edges[lnIdx]->i1;
		int i2=topo->edges[lnIdx]->i2;
		if (topo->wiresOrient[i][0][j])		// 0 is arbitrary (1,2)
		    Wires[i]->add(topo->nodes[i1], topo->nodes[i2]);
		else
		    Wires[i]->add(topo->nodes[i2], topo->nodes[i1]);
	    }
    } else {	// mode.get() == "junctions"
	Junctions = new GeomPts(topo->junctions.size());
	for (int i=0; i<topo->junctions.size(); i++) {
	    Junctions->pts[i*3]=topo->nodes[topo->junctions[i]].x();
	    Junctions->pts[i*3+1]=topo->nodes[topo->junctions[i]].y();
	    Junctions->pts[i*3+2]=topo->nodes[topo->junctions[i]].z();
	}
    }

    ogeom->delAll();
    if (Junctions) {
	ogeom->addObj(Junctions, clString("Topo Junctions"));
	return;
    }
    GeomGroup* ngroup = scinew GeomGroup;
    if (Wires.size()) {
	for (int i=0; i<Wires.size(); i++) {
	    ngroup->add(scinew GeomMaterial(Wires[i], c[i%7]));
	}
	ogeom->addObj(ngroup, clString("Topo Wires"));
    } else {	// Patches
	for (int i=0; i<Patches.size(); i++)
	    ngroup->add(scinew GeomMaterial(Patches[i], c[i%7]));
	ogeom->addObj(ngroup, clString("Topo Patches"));
    }
}
