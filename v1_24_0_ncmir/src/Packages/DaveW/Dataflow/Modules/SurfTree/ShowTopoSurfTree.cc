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

#include <Packages/DaveW/Core/Datatypes/General/TopoSurfTree.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Pt.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class TopoSurfToGeom : public Module {
    SurfaceIPort* isurface;
    GeometryOPort* ogeom;

    Array1<MaterialHandle> c;

    GuiString patchMode;
    GuiString wireMode;
    GuiString junctionMode;
    GuiString nonjunctionMode;
    GuiString rad;

    void surf_to_geom(const SurfaceHandle&, GeomGroup*);
public:
    TopoSurfToGeom(const clString& id);
    virtual ~TopoSurfToGeom();
    virtual void execute();
};

extern "C" Module* make_TopoSurfToGeom(const clString& id)
{
    return scinew TopoSurfToGeom(id);
}

TopoSurfToGeom::TopoSurfToGeom(const clString& id)
: Module("TopoSurfToGeom", id, Filter), patchMode("patchMode", id, this),
  wireMode("wireMode", id, this), junctionMode("junctionMode", id, this),
  rad("rad", id, this), nonjunctionMode("nonjunctionMode", id, this)
{
    // Create the input port
    isurface=scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    c.resize(14);
    c[0]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
    c[2]=scinew Material(Color(.2,.2,.2),Color(.1,.1,.7),Color(.5,.5,.5),20);
    c[3]=scinew Material(Color(.2,.2,.2),Color(.7,.7,.1),Color(.5,.5,.5),20);
    c[4]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.7),Color(.5,.5,.5),20);
    c[5]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.7),Color(.5,.5,.5),20);
    c[6]=scinew Material(Color(.2,.2,.2),Color(.6,.6,.6),Color(.5,.5,.5),20);
    c[7]=scinew Material(Color(.2,.2,.2),Color(.0,.3,.0),Color(.5,.5,.5),20);
    c[8]=scinew Material(Color(.2,.2,.2),Color(.3,.0,.0),Color(.5,.5,.5),20);
    c[9]=scinew Material(Color(.2,.2,.2),Color(.0,.0,.3),Color(.5,.5,.5),20);
    c[10]=scinew Material(Color(.2,.2,.2),Color(.3,.3,.0),Color(.5,.5,.5),20);
    c[11]=scinew Material(Color(.2,.2,.2),Color(.3,.0,.3),Color(.5,.5,.5),20);
    c[12]=scinew Material(Color(.2,.2,.2),Color(.0,.3,.3),Color(.5,.5,.5),20);
    c[13]=scinew Material(Color(.2,.2,.2),Color(.3,.3,.3),Color(.5,.5,.5),20);
}

TopoSurfToGeom::~TopoSurfToGeom()
{
	ogeom->delAll();
}

static MaterialHandle outmatl(new Material(Color(0,0,0), Color(0,0,0),
				    Color(0,0,0), 0));
static MaterialHandle black(new Material(Color(.2,.2,.2), Color(.2,.2,.2), 
				  Color(.5,.5,.5), 30));
static Color blackClr(.2,.2,.2);

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
    Array1<GeomObj* > Junctions;
    //    Array1<GeomObj* > NonJunctions;

    TopoSurfTree* topo=dynamic_cast<TopoSurfTree*>(surf.get_rep());
    if (!topo) {
	SurfTree *st=surf->getSurfTree();
//	if (st) topo=st->toTopoSurfTree();
    }
    if (!topo) {
	cerr << "Error - can only deal with Topo surfaces.\n";
	return;
    }

    reset_vars();
    if (patchMode.get() != "patchNone" && topo->patches.size()) {
	Patches.resize(topo->patches.size());
	int i;
	for (i=0; i<Patches.size(); i++)
	    Patches[i] = scinew GeomTriangles;
	for (i=0; i<topo->patches.size(); i++)
	    for (int j=0; j<topo->patchI[i].faces.size(); j++) {
		int elIdx=topo->patchI[i].faces[j];
		int i1=topo->faces[elIdx]->i1;
		int i2=topo->faces[elIdx]->i2;
		int i3=topo->faces[elIdx]->i3;
		if (topo->patchI[i].facesOrient[0][j])	// 0 is arbitrary (1)
		    Patches[i]->add(topo->nodes[i1], 
				    topo->nodes[i2],
				    topo->nodes[i3]);
		else	
		    Patches[i]->add(topo->nodes[i1], 
				    topo->nodes[i3],
				    topo->nodes[i2]);
	    }	
    }
    if (wireMode.get() != "wireNone" && topo->wires.size()) {
	Wires.resize(topo->wires.size());
	int i;
	for (i=0; i<Wires.size(); i++)
	    Wires[i] = scinew GeomLines;
	for (i=0; i<topo->wires.size(); i++)
	    for (int j=0; j<topo->wireI[i].edges.size(); j++) {
		int lnIdx=topo->wireI[i].edges[j];
		int i1=topo->edges[lnIdx]->i1;
		int i2=topo->edges[lnIdx]->i2;
		if (topo->wireI[i].edgesOrient[0][j]) // 0 is arbitrary (1,2)
		    Wires[i]->add(topo->nodes[i1], topo->nodes[i2]);
		else
		    Wires[i]->add(topo->nodes[i2], topo->nodes[i1]);
	    }
    }
    if (junctionMode.get() != "junctionNone" && topo->junctions.size()) {
	double r;	
	(rad.get()).get_double(r);
	if (r == 0) {
	    GeomPts *gpts = scinew GeomPts(0);
	    for (int i=0; i<topo->junctions.size(); i++) {
		gpts->add(topo->nodes[topo->junctions[i]]);
	    }
	    Junctions.add(gpts);
	} else {
	    for (int i=0; i<topo->junctions.size(); i++) {
		Junctions.add(scinew GeomSphere(topo->nodes[topo->junctions[i]], r));
	    }
	}
    }
#if 0
    if (nonjunctionMode.get() != "nonjunctionNone" && topo->junctionlessWires.size()) {
	double r;
	(rad.get()).get_double(r);
	cerr << "r="<<r<<"\n";
	if (r == 0) {
	    GeomPts *gpts = scinew GeomPts(0);
	    cerr << "1)topo->junctionlessWires.size()="<<topo->junctionlessWires.size()<<"\n";
	    for (int i=0; i<topo->junctionlessWires.size(); i++) {
		gpts->add(topo->nodes[topo->junctionlessWires[i]]);
	    }
	    NonJunctions.add(gpts);
	} else {
	    cerr << "2)topo->junctionlessWires.size()="<<topo->junctionlessWires.size()<<"\n";
	    for (int i=0; i<topo->junctionlessWires.size(); i++) {
		NonJunctions.add(scinew GeomSphere(topo->nodes[topo->junctionlessWires[i]], r));
	    }
	}
    }
#endif

    ogeom->delAll();

    if (Patches.size()) {
	if (patchMode.get() == "patchTog") {
	    GeomGroup* gr=scinew GeomGroup;
	    for (int i=0; i<Patches.size(); i++) {
		gr->add(scinew GeomMaterial(Patches[i], c[i%14]));
	    }
	    ogeom->addObj(gr, clString("Topo Patches"));
	} else {
	    for (int i=0; i<Patches.size(); i++) {
		clString name("Topo Patch "+to_string(i));
		ogeom->addObj(scinew GeomMaterial(Patches[i], c[i%14]), name);
	    }
	}
    }	    
    if (Wires.size()) {
	if (wireMode.get() == "wireTog") {
	    GeomGroup* gr=scinew GeomGroup;
	    for (int i=0; i<Wires.size(); i++) {
		gr->add(scinew GeomMaterial(Wires[i], c[i%14]));
	    }
	    ogeom->addObj(gr, clString("Topo Wires"));
	} else {
	    for (int i=0; i<Wires.size(); i++) {
		clString name("Topo Wire "+to_string(i));
		ogeom->addObj(scinew GeomMaterial(Wires[i], c[i%14]), name);
	    }
	}
    }
    if (Junctions.size()) {
	if (junctionMode.get() == "junctionTog") {
	    GeomGroup* gr=scinew GeomGroup;
	    for (int i=0; i<Junctions.size(); i++)
		gr->add(scinew GeomMaterial(Junctions[i], c[i%14]));
	    ogeom->addObj(gr, clString("Topo Junctions"));
	} else {
	    for (int i=0; i<Junctions.size(); i++) {
		clString name("Topo Junction "+to_string(i));
		ogeom->addObj(scinew GeomMaterial(Junctions[i], c[i%14]), name);
	    }
	}
    }
#if 0
    if (NonJunctions.size()) {
	if (nonjunctionMode.get() == "nonjunctionTog") {
	    GeomGroup* gr=scinew GeomGroup;
	    for (int i=0; i<NonJunctions.size(); i++)
		gr->add(scinew GeomMaterial(NonJunctions[i], c[i%14]));
	    ogeom->addObj(gr, clString("Topo NonJunctions"));
	} else {
	    for (int i=0; i<NonJunctions.size(); i++) {
		clString name("Topo NonJunction "+to_string(i));
		ogeom->addObj(scinew GeomMaterial(NonJunctions[i], c[i%14]), name);
	    }
	}
    }
#endif
}
} // End namespace DaveW


