//static char *id="@(#) $Id$";

/*
 *  Radiosity.cc:  Take in a scene (through a VoidStarPort), and output 
 *		   Geometry and the editted scene
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <DaveW/Datatypes/CS684/DRaytracer.h>
#include <DaveW/Datatypes/CS684/Spectrum.h>
#include <DaveW/Datatypes/CS684/RTPrims.h>
#include <DaveW/Datatypes/CS684/RadPrims.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/VoidStarPort.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/Stack.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/VoidStar.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomCone.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Widgets;
using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::TclInterface;

class Radiosity : public Module {
    VoidStarIPort *iRT;
    VoidStarOPort *oRT;
    GeometryOPort *ogeom;
    DRaytracer *rt;
    int rtGen;
    TCLdouble raderr;
    TCLint ns;
    TCLint nl;
    TCLint ni;
    TCLint nrmls;
    TCLint lnx;
    TCLint vissamp;
    TCLint ffsamp;
    TCLdouble cscale;
    clString msg;
    Array1<int> geom_idx;
    Array1<int> geom_link_idx;
    HashTable<int, RadObj*> hash;
    RadObj* selRadObj;
    Array1<RadObj* > ancestory;
    int drawLevel;	// -1 -> draw all links; -2 -> draw no links
    Point drawCenter;
    double drawRad;
    double CSCALE;
public:
    Radiosity(const clString& id);
    virtual ~Radiosity();
    virtual void execute();
    void removeGeom();
    void removeLinkGeom();
    void buildTriSurfs();

    void createObjLinks(RadObj* so, RadObj* ro);
    void createInitialLinks();
    void globalIteration();
    void globalSolve(int nIter);

    void triSurfsToRadObjs();
    void refineMeshes(double err);
    void meshToGeom(int);
    void radToGeom(RadObj *ro, GeomGroup *gg);
    void buildGeom();
    void radToLinkGeom(RadObj *ro, GeomGroup *gg);
    void radToLinkGeom2(RadObj *ro, GeomGroup *gg,const Point& rcv,double rad);
    void buildLinkGeom();
    int bldAncestory(RadObj* desc, RadObj* anc);
    void tcl_command( TCLArgs&, void * );
    virtual void geom_pick(GeomPick*, void *);
};

Module* make_Radiosity(const clString& id)
{
    return scinew Radiosity(id);
}

static clString module_name("Radiosity");

Radiosity::Radiosity(const clString& id)
: Module("Radiosity", id, Source), rtGen(0), ns("ns", id, this),
  nl("nl", id, this), ni("ni", id, this), raderr("raderr", id, this),
  cscale("cscale", id, this), nrmls("nrmls", id, this), lnx("lnx", id, this),
  vissamp("vissamp", id, this), ffsamp("ffsamp", id, this)
{
    // Create the input port
    iRT = scinew VoidStarIPort(this, "DRaytracer", VoidStarIPort::Atomic);
    add_iport(iRT);
    // Create the output port
    oRT = scinew VoidStarOPort(this, "DRaytracer", VoidStarIPort::Atomic);
    add_oport(oRT);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    drawLevel=-2;
}

Radiosity::~Radiosity()
{
}

void Radiosity::removeGeom() {
    for (int i=0; i<geom_idx.size(); i++) {
	if (geom_idx[i]) ogeom->delObj(geom_idx[i]);
    }
    geom_idx.resize(0);
}

void Radiosity::removeLinkGeom() {
    for (int i=0; i<geom_link_idx.size(); i++) {
	if (geom_link_idx[i]) ogeom->delObj(geom_link_idx[i]);
    }
    geom_link_idx.resize(0);
}

void Radiosity::radToLinkGeom(RadObj *ro, GeomGroup *gg) {
//    if (ro->children.size()) {
//	for (int i=0; i<ro->children.size(); i++) {
//	    radToLinkGeom(ro->children[i], gg);
//	}
//    }
    RadMesh* rm=ro->mesh;
    Material *gm = new Material(ro->rad*(CSCALE*.8));
    MaterialHandle mh(gm);
    Point mpr(AffineCombination(rm->pts[ro->i1], 1/3.,
				rm->pts[ro->i2], 1/3.,
				rm->pts[ro->i3], 1/3.));

    for (int i=0; i<ro->links.size(); i++) {
	RadLink *l = ro->links[i];
	RadObj *src=l->src;
	RadMesh *sm=src->mesh;
	Point mps(AffineCombination(sm->pts[src->i1], 1/3.,
				    sm->pts[src->i2], 1/3.,
				    sm->pts[src->i3], 1/3.));
	gg->add(new GeomMaterial(new GeomLine(mpr,mps), mh));
    }
}

void Radiosity::radToLinkGeom2(RadObj *ro, GeomGroup *gg, const Point &rcv,
			       double rad) {
    int i;
    double maxRatio=0;
    for (i=0; i<ro->links.size(); i++) {
	RadLink *l = ro->links[i];
	RadObj *src=l->src;
	RadMesh *sm=src->mesh;
	Point mps(AffineCombination(sm->pts[src->i1], 1/3.,
				    sm->pts[src->i2], 1/3.,
				    sm->pts[src->i3], 1/3.));
	Material *gm = new Material(src->rad*(CSCALE*.8));
	MaterialHandle mh(gm);
	gg->add(new GeomMaterial(new GeomLine(rcv,mps), mh));
	double length=(rcv-mps).length();
	double error=l->error();
	if (error/length > maxRatio) maxRatio=error/length;
    }
    maxRatio = 1./(maxRatio*2);
    for (i=0; i<ro->links.size(); i++) {
	RadLink *l = ro->links[i];
	RadObj *src=l->src;
	RadMesh *sm=src->mesh;
	Point mps(AffineCombination(sm->pts[src->i1], 1/3.,
				    sm->pts[src->i2], 1/3.,
				    sm->pts[src->i3], 1/3.));
	Vector v(mps-rcv);
	v.normalize();
	v*=(l->error()*maxRatio);
	mps = rcv+v;
	Material *gm = new Material(src->rad*(CSCALE*.8));
	MaterialHandle mh(gm);
	gg->add(new GeomMaterial(new GeomCappedCone(rcv, mps, rad/3, 0), mh));
    }
    Point p1(ro->mesh->pts[ro->i1]);
    Point p2(ro->mesh->pts[ro->i2]);
    Point p3(ro->mesh->pts[ro->i3]);
    Material *gm = new Material(Color(0.3, 0.3, 0.3));
    MaterialHandle mh1(gm);
    gg->add(new GeomMaterial(new GeomCappedCone(p1, p2, rad/10, rad/10), mh1));
    gg->add(new GeomMaterial(new GeomCappedCone(p2, p3, rad/10, rad/10), mh1));
    gg->add(new GeomMaterial(new GeomCappedCone(p1, p3, rad/10, rad/10), mh1));
}

void Radiosity::buildLinkGeom() {
    removeLinkGeom();
    if (lnx.get()) {
	GeomGroup *gg=scinew GeomGroup();
	if (drawLevel == -2) return;	// don't draw anything
	if (drawLevel == -1) {		// go through all RadObj's and add linx
	    for (int i=0; i<rt->scene.mesh.size(); i++) {
		RadMeshHandle rmh=rt->scene.mesh[i];
		for (int e=0; e<rmh->patches.size(); e++) {
		    Stack<RadObj*> stack;
		    stack.push(rmh->patches[e]);
		    while (!stack.empty()) {
			RadObj *ro=stack.pop();
			for (int j=0; j<ro->children.size(); j++) {
			    stack.push(ro->children[j]);
			}
			radToLinkGeom(ro, gg);
		    }
		}
	    }
	}
	if (drawLevel >= 0) radToLinkGeom2(ancestory[drawLevel],gg,drawCenter,
					   drawRad);
	geom_link_idx.add(ogeom->addObj(gg, "links"));
    }
}

void Radiosity::buildTriSurfs() {
    int NL=nl.get();
    rt->scene.mesh.resize(rt->scene.obj.size());
    for (int i=0; i<rt->scene.obj.size(); i++) {
	RTObjectHandle rto=rt->scene.obj[i];
	RadMesh *rm=scinew RadMesh(rto, rt, NL);
	RadMeshHandle rmh(rm);
	rt->scene.mesh[i]=rmh;
	rto->mesh=rmh;
    }
}

void Radiosity::refineMeshes(double err) {
    int NS=ns.get();
    int NVISSAMP = vissamp.get();
    int NFFSAMP = ffsamp.get();
    for (int i=0; i<rt->scene.mesh.size(); i++) {
	cerr << "\n("<<i<<"/"<<rt->scene.mesh.size()-1<<")... ";
	RadMesh* rm=rt->scene.mesh[i].get_rep();
	for (int v=0; v<rm->patches.size(); v++) {
	    RadObj* rcv=rm->patches[v];
	    rcv->refineAllLinks(err, rt, NVISSAMP, NFFSAMP);
	}
    }
    cerr << "\nDone refining!\n";
}

void Radiosity::createObjLinks(RadObj* so, RadObj* ro) {
    int NVISSAMP = vissamp.get();
    int NFFSAMP = ffsamp.get();
    if (so->children.size()) {
	for (int i=0; i<so->children.size(); i++) {
	    createObjLinks(so->children[i], ro);
	}
    } else if (ro->children.size()) {
	for (int i=0; i<ro->children.size(); i++) {
	    createObjLinks(so, ro->children[i]);
	}
    } else {
	ro->createLink(so, rt, NVISSAMP, NFFSAMP);
    }
}

void Radiosity::createInitialLinks() {
    for (int smi=0; smi<rt->scene.mesh.size(); smi++) {
	RadMesh* sm=rt->scene.mesh[smi].get_rep();
	for (int s=0; s<sm->patches.size(); s++) {
	    RadObj* so=sm->patches[s];
	    // we have the top level sending patches, now get the top level
	    // receivers
	    for (int rmi=0; rmi<rt->scene.mesh.size(); rmi++) {
		// efficiency hack -- assume no patch can see itself!
		if (rmi==smi) continue;
		RadMesh *rm=rt->scene.mesh[rmi].get_rep();
		for (int r=0; r<rm->patches.size(); r++) {
		    RadObj* ro=rm->patches[r];
		    createObjLinks(so, ro);
		}
	    }
	}
    }
}

void Radiosity::globalIteration() {
    int rmi, r;
    for (rmi=0; rmi<rt->scene.mesh.size(); rmi++) {
	RadMesh *rm=rt->scene.mesh[rmi].get_rep();
	for (r=0; r<rm->patches.size(); r++) {
	    RadObj* ro=rm->patches[r];
	    ro->gatherRad();
	}
    }
    for (rmi=0; rmi<rt->scene.mesh.size(); rmi++) {
	RadMesh *rm=rt->scene.mesh[rmi].get_rep();
	for (r=0; r<rm->patches.size(); r++) {
	    RadObj* ro=rm->patches[r];
	    ro->radPushPull(Color(0,0,0));
	}
    }
}

void Radiosity::globalSolve(int nIter) {
    int currIter=1;
    while (currIter<=nIter) {
	globalIteration();
	currIter++;
    }
}

void Radiosity::triSurfsToRadObjs() {
    int NS=ns.get();
    Color c;
    for (int i=0; i<rt->scene.mesh.size(); i++) {
	RadMesh* rm=rt->scene.mesh[i].get_rep();
	rm->pts=rm->ts.points;
	if (rm->emitting) c=rm->emit_coeff;
	else c=Color(0,0,0);
	for (int j=0; j<rm->ts.elements.size(); j++) {
	    TSElement *e=rm->ts.elements[j];
	    Point p1(rm->ts.points[e->i1]);
	    Point p2(rm->ts.points[e->i2]);
	    Point p3(rm->ts.points[e->i3]);
	    double area=Cross(p2-p1,p2-p3).length()/2;
	    rm->patches.add(new RadObj(e->i1, e->i2, e->i3, area, c, rm));
	}
    }
}

void Radiosity::radToGeom(RadObj *ro, GeomGroup *gg) {
    if (ro->children.size()) {
//	cerr << "RadToGeom adding children!\n";
	for (int i=0; i<ro->children.size(); i++) {
	    radToGeom(ro->children[i], gg);
	}
    } else {
	RadMesh* rm=ro->mesh;
	GeomTriangles* gt = new GeomTriangles;
	gt->add(rm->pts[ro->i1], ro->rad*(CSCALE),
		rm->pts[ro->i2], ro->rad*(CSCALE),
		rm->pts[ro->i3], ro->rad*(CSCALE));

	if (nrmls.get()) {
	    Vector v1(rm->pts[ro->i2]-rm->pts[ro->i1]);
	    Vector v2(rm->pts[ro->i2]-rm->pts[ro->i3]);
	    Vector v3(Cross(v1,v2));
	    v3.normalize();
	    v3*=v1.length()/4.;
	    Point p1(AffineCombination(rm->pts[ro->i1],.31,
				       rm->pts[ro->i2],.35,
				       rm->pts[ro->i3],.34));
	    Point p2(AffineCombination(rm->pts[ro->i1],.34,
				       rm->pts[ro->i2],.34,
				       rm->pts[ro->i3],.32));
	    Point p3(p1+v3);
	    gt->add(p1, ro->rad*(CSCALE*.8), p2, ro->rad*(CSCALE*.8), p3, 
		    ro->rad*(CSCALE*.8));
	}
	GeomPick *gp=new GeomPick(gt, this);
	hash.insert((int)gp, ro);
//	cerr << "inserting "<<(int) gp<<"...\n";
	geom_idx.add(ogeom->addObj(gp, rm->obj->name));
    }
}

void Radiosity::meshToGeom(int i) {
    RadMeshHandle rmh=rt->scene.mesh[i];
    for (int e=0; e<rmh->patches.size(); e++) {
	GeomGroup *gg=scinew GeomGroup();
	radToGeom(rmh->patches[e], gg);
    }
}

void Radiosity::buildGeom() {
    removeGeom();
    hash.remove_all();
    for (int i=0; i<rt->scene.mesh.size(); i++) {
	meshToGeom(i);
    }
}

void Radiosity::geom_pick(GeomPick* gp, void*) {
    RadObj* rid;
    if (!hash.lookup((int)gp, rid)) {
	cerr << "Error -- couldn't find that GeomPick "<<(int)gp<<" in hash table!\n";
	cerr << "(hash.size() = "<<hash.size()<<")\n";
	return;
    }
    selRadObj = rid;
    msg = clString("polyselect");
    want_to_execute();
}

int Radiosity::bldAncestory(RadObj* descendant, RadObj* ancestor) {
    if (descendant == ancestor) {
	ancestory.add(ancestor);
	return 1;
    } else {
	for (int i=0; i<ancestor->children.size(); i++) 
	    if (bldAncestory(descendant, ancestor->children[i])) {
		ancestory.add(ancestor);
		return 1;
	    }
    }
    return 0;
}
    
void Radiosity::execute()
{
    CSCALE = cscale.get();

    if (msg == "reset") {
	ancestory.remove_all();
	msg = "";
	rtGen = -1;
    }
    if (msg == "redraw") {
	msg="";
	buildGeom();
	buildLinkGeom();
	ogeom->flushViews();
	return;
    }
    if (msg == "uplevel") {
	msg="";
	if (drawLevel+1 < ancestory.size()) drawLevel++;
	buildGeom();
	buildLinkGeom();
	ogeom->flushViews();
	return;
    }
    if (msg == "downlevel") {
	msg="";
	if (drawLevel > 0) drawLevel--;
	buildGeom();
	buildLinkGeom();
	ogeom->flushViews();
	return;
    }
    if (msg == "polyselect") {
	msg = "";
	if (ancestory.size() && selRadObj == ancestory[0])
	    return;
	ancestory.remove_all();
	RadMeshHandle rmh=selRadObj->mesh;
	int e;
	for (e=0; e<rmh->patches.size(); e++)
	    if (rmh->patches[e]->ancestorOf(selRadObj)) break;
	if (e == rmh->patches.size()) {
	    cerr << "ERROR -- couldn't find ancestor!\n";
	    return;
	}
	bldAncestory(selRadObj, rmh->patches[e]);
	drawLevel = 0;
	Point p1(selRadObj->mesh->pts[selRadObj->i1]);
	Point p2(selRadObj->mesh->pts[selRadObj->i2]);
	Point p3(selRadObj->mesh->pts[selRadObj->i3]);
	drawCenter=AffineCombination(p1, 1./3, p2, 1./3, p3, 1./3);
	double d1=(p2-p1).length();
	double d2=(p3-p1).length();
	double d3=(p2-p3).length();
	drawRad=Max(d1,d2,d3)/10;
	buildLinkGeom();
	ogeom->flushViews();
	return;
    }
    if (msg == "removelinks") {
	msg = "";
	drawLevel = -2;
	buildLinkGeom();
	ogeom->flushViews();
	return;
    }
    if (msg == "addlinks") {
	msg = "";
	drawLevel = -1;
	buildLinkGeom();
	ogeom->flushViews();
	return;
    }
    if (msg == "checkff") {
	msg = "";
	for (int i=0; i<rt->scene.mesh.size(); i++) {
	    RadMeshHandle rmh=rt->scene.mesh[i];
	    for (int e=0; e<rmh->patches.size(); e++) {
		cerr << rmh->obj->name<<e<<" FF = "<<rmh->patches[e]->allFF()<<"\n";
	    }
	}
	return;
    }
    VoidStarHandle RTHandle;
    iRT->get(RTHandle);
    if (!RTHandle.get_rep()) return;
    if (!(rt = dynamic_cast<DRaytracer *>(RTHandle.get_rep()))) return;

    if (rtGen != rt->generation) {
	rt->preRayTrace();
	buildTriSurfs();
	triSurfsToRadObjs();

	cerr << "Creating initial links...\n";
	createInitialLinks();
    }	
    double err=raderr.get();
    int nSolveIter=ns.get();
    int nRefineIter=ni.get();
    
    cerr << "Solving...\n";
    int refineIter=1;
    while(refineIter<=nRefineIter) {
	cerr << "Global solve...\n";
	globalSolve(nSolveIter);
	cerr << "Refining...\n";
	refineMeshes(err);
	refineIter++;
	buildGeom();
	buildLinkGeom();
	ogeom->flushViews();
    }
    err/=2;
    raderr.set(err);
    rtGen=rt->generation;
    RTHandle=rt;
    oRT->send(RTHandle);
    return;
}

void Radiosity::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "reset") {
	msg = "reset";
	want_to_execute();
    } else if (args[1] == "remove") {
	msg = "removelinks";
	want_to_execute();
    } else if (args[1] == "redraw") {
	msg = "redraw";
	want_to_execute();
    } else if (args[1] == "add") {
	msg = "addlinks";
	want_to_execute();
    } else if (args[1] == "checkff") {
	msg = "checkff";
	want_to_execute();
    } else if (args[1] == "uplevel") {
	msg = "uplevel";
	want_to_execute();
    } else if (args[1] == "downlevel") {
	msg = "downlevel";
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}
} // End namespace DaveW
} // End namespace Uintah

//
// $Log
//
//
