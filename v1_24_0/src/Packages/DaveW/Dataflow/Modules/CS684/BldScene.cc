
/*
 *  BldScene.cc:  Take in a scene (through a VoidStarPort), and output 
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

#include <Packages/DaveW/Core/Datatypes/CS684/DRaytracer.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/VoidStarPort.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Dataflow/Widgets/RingWidget.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/VoidStar.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Tester/RigorousTest.h>

#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class BldScene : public Module {
    VoidStarIPort *iRT;
    VoidStarOPort *oRT;
    GeometryOPort *ogeom;
    DRaytracer *rt;
    int rtGen;

    Array1<int> light_geom_id;
    Array1<int> light_widget_id;
    Array1<RingWidget*> light_widget;
    Array1<int> sphere_geom_id;
    Array1<int> sphere_widget_id;
    Array1<RingWidget*> sphere_widget;
    Array1<int> sphere_idx;
    Array1<int> plane_geom_id;
    Array1<int> plane_widget_id;
    Array1<FrameWidget*> plane_widget;
    Array1<int> plane_idx;
    Array1<int> box_geom_id;
    Array1<int> box_widget_id;
    Array1<BoxWidget*> box_widget;
    Array1<int> box_idx;
    Array1<int> rect_geom_id;
    Array1<int> rect_widget_id;
    Array1<FrameWidget*> rect_widget;
    Array1<int> rect_idx;
    Array1<int> tris_geom_id;
    Array1<int> tris_widget_id;
    Array1<BoxWidget*> tris_widget;
    Array1<int> tris_idx;
    Array1<int> global_idx;

    CrowdMonitor widget_lock;

    GuiMaterial material;
    GuiInt nb;
    GuiInt atten;

    char widgetType;	// callback info
    int widgetNum;	// callback info
    int globalNum;
    int matlChanged;
    int widgetMoved;
    int tcl_exec;
    int init;

public:
    BldScene(const clString& id);
    virtual ~BldScene();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
    void removeWidgets();
    void removeGeom();
    void removeIdx();
    void buildWidgets();
    int buildGeomObj(int);
    int buildGeomLight(int);
    void buildGeom();
    virtual void widget_moved2(int last, void *userdata);    
};

extern "C" Module* make_BldScene(const clString& id)
{
    return scinew BldScene(id);
}

static clString module_name("BldScene");

BldScene::BldScene(const clString& id)
: Module("BldScene", id, Source),
  atten("atten", id, this), material("material", id, this), nb("nb", id, this),
  tcl_exec(0), widgetMoved(0), matlChanged(0), widget_lock("BldScene widget lock")
{
    // Create the input port
    iRT = scinew VoidStarIPort(this, "DRaytracer", VoidStarIPort::Atomic);
    add_iport(iRT);
    // Create the output port
    oRT = scinew VoidStarOPort(this, "DRaytracer", VoidStarIPort::Atomic);
    add_oport(oRT);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    init=0;
}

BldScene::~BldScene()
{
}

void BldScene::removeWidgets() {
    int i;
    for (i=0; i<sphere_widget_id.size(); i++) {
	if (sphere_widget_id[i]) ogeom->delObj(sphere_widget_id[i]);
	sphere_widget_id[i]=0;
	sphere_widget[i]=0;
    }
    for (i=0; i<plane_widget_id.size(); i++) {
	if (plane_widget_id[i]) ogeom->delObj(plane_widget_id[i]);
	plane_widget_id[i]=0;
	plane_widget[i]=0;
    }
    for (i=0; i<box_widget_id.size(); i++) {
	if (box_widget_id[i]) ogeom->delObj(box_widget_id[i]);
	box_widget_id[i]=0;
	box_widget[i]=0;
    }
    for (i=0; i<rect_widget_id.size(); i++) {
	if (rect_widget_id[i]) ogeom->delObj(rect_widget_id[i]);
	rect_widget_id[i]=0;
	rect_widget[i]=0;
    }
    for (i=0; i<tris_widget_id.size(); i++) {
	if (tris_widget_id[i]) ogeom->delObj(tris_widget_id[i]);
	tris_widget_id[i]=0;
	tris_widget[i]=0;
    }
    for (i=0; i<light_widget_id.size(); i++) {
	if (light_widget_id[i]) ogeom->delObj(light_widget_id[i]);
	light_widget_id[i]=0;
    }
    sphere_widget_id.resize(0);
    plane_widget_id.resize(0);
    box_widget_id.resize(0);
    rect_widget_id.resize(0);
    tris_widget_id.resize(0);
    light_widget_id.resize(0);
    sphere_widget.resize(0);
    plane_widget.resize(0);
    box_widget.resize(0);
}

void BldScene::removeGeom() {
    int i;
    for (i=0; i<sphere_geom_id.size(); i++) {
	if (sphere_geom_id[i]) ogeom->delObj(sphere_geom_id[i]);
	sphere_geom_id[i]=0;
    }
    for (i=0; i<plane_geom_id.size(); i++) {
	if (plane_geom_id[i]) ogeom->delObj(plane_geom_id[i]);
	plane_geom_id[i]=0;
    }
    for (i=0; i<box_geom_id.size(); i++) {
	if (box_geom_id[i]) ogeom->delObj(box_geom_id[i]);
	box_geom_id[i]=0;
    }
    for (i=0; i<rect_geom_id.size(); i++) {
	if (rect_geom_id[i]) ogeom->delObj(rect_geom_id[i]);
	rect_geom_id[i]=0;
    }
    for (i=0; i<tris_geom_id.size(); i++) {
	if (tris_geom_id[i]) ogeom->delObj(tris_geom_id[i]);
	tris_geom_id[i]=0;
    }
    for (i=0; i<light_geom_id.size(); i++) {
	if (light_geom_id[i]) ogeom->delObj(light_geom_id[i]);
	light_geom_id[i]=0;
    }
    sphere_geom_id.resize(0);
    plane_geom_id.resize(0);
    box_geom_id.resize(0);
    rect_geom_id.resize(0);
    tris_geom_id.resize(0);
    light_geom_id.resize(0);
}

void BldScene::removeIdx() {
    sphere_idx.resize(0);
    plane_idx.resize(0);
    box_idx.resize(0);
    rect_idx.resize(0);
    tris_idx.resize(0);
    global_idx.resize(0);
}

void BldScene::buildWidgets() {
    clString objName;
    int i;
    for (i=0; i<rt->scene.obj.size(); i++) {
	char *ud=new char[20];
	RTObjectHandle rto=rt->scene.obj[i];
	RTSphere* rts=rto->getSphere();
	RTBox* rtb=rto->getBox();
	RTRect* rtr=rto->getRect();
	RTTris* rtt=rto->getTris();
	RTPlane* rtp=rto->getPlane();
	GeomObj *w;
	if (rts) {
	    name="Sphere Widget "+to_string(i);
	    sprintf(ud, "s%d", sphere_idx.size());
	    global_idx.add(sphere_idx.size());
	    RingWidget* r=scinew
		RingWidget(this, &widget_lock, 0.2);
	    r->SetCurrentMode(2);
	    r->SetPosition(rts->center, Vector(0,0,1), rts->radius);
	    r->userdata=(void *) ud;
	    w=r->GetWidget();
	    int idx=ogeom->addObj(w, name, &widget_lock);
	    sphere_widget_id.add(idx);
	    sphere_widget.add(r);
	    sphere_idx.add(i);
	}
	if (rtb) {
	    name="Box Widget "+to_string(i);
	    sprintf(ud, "b%d", box_idx.size());
	    global_idx.add(box_idx.size());
	    BoxWidget* b=scinew
		BoxWidget(this, &widget_lock, 0.1, false, true);
	    b->SetCurrentMode(1);
	    b->SetPosition(rtb->center, 
			   rtb->center+Vector(rtb->d.x(),0,0),
			   rtb->center+Vector(0,-rtb->d.y(),0),
			   rtb->center+Vector(0,0,-rtb->d.z()));
	    b->userdata=(void *) ud;
	    w=b->GetWidget();
	    int idx=ogeom->addObj(w, name, &widget_lock);
	    box_widget_id.add(idx);
	    box_widget.add(b);
	    box_idx.add(i);
	}
	if (rtr) {
	    name="Rect Widget "+to_string(i);
	    sprintf(ud, "r%d", rect_idx.size());
	    global_idx.add(rect_idx.size());
	    FrameWidget* f=scinew
		FrameWidget(this, &widget_lock, 0.1);
	    f->SetCurrentMode(2);
	    f->SetPosition(rtr->c, rtr->c+rtr->v1, rtr->c+rtr->v2);
	    f->userdata=(void *) ud;
	    w=f->GetWidget();
	    int idx=ogeom->addObj(w, name, &widget_lock);
	    rect_widget_id.add(idx);
	    rect_widget.add(f);
	    rect_idx.add(i);
	}
	if (rtt) {
	    name="Tris Widget "+to_string(i);
	    sprintf(ud, "t%d", tris_idx.size());
	    global_idx.add(tris_idx.size());
	    BoxWidget* b=scinew
		BoxWidget(this, &widget_lock, 0.1, false, true);
	    b->SetCurrentMode(3);
	    Vector d(rtt->bb.max()-rtt->bb.center());
	    b->SetPosition(rtt->bb.center(), 
			   rtt->bb.center()+Vector(d.x(),0,0),
			   rtt->bb.center()-Vector(0,d.y(),0),
			   rtt->bb.center()-Vector(0,0,d.z()));
	    b->userdata=(void *) ud;
	    w=b->GetWidget();
	    int idx=ogeom->addObj(w, name, &widget_lock);
	    tris_widget_id.add(idx);
	    tris_widget.add(b);
	    tris_idx.add(i);
	}
	if (rtp) {
	    name="Plane Widget "+to_string(i);
	    sprintf(ud, "p%d", plane_idx.size());
	    global_idx.add(plane_idx.size());
	    FrameWidget* f=scinew
		FrameWidget(this, &widget_lock, 0.1);
	    f->SetCurrentMode(2);
	    
	    // find closest axis penetration -- make that the center
	    RTHit h; RTRay r;
	    r.dir=Vector(1,0,0); rtp->intersect(r,h);
	    r.dir=Vector(0,1,0); rtp->intersect(r,h);
	    r.dir=Vector(0,0,1); rtp->intersect(r,h);
	    r.dir=Vector(-1,0,0); rtp->intersect(r,h);
	    r.dir=Vector(0,-1,0); rtp->intersect(r,h);
	    r.dir=Vector(0,0,-1); rtp->intersect(r,h);
	    
	    f->SetPosition(h.p, rtp->n, 1, 1);
	    f->userdata=(void *) ud;
	    w=f->GetWidget();
	    int idx=ogeom->addObj(w, name, &widget_lock);
	    plane_widget_id.add(idx);
	    plane_widget.add(f);
	    plane_idx.add(i);
	}
    }

    for (i=0; i<rt->scene.light.size(); i++) {
	char *ud=new char[20];
	sprintf(ud, "l%d", i);
	RingWidget* r=scinew
	    RingWidget(this, &widget_lock, 0.1);
	r->SetCurrentMode(2);
	r->SetPosition(rt->scene.light[i].pos, Vector(1,0,0), .3);
	r->userdata=(void *) ud;
	GeomObj *w=r->GetWidget();
	name="Light Widget "+to_string(i);
	int idx=ogeom->addObj(w, name, &widget_lock);
	light_widget_id.add(idx);
	light_widget.add(r);
    }
}

int BldScene::buildGeomObj(int i) {
    if (rt->scene.obj[i]->visible == 0) return 0;
    RTObjectHandle rto=rt->scene.obj[i];
    RTSphere* rts=rto->getSphere();
    RTBox* rtb=rto->getBox();
    RTRect* rtr=rto->getRect();
    RTTris* rtt=rto->getTris();
    RTPlane* rtp=rto->getPlane();
    GeomMaterial *m;
    clString name;
    if (rts) {
cerr << "Radius="<<rts->radius<<"\n";	
	GeomSphere* s=scinew GeomSphere(rts->center, rts->radius);
	m=scinew GeomMaterial(s, rts->matl->base);
	name="Sphere"+to_string(i);
    }
    if (rtb) {
	GeomTriangles* t=scinew GeomTriangles();
	Array1<Point> p(8);
	Point c(rtb->center);
	p[0]=c-rtb->d;
	p[7]=c+rtb->d;
	p[1]=p[0]+Vector(rtb->d.x()*2,0,0);
	p[2]=p[0]+Vector(0,rtb->d.y()*2,0);
	p[3]=p[0]+Vector(rtb->d.x()*2,rtb->d.y()*2,0);
	p[4]=p[0]+Vector(0,0,2*rtb->d.z());
	p[5]=p[1]+Vector(0,0,2*rtb->d.z());
	p[6]=p[2]+Vector(0,0,2*rtb->d.z());
	t->add(p[0],p[1],p[2]);
	t->add(p[1],p[2],p[3]);
	t->add(p[4],p[5],p[6]);
	t->add(p[5],p[6],p[7]);
	t->add(p[0],p[1],p[4]);
	t->add(p[1],p[5],p[4]);
	t->add(p[2],p[3],p[6]);
	t->add(p[3],p[7],p[6]);
	t->add(p[0],p[2],p[4]);
	t->add(p[2],p[4],p[6]);
	t->add(p[1],p[3],p[5]);
	t->add(p[3],p[5],p[7]);
	m=scinew GeomMaterial(t, rtb->matl->base);
	name="Box"+to_string(i);
    }
    if (rtr) {
	GeomTriangles* t=scinew GeomTriangles();
	t->add(rtr->c+rtr->v1, rtr->c+rtr->v2, rtr->c-rtr->v1);
	t->add(rtr->c+rtr->v1, rtr->c-rtr->v1, rtr->c-rtr->v2);
	m=scinew GeomMaterial(t, rtr->matl->base);
	name="Rect"+to_string(i);
    }
    if (rtt) {
	GeomTriangles* t=scinew GeomTriangles();
	TriSurfFieldace *ts=rtt->surf->getTriSurfFieldace();
	for (int j=0; j<ts->elements.size(); j++) {
	    TSElement* e=ts->elements[j];
	    t->add(ts->points[e->i1], ts->points[e->i2],
		   ts->points[e->i3]);
	}
	m=scinew GeomMaterial(t, rtt->matl->base);
	name="Tris"+to_string(i);
    }
    if (rtp) {
	GeomTriangles* t=scinew GeomTriangles();
	Point p, R, D;
	plane_widget[global_idx[i]]->GetPosition(p,R,D);
	Vector u(R-p);
	Vector v(p-D);
	t->add(p+(v+u)*10, p+(v-u)*10, p+(-v-u)*10);
	t->add(p+(v+u)*10, p+(-v-u)*10, p+(-v+u)*10);
	m=scinew GeomMaterial(t, rtp->matl->base);
	name="Plane"+to_string(i);
    }
    return(ogeom->addObj(m, name, &widget_lock));
}

int BldScene::buildGeomLight(int i) {
    if (rt->scene.light[i].visible == 0) return 0;
    GeomSphere* s=scinew GeomSphere(rt->scene.light[i].pos, .3);
    Material* matl=scinew Material(rt->scene.light[i].color);
    GeomMaterial *m=scinew GeomMaterial(s, MaterialHandle(matl));
    clString name="Light"+to_string(i);
    return(ogeom->addObj(m, name, &widget_lock));
}

void BldScene::buildGeom() {
    if (widgetMoved || matlChanged) {
	if (widgetType == 'l') {	// light
	    ogeom->delObj(light_geom_id[widgetNum]);
	    Point p;
	    Vector n;
	    double r;
	    light_widget[widgetNum]->GetPosition(p,n,r);
	    rt->scene.light[widgetNum].pos=p;
	    light_geom_id[widgetNum]=buildGeomLight(widgetNum);
	} else if (widgetType == 's') {	// sphere
	    ogeom->delObj(sphere_geom_id[widgetNum]);
	    Point p;
	    Vector n;
	    double r;
	    sphere_widget[widgetNum]->GetPosition(p,n,r);
	    RTSphere* rts=rt->scene.obj[globalNum]->getSphere();
	    rts->center=p;
//	    rts->radius=r;     ring widget doesn't seem to deal w/ radius
	    sphere_geom_id[widgetNum]=buildGeomObj(globalNum);
	} else if (widgetType == 'b') {	// box
	    ogeom->delObj(box_geom_id[widgetNum]);
	    Point c, R, U, I;
	    box_widget[widgetNum]->GetPosition(c,R,U,I);
	    RTBox* rtb=rt->scene.obj[globalNum]->getBox();
	    rtb->center=c;
	    rtb->d=(c*3)-(R-(-U-I));
	    rtb->d.x(rtb->d.x()*-1);
	    box_geom_id[widgetNum]=buildGeomObj(globalNum);
	} else if (widgetType == 'r') {	// rect
	    ogeom->delObj(rect_geom_id[widgetNum]);
	    Point c, R, U;
	    rect_widget[widgetNum]->GetPosition(c,R,U);
	    RTRect* rtr=rt->scene.obj[globalNum]->getRect();
	    rtr->c=c;
	    rtr->v1=R-c;
	    rtr->v2=U-c;
	    rect_geom_id[widgetNum]=buildGeomObj(globalNum);
	} else if (widgetType == 't') {	// tris
	    ogeom->delObj(tris_geom_id[widgetNum]);
	    Point c, R, U, I;
	    tris_widget[widgetNum]->GetPosition(c,R,U,I);
	    RTTris* rtt=rt->scene.obj[globalNum]->getTris();
	    Vector trans(c-rtt->bb.center());
//cerr << "widget center is: "<<c<<"  obj scenter is: "<<rtt->bb.center()<<"  translating all obj points by: "<<trans<<"\n";
	    rtt->bb.translate(trans);
	    TriSurfFieldace *ts=rtt->surf->getTriSurfFieldace();
	    for (int i=0; i<ts->points.size(); i++) {
		ts->points[i] = ts->points[i]+trans;
	    }
	    tris_geom_id[widgetNum]=buildGeomObj(globalNum);
	} else if (widgetType == 'p') {	// plane
	    ogeom->delObj(plane_geom_id[widgetNum]);
	    double dum1, dum2;
	    Vector n;
	    Point p;
	    plane_widget[widgetNum]->GetPosition(p,n,dum1,dum2);
	    RTPlane* rtp=rt->scene.obj[globalNum]->getPlane();
	    rtp->n=n;
	    rtp->d=-(Dot(n,p));
	    plane_geom_id[widgetNum]=buildGeomObj(globalNum);
	} else {
	    cerr << "Error: can't match widgetType "<<widgetType<< "\n";
	}
	return;
    }
    removeGeom();
    int i;
    for (i=0; i<sphere_widget_id.size(); i++) {
	sphere_geom_id.add(buildGeomObj(sphere_idx[i]));
    }
    for (i=0; i<box_widget_id.size(); i++) {
	box_geom_id.add(buildGeomObj(box_idx[i]));
    }
    for (i=0; i<rect_widget_id.size(); i++) {
	rect_geom_id.add(buildGeomObj(rect_idx[i]));
    }
    for (i=0; i<tris_widget_id.size(); i++) {
	tris_geom_id.add(buildGeomObj(tris_idx[i]));
    }
    for (i=0; i<plane_widget_id.size(); i++) {
	plane_geom_id.add(buildGeomObj(plane_idx[i]));
    }
    for (i=0; i<light_widget_id.size(); i++) {
	light_geom_id.add(buildGeomLight(i));
    }
}
	
void BldScene::execute()
{
    if (!init) {
	rtGen=0;
	init=1;
    }	
    VoidStarHandle RTHandle;
    iRT->get(RTHandle);
    if (!RTHandle.get_rep() || !(rt=dynamic_cast<DRaytracer* >(RTHandle.get_rep()))) return;

//    cerr << "Number of objects: (rt->scene.obj.size()) "<<rt->scene.obj.size()<<"\n";

    // OK.  Everything's valid -- time to do some work...

    if (rtGen != rt->generation) {
	nb.set(rt->scene.numBounces);
	atten.set(rt->scene.attenuateFlag);
    }

    // New scene -- have to build widgets
    if (rtGen != rt->generation) {
	removeIdx();
	removeWidgets();
	buildWidgets();
    }

    // Scene has changed -- must build new Geometry objects
    if (widgetMoved || (rtGen != rt->generation) || matlChanged) {
	buildGeom();
	ogeom->flushViews();
    }

    // Only ~really~ raytrace if tcl said to
    if (tcl_exec) {
	rt->scene.numBounces=nb.get();
	rt->scene.attenuateFlag=atten.get();
    }

    // reset all flags
    rtGen=rt->generation; widgetMoved=0; matlChanged=0;
    tcl_exec=0;
    RTHandle=rt;
    oRT->send(RTHandle);
    return;
}

void BldScene::widget_moved2(int last, void *userdata)
{
    if (last && !abort_flag) {
	abort_flag=1;
	char *name=(char *) userdata;
	widgetMoved=1;
	widgetType=name[0];
	widgetNum=atoi(&(name[1]));
	if (widgetType == 'l') {
	    globalNum=widgetNum;
	} else if (widgetType == 's') {
	    globalNum=sphere_idx[widgetNum];
	} else if (widgetType == 'b') {
	    globalNum=box_idx[widgetNum];
	} else if (widgetType == 'r') {
	    globalNum=rect_idx[widgetNum];
	} else if (widgetType == 't') {
	    globalNum=tris_idx[widgetNum];
	} else if (widgetType == 'p') {
	    globalNum=plane_idx[widgetNum];
	} else {
	    cerr << "Unknown type: "<<widgetType<<"\n";
	}
	want_to_execute();
    }
}

void BldScene::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "tcl_exec") {
	tcl_exec=1;
	want_to_execute();
    } else if (args[1] == "setmatl") {
	reset_vars();	// need instant update since we're gonna copy it
	if (widgetType == 'l') {
	    Material m(material.get());
	    rt->scene.light[globalNum].color=m.diffuse;
	} else {
	    rt->scene.obj[globalNum]->matl->base=scinew 
		Material(material.get());
	}
	matlChanged=1;
	want_to_execute();
	return;
    } else if (args[1] == "getmatl") {
	if (widgetType == 'l') {
	    Material m(rt->scene.light[globalNum].color);
	    material.set(m);
	} else {
	    material.set(*(rt->scene.obj[globalNum]->matl->base.get_rep()));
	}
	reset_vars();	// want instant update so we have the right edit color
	return;
    } else if (args[1] == "toggle") {
	if (widgetType == 'l') {
	    if (rt->scene.light[globalNum].visible == 1)
		rt->scene.light[globalNum].visible=0;
	    else
		rt->scene.light[globalNum].visible=1;
	} else {
	    if (rt->scene.obj[globalNum]->visible == 1)
		rt->scene.obj[globalNum]->visible=0;
	    else
		rt->scene.obj[globalNum]->visible=1;
	}
	matlChanged=1;
	want_to_execute();
    } else if (args[1] == "widgetMoved") {
	reset_vars();
        widgetMoved=1;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
} // End namespace DaveW
}
