/*
 *  Salmon.cc:  The Geometry Viewer Window
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"
#include <Geom.h>
#include <Salmon/Salmon.h>
#include <Salmon/Roe.h>
#include <MotifCallback.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <Mt/Form.h>
#include <Mt/Frame.h>
#include <Mt/GLwMDraw.h>
#include <Mt/RowColumn.h>
#include <Mt/Label.h>
#include <Mt/ScrolledWindow.h>
#include <Mt/ToggleButton.h>
#include <Mt/PushButton.h>
#include <Mt/Separator.h>
#include <iostream.h>
#include <Geometry/Vector.h>
#include <GL/glu.h>
#include <CallbackCloners.h>
#include <Math/MiscMath.h>
#include <Geometry/BBox.h>
extern MtXEventLoop* evl;

GeomItem::GeomItem() {
}

GeomItem::~GeomItem() {
    delete btn;
}

Roe::Roe(Salmon* s, double *m) {
    haveInheritMat=1;
    for (int i=0; i<16; i++)
	inheritMat[i]=m[i];
    RoeInit(s);
}

Roe::Roe(Salmon* s) {
    haveInheritMat=0;
    RoeInit(s);
}

void Roe::RoeInit(Salmon* s) {
    evl->lock();
    doneInit=0;
    manager=s;
    firstGen=False;
    dialog=new DialogShellC;
    dialog->SetAllowShellResize(true);
    dialog->SetDeleteResponse(XmDESTROY);
    new MotifCallback<Roe>FIXCB(dialog, XmNdestroyCallback,
				   &manager->mailbox, this,
				   &Roe::destroyWidgetCB, 0, 0);    
    dialog->SetWidth(600);
    dialog->SetHeight(400);
    dialog->Create("sci", "sci", evl->get_display());

    gr_frame=new FrameC;
    gr_frame->SetShadowType(XmSHADOW_IN);
    gr_frame->Create(*dialog, "frame");

    wholeWin=new RowColumnC;
    wholeWin->SetOrientation(XmHORIZONTAL);
    wholeWin->Create(*gr_frame, "wholeWin");

    left=new RowColumnC;
    left->SetOrientation(XmVERTICAL);
    left->Create(*wholeWin, "left");

    right=new RowColumnC;
    right->SetOrientation(XmVERTICAL);
    right->Create(*wholeWin, "right");

    graphics=new GLwMDrawC;
    graphics->SetWidth(400);
    graphics->SetHeight(300);
    graphics->SetRgba(True);
    graphics->SetDoublebuffer(True);
    new MotifCallback<Roe>FIXCB(graphics, GLwNexposeCallback,
				&manager->mailbox, this,
				&Roe::redrawCB,
				0, 0);
    new MotifCallback<Roe>FIXCB(graphics, GLwNginitCallback,
				&manager->mailbox, this,
				&Roe::initCB,
				0, 0);
    graphics->Create(*left, "opengl_viewer");

    new MotifCallback<Roe>FIXCB(graphics, "<Btn1Up>", 
				&manager->mailbox, this,
				&Roe::btn1upCB, 0, 
				&CallbackCloners::event_clone);
    new MotifCallback<Roe>FIXCB(graphics, "<Btn1Down>",
				&manager->mailbox, this,
				&Roe::btn1downCB, 0, 
				&CallbackCloners::event_clone);
    new MotifCallback<Roe>FIXCB(graphics, "<Btn1Motion>",
				&manager->mailbox, this,
				&Roe::btn1motionCB, 0, 
				&CallbackCloners::event_clone);
    new MotifCallback<Roe>FIXCB(graphics, "<Btn2Up>",
				&manager->mailbox, this,
				&Roe::btn2upCB, 0, 
				&CallbackCloners::event_clone);
    new MotifCallback<Roe>FIXCB(graphics, "<Btn2Down>",
				&manager->mailbox, this,
				&Roe::btn2downCB, 0, 
				&CallbackCloners::event_clone);
    new MotifCallback<Roe>FIXCB(graphics, "<Btn2Motion>",
				&manager->mailbox, this,
				&Roe::btn2motionCB, 0, 
				&CallbackCloners::event_clone);

    controls=new RowColumnC;
    controls->SetOrientation(XmHORIZONTAL);
    controls->Create(*left, "controls");
    
    objBox=new RowColumnC;
    objBox->SetOrientation(XmVERTICAL);
    objBox->Create(*right, "objBox");
    
    objLabel=new LabelC;
    objLabel->Create(*objBox, "Objects");
    objSep=new SeparatorC;
    objSep->Create(*objBox, "objSep");

    objScroll=new ScrolledWindowC;
    objScroll->SetScrollingPolicy(XmAUTOMATIC);
    objScroll->Create(*right, "objects");

    objRC=new RowColumnC;
    objRC->SetOrientation(XmVERTICAL);
    objRC->Create(*objScroll, "objRC");

    shadeBox=new RowColumnC;
    shadeBox->SetOrientation(XmVERTICAL);
    shadeBox->Create(*right, "shadeBox");
    
    shadeLabel=new LabelC;
    shadeLabel->Create(*shadeBox, "Shading");
    shadeSep=new SeparatorC;
    shadeSep->Create(*shadeBox, "objSep");

    shadeRC=new RowColumnC;
    shadeRC->SetOrientation(XmVERTICAL);
    shadeRC->SetRadioAlwaysOne(True);
    shadeRC->SetRadioBehavior(True);
    shadeRC->Create(*shadeBox, "shadeRC");
    wire=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(wire, XmNvalueChangedCallback,
				&manager->mailbox, this,
				&Roe::wireCB,
				0, 0);
    wire->Create(*shadeRC, "Wire");
    flat=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(flat, XmNvalueChangedCallback,
				&manager->mailbox, this,
				&Roe::flatCB,
				0, 0);
    flat->Create(*shadeRC, "Flat");
    gouraud=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(gouraud, XmNvalueChangedCallback,
				&manager->mailbox, this,
				&Roe::gouraudCB,
				0, 0);
    gouraud->SetSet(True);
    gouraud->Create(*shadeRC, "Gouraud");
    phong=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(phong, XmNvalueChangedCallback,
			&manager->mailbox, this,
				&Roe::phongCB,
				0, 0);
    phong->Create(*shadeRC, "Phong");

    lightBox=new RowColumnC;
    lightBox->SetOrientation(XmVERTICAL);
    lightBox->Create(*right, "objBox");
    
    lightLabel=new LabelC;
    lightLabel->Create(*lightBox, "Lighting");
    lightSep=new SeparatorC;
    lightSep->Create(*lightBox, "lightSep");

    lightScroll=new ScrolledWindowC;
    lightScroll->SetScrollingPolicy(XmAUTOMATIC);
    lightScroll->Create(*lightBox, "lightScroll");

    lightRC=new RowColumnC;
    lightRC->SetOrientation(XmVERTICAL);
    lightRC->Create(*lightScroll, "lightRC");

    ambient=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(ambient, XmNvalueChangedCallback,
				&manager->mailbox, this,
				&Roe::ambientCB,
				0, 0);
    ambient->SetSet(True);
    ambient->Create(*lightRC, "Ambient");
    point1=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(point1, XmNvalueChangedCallback,
				&manager->mailbox, this,
				&Roe::point1CB,
				0, 0);
    point1->SetSet(True);
    point1->Create(*lightRC, "Point1");
    options=new RowColumnC;
    options->SetOrientation(XmHORIZONTAL);
    options->Create(*left, "options");

    viewRC=new RowColumnC;
    viewRC->SetOrientation(XmVERTICAL);
    viewRC->Create(*options, "view");

    autoView=new PushButtonC;
    new MotifCallback<Roe>FIXCB(autoView, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::autoViewCB,
				0, 0);
    autoView->Create(*viewRC, "Autoview");
    setHome=new PushButtonC;
    new MotifCallback<Roe>FIXCB(setHome, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::setHomeCB,
				0, 0);
    setHome->Create(*viewRC, "Set Home View");
    goHome=new PushButtonC;
    new MotifCallback<Roe>FIXCB(goHome, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::goHomeCB,
				0, 0);
    goHome->Create(*viewRC, "Go Home");


    buttons=new DrawingAreaC;
    buttons->SetWidth(200);
    buttons->SetResizePolicy(XmRESIZE_GROW);
    buttons->Create(*options, "buttons");
    
    spawnRC=new RowColumnC;
    spawnRC->SetOrientation(XmVERTICAL);
    spawnRC->Create(*options, "spawn");

    spawnCh=new PushButtonC;
    new MotifCallback<Roe>FIXCB(spawnCh, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::spawnChCB,
				0, 0);
    spawnCh->Create(*spawnRC, "Spawn Child");
    spawnInd=new PushButtonC;
    new MotifCallback<Salmon>FIXCB(spawnInd, XmNactivateCallback,
				&manager->mailbox, manager,
				&Salmon::spawnIndCB,
				0, 0);
    spawnInd->Create(*spawnRC, "Spawn Independent");
    
    evl->unlock();
}

void Roe::redrawCB(CallbackData*, void*){
    if(!doneInit)
	initCB(0, 0);
    redrawAll();
}

void Roe::initCB(CallbackData*, void*) {
    XVisualInfo* vi;
    graphics->GetVisualInfo(&vi);
    graphics->GetValues();
    // Create a GLX context
    evl->lock();
    cx = glXCreateContext(XtDisplay(*graphics), vi, 0, GL_TRUE);

    make_current();

    // set the view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90, 1.33, 1, 10);
    glMatrixMode(GL_MODELVIEW);
    if (haveInheritMat) {
	glLoadMatrixd(inheritMat);
    } else {
	glLoadIdentity();
	gluLookAt(2,2,5,2,2,2,0,1,0);
	glGetDoublev(GL_MODELVIEW_MATRIX, inheritMat);
    }

    GLfloat light_position[] = {3,3,100,1};
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
//    glDepthFunc(GL_ALWAYS);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    evl->unlock();
    doneInit=1;
}

void Roe::make_current() {
    evl->lock();
    GLwDrawingAreaMakeCurrent(*graphics, cx);
    evl->unlock();
}

void Roe::itemAdded(GeomObj *g, char *name) {
    GeomItem *item;
    item= new GeomItem;
    ToggleButtonC *bttn;
    bttn = new ToggleButtonC;
    item->btn=bttn;
    item->vis=1;
    item->geom=g;
    geomItemA.add(item);
    new MotifCallback<Roe>FIXCB(item->btn, XmNvalueChangedCallback,
				&manager->mailbox, this,
				&Roe::itemCB,
				(void *) item, 0);
    item->btn->SetSet(True);
    item->btn->Create(*objRC, name);
    for (int i=0; i<kids.size(); i++) {
	kids[i]->itemAdded(g, name);
    }
}

void Roe::itemDeleted(GeomObj *g) {
    for (int i=0; i<geomItemA.size(); i++) {
	if (geomItemA[i]->geom == g) {
	    delete (geomItemA[i]->btn);
	    geomItemA.remove(i);
	}
    }
    for (i=0; i<kids.size(); i++) {
	kids[i]->itemDeleted(g);
    }
}


void Roe::redrawAll()
{
    if (doneInit) {
	// clear screen
	evl->lock();
        make_current();  
	glClearColor(0,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	HashTableIter<int,HashTable<int, GeomObj*>*> iter(&manager->portHash);
	for (iter.first(); iter.ok(); ++iter) {
	    HashTable<int, GeomObj*>* serHash=iter.get_data();
	    HashTableIter<int, GeomObj*> serIter(serHash);
	    for (serIter.first(); serIter.ok(); ++serIter) {
		GeomObj *geom=serIter.get_data();
		for (int i=0; i<geomItemA.size(); i++)
		    if (geomItemA[i]->geom == geom)
			if (geomItemA[i]->vis)
			    geom->draw();
	    }	
	}
	GLwDrawingAreaSwapBuffers(*graphics);
	for (int i=0; i<kids.size(); i++) {
	    kids[i]->redrawAll();
	}
	evl->unlock();       
    }
}

void Roe::printLevel(int level, int&flag) {
    if (level == 0) {
	flag=1;
	cerr << "* ";
    } else {
	for (int i=0; i<kids.size(); i++) {
	    kids[i]->printLevel(level-1, flag);
	}
    }
}
 
// need to fill this in!   
void Roe::itemCB(CallbackData*, void *gI) {
    GeomItem *g = (GeomItem *)gI;
    for (int i=0; i<geomItemA.size(); i++) {
	if (geomItemA[i]->geom == g->geom) {
	    if (geomItemA[i]->vis) {
		geomItemA[i]->vis=0;
	    } else {
		geomItemA[i]->vis=1;
	    }
	}
    }
    redrawAll();
}

void Roe::destroyWidgetCB(CallbackData*, void*)
{
    // can't close the only window -- this doesn't seem to work, though...
    if (firstGen && (manager->topRoe.size() == 1) && (kids.size()==0)) 
	return;
    else
	delete this;
}

void Roe::spawnChCB(CallbackData*, void*)
{
  double mat[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, mat);

/*  for (int i=0;i<16;i++)
      cerr << mat[i] << " ";
  cerr << "\n";
*/
  kids.add(new Roe(manager, mat));
  kids[kids.size()-1]->SetParent(this);
//  manager->printFamilyTree();

}
    
Roe::~Roe()
{
    delete dialog;
    delete wholeWin;
    delete left;
    delete right;
    delete graphics;
    delete controls;
    delete objBox;
    delete objLabel;
    delete objSep;
    delete objScroll;
    delete objRC;
    delete shadeBox;
    delete shadeLabel;
    delete shadeSep;
    delete shadeRC;
    delete wire;
    delete flat;
    delete phong;
    delete gouraud;
    delete lightBox;
    delete lightLabel;
    delete lightScroll;
    delete lightSep;
    delete lightRC;
    delete ambient;
    delete point1;
    delete options;
    delete viewRC;
    delete autoView;
    delete setHome;
    delete goHome;
    delete buttons;
    delete spawnRC;
    delete spawnCh;
    delete spawnInd;
    delete form;
    delete gr_frame;
    for (int i=0; i<geomItemA.size(); i++)
	delete geomItemA[i];
    geomItemA.remove_all();

    // tell my parent to delete me from their kid list
    if (firstGen) {
	manager->delTopRoe(this);
    } else {
	parent->deleteChild(this);
    }

    // now take care of the kids!  If I'm first generation, add them
    // to the Salmon's topRoe and delete myself from the Salmon's topRoe
    // Otherwise, give them to my parents and delete myself from my parents
    // Don't forget to set their firstGen, and parent variables accordingly
    if (firstGen) {
	for (int i=0; i<kids.size(); i++) {
	    kids[i]->SetTop();
	    manager->addTopRoe(kids[i]);
	}
    } else {
	for (int i=0; i<kids.size(); i++) {
	    kids[i]->SetParent(parent);
	    parent->addChild(kids[i]);
	}
    }
//    manager->printFamilyTree();
}

void Roe::SetParent(Roe *r)
{
  parent = r;
}

void Roe::SetTop()
{
  firstGen=True;
}

void Roe::addChild(Roe *r)
{
    kids.add(r);
}

// self-called method
void Roe::deleteChild(Roe *r)
{
    for (int i=0; i<kids.size(); i++)
	if (r==kids[i]) kids.remove(i);
}
void Roe::wireCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::wireCB");
}
void Roe::flatCB(CallbackData*, void*)
{
    if (glIsEnabled(GL_LIGHTING)) {
	make_current();
	glDisable(GL_LIGHTING);
	redrawAll();
    }
}
void Roe::gouraudCB(CallbackData*, void*)
{
    if (!glIsEnabled(GL_LIGHTING)) {
	make_current();
	glEnable(GL_LIGHTING);
	redrawAll();
    }
}
void Roe::phongCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::phongCB");
}
void Roe::ambientCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::ambientCB");
}
void Roe::point1CB(CallbackData*, void*)
{
    make_current();
    if (!glIsEnabled(GL_LIGHT0)) {
	glEnable(GL_LIGHT0);
    } else {
	glDisable(GL_LIGHT0);
    }
    redrawAll();
}
void Roe::goHomeCB(CallbackData*, void*)
{
    make_current();
    glLoadMatrixd(inheritMat);
    redrawAll();
}
void Roe::autoViewCB(CallbackData*, void*)
{
    BBox bbox;
    HashTableIter<int,HashTable<int, GeomObj*>*> iter(&manager->portHash);
    for (iter.first(); iter.ok(); ++iter) {
	HashTable<int, GeomObj*>* serHash=iter.get_data();
	HashTableIter<int, GeomObj*> serIter(serHash);
	for (serIter.first(); serIter.ok(); ++serIter) {
	    GeomObj *geom=serIter.get_data();
	    for (int i=0; i<geomItemA.size(); i++)
		if (geomItemA[i]->geom == geom)
		    if (geomItemA[i]->vis)
			bbox.extend(geom->bbox());
	}		
    }	
    Point lookat(bbox.center());
    lookat.z(bbox.max().z());
    double xwidth=lookat.x()-bbox.min().x();
    double ywidth=lookat.y()-bbox.min().y();
    double dist=Max(xwidth, ywidth);
    make_current();
    evl->lock();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90, 1.33, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(lookat.x(), lookat.y(), lookat.z()+dist, lookat.x(), lookat.y(), lookat.z(), 0, 1, 0);
    evl->unlock();
    redrawAll();
}    

void Roe::setHomeCB(CallbackData*, void*)
{
    make_current();
    glGetDoublev(GL_MODELVIEW_MATRIX, inheritMat);
}
Roe::Roe(const Roe& copy)
{
    NOT_FINISHED("Roe::Roe");
}

void Roe::rotate(double angle, Vector v)
{
    make_current();
    glScaled(v.x(), v.y(), v.z());
}

void Roe::translate(Vector v)
{
    make_current();
    glTranslated(v.x(), v.y(), v.z());
}

void Roe::scale(Vector v)
{
    make_current();
    glScaled(v.x(), v.y(), v.z());
}

void Roe::btn1upCB(CallbackData* cbdata, void*) {
    XEvent* event=cbdata->get_event();
}

void Roe::btn1downCB(CallbackData* cbdata, void*) {
    XEvent* event=cbdata->get_event();
    last_x=event->xbutton.x;
    last_y=event->xbutton.y;
}

void Roe::btn1motionCB(CallbackData* cbdata, void*) {
    XEvent* event=cbdata->get_event();
    double xmtn=last_x-event->xmotion.x;
    double ymtn=last_y-event->xmotion.y;
    xmtn/=10;
    ymtn/=10;
    last_x = event->xmotion.x;
    last_y = event->xmotion.y;
    make_current();
    glTranslated(-xmtn, ymtn, 0);
    for (int i=0; i<kids.size(); i++)
	kids[i]->translate(Vector(-xmtn, ymtn, 0));
    redrawAll();
}

void Roe::btn2upCB(CallbackData* cbdata, void*) {
    XEvent* event=cbdata->get_event();
}

void Roe::btn2downCB(CallbackData* cbdata, void*) {
    XEvent* event=cbdata->get_event();
    last_x=event->xbutton.x;
    last_y=event->xbutton.y;
}

void Roe::btn2motionCB(CallbackData* cbdata, void*) {
    double scl;
    XEvent* event=cbdata->get_event();
    double xmtn=last_x-event->xmotion.x;
    double ymtn=last_y-event->xmotion.y;
    xmtn/=30;
    ymtn/=30;
    last_x = event->xmotion.x;
    last_y = event->xmotion.y;
    make_current();
    if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
    glScaled(1+scl, 1+scl, 1+scl);
    for (int i=0; i<kids.size(); i++)
	kids[i]->scale(Vector(1+scl, 1+scl, 1+scl));
    redrawAll();
}

