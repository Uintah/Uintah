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
#include <Geom.h>
#include <iostream.h>
#include <GL/glu.h>
extern MtXEventLoop* evl;

Roe::Roe(Salmon* s) 
{
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

    item1=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(item1, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::item1CB,
				0, 0);
    item1->Create(*objRC, "Item1");

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
    new MotifCallback<Roe>FIXCB(wire, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::wireCB,
				0, 0);
    wire->Create(*shadeRC, "Wire");
    flat=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(flat, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::flatCB,
				0, 0);
    flat->Create(*shadeRC, "Flat");
    gouraud=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(gouraud, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::gouraudCB,
				0, 0);
    gouraud->Create(*shadeRC, "Gouraud");
    phong=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(phong, XmNactivateCallback,
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
    new MotifCallback<Roe>FIXCB(ambient, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::ambientCB,
				0, 0);
    ambient->Create(*lightRC, "Ambient");
    point1=new ToggleButtonC;
    new MotifCallback<Roe>FIXCB(point1, XmNactivateCallback,
				&manager->mailbox, this,
				&Roe::point1CB,
				0, 0);
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
    glEnable(GL_COLOR_MATERIAL);
    evl->unlock();
    doneInit=1;
}

void Roe::make_current() {
    evl->lock();
    GLwDrawingAreaMakeCurrent(*graphics, cx);
    evl->unlock();
}
void Roe::redrawAll()
{
    if (doneInit) {
	// clear screen
evl->lock();
	make_current();

	// set the view
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90, 1.33, 1, 10);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(2,2,5,2,2,2,0,1,0);
	glClearColor(0,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	HashTableIter<int,HashTable<int, GeomObj*>*> iter(&manager->portHash);
	for (iter.first(); iter.ok(); ++iter) {
	    HashTable<int, GeomObj*>* serHash=iter.get_data();
	    HashTableIter<int, GeomObj*> serIter(serHash);
	    for (serIter.first(); serIter.ok(); ++serIter) {
		GeomObj *geom=serIter.get_data();
		geom->draw();
	    }
	}
	for (int i=0; i<kids.size(); i++) {
	    kids[i]->redrawAll();
	}
	GLwDrawingAreaSwapBuffers(*graphics);
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
  kids.add(new Roe(manager));
  kids[kids.size()-1]->SetParent(this);
  manager->printFamilyTree();

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
    delete item1;
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
    manager->printFamilyTree();
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
void Roe::item1CB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::Item1CB");
}
void Roe::wireCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::wireCB");
}
void Roe::flatCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::flatCB");
}
void Roe::gouraudCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::gouraudCB");
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
    NOT_FINISHED("Roe::point1CB");
}
void Roe::goHomeCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::goHomeCB");
}
void Roe::autoViewCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::autoViewCB");
}
void Roe::setHomeCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::setHomeCB");
}
Roe::Roe(const Roe& copy)
{
    NOT_FINISHED("Roe::Roe");
}

void Roe::rotate(double angle, Vector v)
{
    NOT_FINISHED("Roe::rotate");
}

void Roe::translate(Vector v)
{
    NOT_FINISHED("Roe:translate");
}

void Roe::scale(Vector v)
{
    NOT_FINISHED("Roe::scale");
}

