
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

#include <Modules/Salmon/Salmon.h>
#include <Modules/Salmon/Roe.h>
#include <Modules/Salmon/Renderer.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Devices/DBCallback.h>
#include <Devices/DBContext.h>
#include <Devices/Dialbox.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <Geometry/BBox.h>
#include <Geometry/Vector.h>
#include <Geom/Geom.h>
#include <Geom/PointLight.h>
#include <iostream.h>
#include <stdio.h>
#include <string.h>

typedef void (Roe::*MouseHandler)(int action, int x, int y,
				  int x_root, int y_root);
#define BUTTON_DOWN 0
#define BUTTON_UP 1
#define BUTTON_MOTION 2
#define SHIFT_MASK 1
#define CONTROL_MASK 2
#define META_MASK 4

static const int pick_buffer_size = 512;
static const double pick_window = 5.0;

struct MouseHandlerData {
    MouseHandler handler;	
    clString title;	
    MouseHandlerData(MouseHandler, const clString&);
    ~MouseHandlerData();
};

MouseHandlerData::MouseHandlerData(MouseHandler handler, const clString& title)
: handler(handler), title(title)
{
}

MouseHandlerData::~MouseHandlerData()
{
}

static MouseHandlerData mode_translate(&Roe::mouse_translate, "translate");
static MouseHandlerData mode_scale(&Roe::mouse_scale, "scale");
static MouseHandlerData mode_rotate(&Roe::mouse_rotate, "rotate");
static MouseHandlerData mode_pick(&Roe::mouse_pick, "pick");

static MouseHandlerData* mouse_handlers[8][3] = {
    &mode_translate, 	// No modifiers, button 1
    &mode_scale,       	// No modifiers, button 2
    &mode_rotate,	// No modifiers, button 3
    &mode_pick,		// Shift, button 1
    0,			// Shift, button 2
    0,			// Shift, button 3
    0,			// Control, button 1
    0,			// Control, button 2
    0,			// Control, button 3
    0,			// Control+Shift, button 1
    0,			// Control+Shift, button 2
    0,			// Control+Shift, button 3
    0,			// Alt, button 1
    0,			// Alt, button 2
    0,			// Alt, button 3
    0,			// Alt+Shift, button 1
    0,			// Alt+Shift, button 2
    0,			// Alt+Shift, button 3
    0,			// Alt+Control, button 1
    0,			// Alt+Control, button 2
    0,			// Alt+Control, button 3
    0,			// Alt+Control+Shift, button 1
    0,			// Alt+Control+Shift, button 2
    0,			// Alt+Control+Shift, button 3
};	

static total_salmon_count=1;

Roe::Roe(Salmon* s, const clString& id)
: id(id), manager(s), view(Point(0,0,1), Point(0,0,0), Vector(0,1,0), 45)
{
    TCL::add_command(id, this, 0);
    current_renderer=0;
}

#ifdef OLDUI
void Roe::orthoCB(CallbackData*, void*) {
    evl->lock();
    make_current();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-8, 8, -6, 6, -100, 100);
    // determine the bounding box so we can scale the view
    get_bounds(bb);
    if (bb.valid()) {
	Point cen(bb.center());
	double cx=cen.x();
	double cy=cen.y();
	double xw=cx-bb.min().x();
	double yw=cy-bb.min().y();
	double scl=4/Max(xw,yw);
	glScaled(scl,scl,scl);
    }
    glMatrixMode(GL_MODELVIEW);
    evl->unlock();
    need_redraw=1;
}

void Roe::perspCB(CallbackData*, void*) {
    evl->lock();
    make_current();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90, 1.33, 1, 100);
    glMatrixMode(GL_MODELVIEW);
    evl->unlock();
    need_redraw=1;
}

void Roe::redrawCB(CallbackData*, void*){
    if(!doneInit)
	initCB(0, 0);
    need_redraw=1;
}

void Roe::initCB(CallbackData*, void*) {
    XVisualInfo* vi;
    graphics->GetVisualInfo(&vi);
    graphics->GetValues();
    // Create a GLX context
    evl->lock();
    cx = glXCreateContext(XtDisplay(*graphics), vi, 0, GL_TRUE);
    if(!cx){
	cerr << "Error making glXCreateContext\n";
	exit(0);
    }
    make_current();

    // set the view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // opted for orthographic as the default
    glOrtho(-8, 8, -6, 6, -100, 100);
    get_bounds(bb);
    if (bb.valid()) {
	Point cen(bb.center());
	double cx=cen.x();
	double cy=cen.y();
	double xw=cx-bb.min().x();
	double yw=cy-bb.min().y();
	double scl=4/Max(xw,yw);
	glScaled(scl,scl,scl);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(2,2,5,2,2,2,0,1,0);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    GLfloat light_position[] = { 0, 0, -1, 0 };
    glLightfv(GL_LIGHT2, GL_POSITION, light_position);
    GLfloat light_white[] = {1,1,1,1};
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_white);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_white);
    glLightfv(GL_LIGHT1, GL_SPECULAR, light_white);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, light_white);
    glLightfv(GL_LIGHT2, GL_SPECULAR, light_white);
    GLfloat light_black[] = {0,0,0,1};
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_black);;
    glLightfv(GL_LIGHT1, GL_AMBIENT, light_black);
    glLightfv(GL_LIGHT2, GL_AMBIENT, light_black);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHT1);
    glEnable(GL_LIGHT2);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glFogf(GL_FOG_END, 30.0);
    glFogf(GL_FOG_MODE, GL_LINEAR);

    if (haveInheritMat) {
	glLoadMatrixd(inheritMat);
    } else {
	glGetDoublev(GL_MODELVIEW_MATRIX, inheritMat);
    }

    evl->unlock();
    doneInit=1;
}
#endif

void Roe::itemAdded(GeomObj *g, const clString& name)
{
    NOT_FINISHED("Roe::itemAdded");
    // invalidate the bounding box
    bb.reset();
}

void Roe::itemDeleted(GeomObj *g)
{
    NOT_FINISHED("Roe::itemDeleted");
    // invalidate the bounding box
    bb.reset();
}


#ifdef OLDUI
void Roe::redrawAll()
{
    if (doneInit) {
	// clear screen
        make_current();  

	// have to redraw the lights for them to have the right transformation
	drawinfo->polycount=0;
	WallClockTimer timer;
	timer.clear();
	timer.start();
	GLfloat light_position0[] = { 500,500,-100,1};
	glLightfv(GL_LIGHT0, GL_POSITION, light_position0);
	GLfloat light_position1[] = { -50,-100,100,1};
	glLightfv(GL_LIGHT1, GL_POSITION, light_position1);
	if (!drawinfo->pick_mode && !drawinfo->edit_mode) {
	    glClearColor(0,0,0,1);
	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
	drawinfo->push_matl(manager->default_matl);
	HashTableIter<int,HashTable<int, GeomObj*>*> iter(&manager->portHash);
	for (iter.first(); iter.ok(); ++iter) {
	    HashTable<int, GeomObj*>* serHash=iter.get_data();
	    HashTableIter<int, GeomObj*> serIter(serHash);
	    for (serIter.first(); serIter.ok(); ++serIter) {
		GeomObj *geom=serIter.get_data();
		for (int i=0; i<geomItemA.size(); i++)
		    if (geomItemA[i]->geom == geom)
			// we draw it if we're editing, if it's pickable,
			//  or if we're not in pick-mode
			if (geomItemA[i]->vis && 
			    (drawinfo->edit_mode || !drawinfo->pick_mode || 
			     geom->pick)){
			    if (drawinfo->edit_mode || drawinfo->pick_mode)
				glLoadName((GLuint)geom);
			    if (drawinfo->pick_mode)
				glPushName((GLuint)geom->pick);
			    geom->draw(drawinfo);
			    if (drawinfo->pick_mode)
				glPopName();
			}
	    }
	}
	drawinfo->pop_matl();
#ifdef OLDUI
	if (!drawinfo->pick_mode && !drawinfo->edit_mode)
	    GLwDrawingAreaSwapBuffers(*graphics);
#endif
	timer.stop();
#ifdef OLDUI
	perf_string1=to_string(drawinfo->polycount)+" polygons in "
	    +to_string(timer.time())+" seconds";
	perf_string2=to_string(double(drawinfo->polycount)/timer.time())
	    +" polygons/second";
	redraw_perf(0, 0);
	evl->unlock();       
#endif
    }
}
#endif

// need to fill this in!   
#ifdef OLDUI
void Roe::itemCB(CallbackData*, void *gI) {
    GeomItem *g = (GeomItem *)gI;
    g->vis = !g->vis;
    need_redraw=1;
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

  kids.add(new Roe(manager, mat, mtnScl));
  kids[kids.size()-1]->SetParent(this);
  for (int i=0; i<geomItemA.size(); i++)
      kids[kids.size()-1]->itemAdded(geomItemA[i]->geom, geomItemA[i]->name);

}
#endif
    
Roe::~Roe()
{
}

#ifdef OLDUI
void Roe::wireCB(CallbackData*, void*)
{
    drawinfo->drawtype=DrawInfo::WireFrame;
    drawinfo->current_matl=0;
    drawinfo->lighting=0;
    need_redraw=1;
}

void Roe::flatCB(CallbackData*, void*)
{
    drawinfo->drawtype=DrawInfo::Flat;
    drawinfo->current_matl=0;
    drawinfo->lighting=0;
    need_redraw=1;
}

void Roe::gouraudCB(CallbackData*, void*)
{
    drawinfo->drawtype=DrawInfo::Gouraud;
    drawinfo->current_matl=0;
    drawinfo->lighting=1;
    need_redraw=1;
}

void Roe::phongCB(CallbackData*, void*)
{
    drawinfo->drawtype=DrawInfo::Phong;
    drawinfo->current_matl=0;
    drawinfo->lighting=1;
    need_redraw=1;
}

void Roe::ambientCB(CallbackData*, void*)
{
    NOT_FINISHED("Roe::ambientCB");
}

void Roe::fogCB(CallbackData*, void*) {
    evl->lock();
    make_current();
    if (!glIsEnabled(GL_FOG)) {
	glEnable(GL_FOG);
    } else {
	glDisable(GL_FOG);
    }
    evl->unlock();
    need_redraw=1;
}

void Roe::point1CB(CallbackData*, void*)
{
    evl->lock();
    make_current();
    if (!glIsEnabled(GL_LIGHT0)) {
	glEnable(GL_LIGHT0);
    } else {
	glDisable(GL_LIGHT0);
    }
    evl->unlock();
    need_redraw=1;
}

void Roe::point2CB(CallbackData*, void*)
{
    evl->lock();
    make_current();
    if (!glIsEnabled(GL_LIGHT1)) {
	glEnable(GL_LIGHT1);
    } else {
	glDisable(GL_LIGHT1);
    }
    evl->unlock();
    need_redraw=1;
}

void Roe::head1CB(CallbackData*, void*)
{
    evl->lock();
    make_current();
    if (!glIsEnabled(GL_LIGHT2)) {
	glEnable(GL_LIGHT2);
    } else {
	glDisable(GL_LIGHT2);
    }
    evl->unlock();
    need_redraw=1;
}

void Roe::goHomeCB(CallbackData*, void*)
{
    evl->lock();
    make_current();
    glLoadMatrixd(inheritMat);
    evl->unlock();
    need_redraw=1;
}
#endif

void Roe::get_bounds(BBox& bbox)
{
    NOT_FINISHED("Roe::get_bounds");
    bbox.reset();
    HashTableIter<int,HashTable<int, GeomObj*>*> iter(&manager->portHash);
    for (iter.first(); iter.ok(); ++iter) {
	HashTable<int, GeomObj*>* serHash=iter.get_data();
	HashTableIter<int, GeomObj*> serIter(serHash);
	for (serIter.first(); serIter.ok(); ++serIter) {
	    GeomObj *geom=serIter.get_data();
	    geom->get_bounds(bbox);
	}
    }
}

#ifdef OLDUI
void Roe::autoViewCB(CallbackData*, void*)
{
    BBox bbox;
    get_bounds(bbox);
    if (!bbox.valid()) return;
    Point lookat(bbox.center());
    lookat.z(bbox.max().z());
    double lx=lookat.x();
    double xwidth=lx-bbox.min().x();
    double ly=lookat.y();
    double ywidth=ly-bbox.min().y();
    double dist=Max(xwidth, ywidth);
    evl->lock();
    make_current();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    double lz=lookat.z();
    gluLookAt(lx, ly, lz+dist, lx, ly, lz, 0, 1, 0);
    mtnScl=1;
    // if we're in orthographics mode, scale appropriately
    char ort;
    orthoB->GetSet(&ort);
    orthoB->GetValues();
    if (ort) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-8,8,-6,6,-100,100);
	glScaled(4/dist, 4/dist, 4/dist);
	glMatrixMode(GL_MODELVIEW);
    }
    evl->unlock();
    need_redraw=1;
}    

void Roe::setHomeCB(CallbackData*, void*)
{
    evl->lock();
    make_current();
    glGetDoublev(GL_MODELVIEW_MATRIX, inheritMat);
    evl->unlock();
}
#endif

Roe::Roe(const Roe& copy)
: view(copy.view)
{
    NOT_FINISHED("Roe::Roe");
}

void Roe::rotate(double angle, Vector v, Point c)
{
    NOT_FINISHED("Roe::rotate");
#ifdef OLDUI
    evl->lock();
    make_current();
    double temp[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, temp);
    glPopMatrix();
    glLoadIdentity();
    glTranslated(c.x(), c.y(), c.z());
    glRotated(angle,v.x(), v.y(), v.z());
    glTranslated(-c.x(), -c.y(), -c.z());
    glMultMatrixd(temp);
    for (int i=0; i<kids.size(); i++)
	kids[i]->rotate(angle, v, c);
    evl->unlock();
#endif
    need_redraw=1;
}

void Roe::rotate_obj(double angle, const Vector& v, const Point& c)
{
    NOT_FINISHED("Roe::rotate_obj");
#ifdef OLDUI
    evl->lock();
    make_current();
    glTranslated(c.x(), c.y(), c.z());
    glRotated(angle, v.x(), v.y(), v.z());
    glTranslated(-c.x(), -c.y(), -c.z());
    for(int i=0; i<kids.size(); i++)
	kids[i]->rotate(angle, v, c);
    evl->unlock();
#endif
    need_redraw=1;
}

static void mmult(double *m, double *p1, double *p2) {
    for (int i=0; i<4; i++) {
	p2[i]=0;
	for (int j=0; j<4; j++) {
	    p2[i]+=m[j*4+i]*p1[j];
	}
    }
    for (i=0; i<3; i++)
	p2[i]/=p2[3];
}

void Roe::translate(Vector v)
{
    NOT_FINISHED("Roe::translate");
#ifdef OLDUI
    evl->lock();
    make_current();
    double temp[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, temp);
    glPopMatrix();
    glLoadIdentity();
    glTranslated(v.x()*mtnScl, v.y()*mtnScl, v.z()*mtnScl);
    glMultMatrixd(temp);
    for (int i=0; i<kids.size(); i++)
	kids[i]->translate(v);
    evl->unlock();
#endif
    need_redraw=1;
}

void Roe::scale(Vector v, Point c)
{
    NOT_FINISHED("Roe::scale");
#ifdef OLDUI
    evl->lock();
    make_current();
    glTranslated(c.x(), c.y(), c.z());
    glScaled(v.x(), v.y(), v.z());
    glTranslated(-c.x(), -c.y(), -c.z());
    mtnScl*=v.x();
    for (int i=0; i<kids.size(); i++)
	kids[i]->scale(v, c);
    evl->unlock();
#endif
    need_redraw=1;
}

#ifdef OLDUI
void Roe::eventCB(CallbackData* cbdata, void*)
{
    XEvent* event=cbdata->get_event();
    switch(event->type){
    case EnterNotify:
	evl->lock();
	XmProcessTraversal(*graphics, XmTRAVERSE_CURRENT);
	evl->unlock();
	return;
    case KeyPress:
	if(event->xkey.state & (Button1Mask|Button2Mask|Button3Mask)){
	    // Skip it...
	} else{
	    int mask=0;
	    if(event->xkey.state & ControlMask)
		mask|=CONTROL_MASK;
	    if(event->xkey.state & ShiftMask)
		mask|=SHIFT_MASK;
	    if(event->xkey.state & (Mod1Mask|Mod2Mask|Mod3Mask|Mod4Mask|Mod5Mask))
		mask|=META_MASK;
	    switch(XLookupKeysym(&event->xkey, 0)){
	    case XK_Shift_L:
	    case XK_Shift_R:
		mask|=SHIFT_MASK;
		break;
	    case XK_Control_L:
	    case XK_Control_R:
		mask|=CONTROL_MASK;
		break;
	    case XK_Meta_L:
	    case XK_Meta_R:
	    case XK_Alt_L:
	    case XK_Alt_R:
		mask|=META_MASK;
		break;
	    }
	    
	    if(mask != modifier_mask){
		modifier_mask=mask;
		update_modifier_widget();
	    }
	}
	break;
    case KeyRelease:
	if(event->xkey.state & (Button1Mask|Button2Mask|Button3Mask)){
	    // Skip it...
	} else{
	    int mask=0;
	    if(event->xkey.state & ControlMask)
		mask|=CONTROL_MASK;
	    if(event->xkey.state & ShiftMask)
		mask|=SHIFT_MASK;
	    if(event->xkey.state & (Mod1Mask|Mod2Mask|Mod3Mask|Mod4Mask|Mod5Mask))
		mask|=META_MASK;
	    switch(XLookupKeysym(&event->xkey, 0)){
	    case XK_Shift_L:
	    case XK_Shift_R:
		mask&=~SHIFT_MASK;
		break;
	    case XK_Control_L:
	    case XK_Control_R:
		mask&=~CONTROL_MASK;
		break;
	    case XK_Meta_L:
	    case XK_Meta_R:
	    case XK_Alt_L:
	    case XK_Alt_R:
		mask&=~META_MASK;
		break;
	    }
	    if(mask != modifier_mask){
		modifier_mask=mask;
		update_modifier_widget();
	    }
	}
	break;
    case ButtonPress:
	{
	    switch(event->xbutton.button){
	    case Button1:
		last_btn=1;
		break;
	    case Button2:
		last_btn=2;
		break;
	    case Button3:
		last_btn=3;
		break;
	    default:
		last_btn=1;
		break;
	    }
	    int mask=0;
	    if(event->xbutton.state & ControlMask)
		mask|=CONTROL_MASK;
	    if(event->xbutton.state & ShiftMask)
		mask|=SHIFT_MASK;
	    if(event->xbutton.state & (Mod1Mask|Mod2Mask|Mod3Mask|Mod4Mask|Mod5Mask))
		mask|=META_MASK;
	    if(mask != modifier_mask){
		modifier_mask=mask;
		update_modifier_widget();
	    }
	    MouseHandler handler=mouse_handlers[modifier_mask][last_btn-1]->handler;
	    if(handler){
		(this->*handler)(BUTTON_DOWN,
				 event->xbutton.x, event->xbutton.y,
				 event->xbutton.x_root, event->xbutton.y_root);
	    }
	}
	break;
    case ButtonRelease:
	{
	    switch(event->xbutton.button){
	    case Button1:
		last_btn=1;
		break;
	    case Button2:
		last_btn=2;
		break;
	    case Button3:
		last_btn=3;
		break;
	    default:
		last_btn=1;
		break;
	    }
	    MouseHandler handler=mouse_handlers[modifier_mask][last_btn-1]->handler;
	    if(handler){
		(this->*handler)(BUTTON_UP,
				 event->xbutton.x, event->xbutton.y,
				 event->xbutton.x_root, event->xbutton.y_root);
	    }
	    int mask=0;
	    if(event->xbutton.state & ControlMask)
		mask|=CONTROL_MASK;
	    if(event->xbutton.state & ShiftMask)
		mask|=SHIFT_MASK;
	    if(event->xbutton.state & (Mod1Mask|Mod2Mask|Mod3Mask|Mod4Mask|Mod5Mask))
		mask|=META_MASK;
	    if(mask != modifier_mask){
		modifier_mask=mask;
		update_modifier_widget();
	    }
	}
	break;
    case MotionNotify:
	{
	    MouseHandler handler=mouse_handlers[modifier_mask][last_btn-1]->handler;
	    if(handler){
		(this->*handler)(BUTTON_MOTION,
				 event->xmotion.x, event->xmotion.y,
				 event->xmotion.x_root, event->xmotion.y_root);
	    }
	}
	break;
    default:
	cerr << "Unknown event..\n";
	break;
    }
}
#endif

void Roe::mouse_translate(int action, int x, int y, int, int)
{
    NOT_FINISHED("Roe::mouse_translate");
#ifdef OLDUI
    switch(action){
    case BUTTON_DOWN:
	last_x=x;
	last_y=y;
	total_x = 0;
	total_y = 0;
#ifdef OLDUI
	update_mode_string("translate: ");
#endif
	break;
    case BUTTON_MOTION:
	{
	    double xmtn=last_x-x;
	    double ymtn=last_y-y;
	    xmtn/=10/mtnScl;
	    ymtn/=10/mtnScl;
	    last_x = x;
	    last_y = y;
	    total_x += xmtn;
	    total_y += ymtn;
	    if (Abs(total_x) < .001) total_x = 0;
	    if (Abs(total_y) < .001) total_y = 0;
#ifdef OLDUI
	    evl->lock();
#endif
	    make_current();
	    double temp[16];
	    glGetDoublev(GL_MODELVIEW_MATRIX, temp);
	    glPopMatrix();
	    glLoadIdentity();
	    glTranslated(-xmtn, ymtn, 0);
	    // post-multiply by the translate to be sure it happens last!
	    glMultMatrixd(temp);
	    for (int i=0; i<kids.size(); i++)
		kids[i]->translate(Vector(-xmtn/mtnScl, ymtn/mtnScl, 0));
	    need_redraw=1;
#ifdef OLDUI
	    update_mode_string(clString("translate: ")+to_string(total_x)
			       +", "+to_string(total_y));
	    evl->unlock();
#endif
	}
	break;
    case BUTTON_UP:
#ifdef OLDUI
	update_mode_string("");
#endif
	break;
    }
#endif
}

void Roe::mouse_scale(int action, int x, int y, int, int)
{
    NOT_FINISHED("Roe::mouse_scale");
#ifdef OLDUI
    switch(action){
    case BUTTON_DOWN:
	{
#ifdef OLDUI
	    update_mode_string("scale: ");
#endif
	    last_x=x;
	    last_y=y;
	    total_x=1;
	    get_bounds(bb);
	}
	break;
    case BUTTON_MOTION:
	{
	    if (!bb.valid()) break;
	    double scl;
	    double xmtn=last_x-x;
	    double ymtn=last_y-y;
	    xmtn/=30;
	    ymtn/=30;
	    last_x = x;
	    last_y = y;
#ifdef OLDUI
	    evl->lock();
#endif
	    make_current();
	    if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
	    if (scl<0) scl=1/(1-scl); else scl+=1;
	    total_x*=scl;
	    mtnScl*=scl;
	    Point cntr(bb.center());
	    // premultiplying by the scale works just fine
	    glTranslated(cntr.x(), cntr.y(), cntr.z());
	    glScaled(scl, scl, scl);
	    glTranslated(-cntr.x(), -cntr.y(), -cntr.z());
	    for (int i=0; i<kids.size(); i++) {
		kids[i]->scale(Vector(scl, scl, scl), cntr);
	    }
	    need_redraw=1;
#ifdef OLDUI
	    update_mode_string(clString("scale: ")+to_string(total_x*100)+"%");
	    evl->unlock();
#endif 
	}
	break;
    case BUTTON_UP:
#ifdef OLDUI
	update_mode_string("");
#endif
	break;
    }	
#endif
}

void Roe::mouse_rotate(int action, int x, int y, int, int)
{
    NOT_FINISHED("Roe::mouse_rotate");
#ifdef OLDUI
    switch(action){
    case BUTTON_DOWN:
	{
#ifdef OLDUI
	    update_mode_string("rotate:");
#endif
	    last_x=x;
	    last_y=y;
	    get_bounds(bb);
	}
	break;
    case BUTTON_MOTION:
	{
	    if (!bb.valid()) break;
	    double xmtn=last_x-x;
	    double ymtn=last_y-y;
	    last_x = x;
	    last_y = y;
#ifdef OLDUI
	    evl->lock();
#endif
	    make_current();
	    Point cntr(bb.center());
	    double mm[16];
	    double pm[16];
	    int vp[4];
	    double trans[4];
	    double centr[4];
	    double cx, cy, cz;
	    glGetDoublev(GL_MODELVIEW_MATRIX, mm);
	    glMatrixMode(GL_PROJECTION);
	    glPushMatrix();
	    glLoadIdentity();
	    gluPerspective(90,1.33, 1, 100);
	    glGetDoublev(GL_PROJECTION_MATRIX, pm);
	    glPopMatrix();
	    glGetIntegerv(GL_VIEWPORT, vp);
	    // unproject the center of the viewport, w/ z-value=.5
	    // to find the point we want to rotate around.
	    if (gluUnProject(vp[0]+vp[2]/2, vp[1]+vp[3]/2, .1, mm, pm, vp, 
			 &cx, &cy, &cz) == GL_FALSE) 
		cerr << "Error Projecting!\n";
	    centr[0]=cx; centr[1]=cy; centr[2]=cz; centr[3]=1;
	    // multiply that point by our current modelview matrix to get
	    // the z-translate necessary to put that point at the origin
	    mmult(mm, centr, trans);
	    glMatrixMode(GL_MODELVIEW);
	    glPopMatrix();
	    glLoadIdentity();
	    double totMtn=Sqrt(xmtn*xmtn+ymtn*ymtn)/5;
	    // these also need to be post-multiplied
	    glTranslated(0, 0, trans[2]);
	    glRotated(totMtn,-ymtn,-xmtn,0);
	    glTranslated(0,0, -trans[2]);
	    glMultMatrixd(mm);
	    for (int i=0; i<kids.size(); i++) {
		kids[i]->rotate(totMtn,Vector(-ymtn,-xmtn,0),
				Point(0,0,trans[2]));
	    }
	    need_redraw=1;
#ifdef OLDUI
	    update_mode_string("rotate:");
	    evl->unlock();
#endif
	}
	break;
    case BUTTON_UP:
#ifdef OLDUI
	update_mode_string("");
#endif
	break;
    }
#endif
}

void Roe::mouse_pick(int action, int x, int y, int, int)
{
    NOT_FINISHED("Roe::mouse_pick");
#ifdef OLDUI
    switch(action){
    case BUTTON_DOWN:
	{
	    total_x=0;
	    total_y=0;
	    total_z=0;
	    last_x=x;
	    last_y=y;
#ifdef OLDUI
	    evl->lock();
#endif
	    make_current();
	    GLint viewport[4];
	    glGetIntegerv(GL_VIEWPORT, viewport);
	    GLuint pick_buffer[pick_buffer_size];
	    glSelectBuffer(pick_buffer_size, pick_buffer);
	    glRenderMode(GL_SELECT);
	    glInitNames();
	    glPushName(0);

	    // load the old perspetive matrix, so we can use it
	    double pm[16];
	    glGetDoublev(GL_PROJECTION_MATRIX, pm);
	    glMatrixMode(GL_PROJECTION);
	    glPushMatrix();
	    glLoadIdentity();
	    gluPickMatrix(x,viewport[3]-y,pick_window,pick_window,viewport);
	    glMultMatrixd(pm);

	    // Redraw the scene
#ifdef __GNUG__
	    int
#else
	    DrawInfo::DrawType
#endif
	           olddrawtype(drawinfo->drawtype);
	    drawinfo->drawtype=DrawInfo::Flat;
	    drawinfo->pick_mode=1;
	    redrawAll();
	    drawinfo->drawtype=olddrawtype;
	    drawinfo->pick_mode=0;

	    glPopMatrix();
	    glMatrixMode(GL_MODELVIEW);
	    glFlush();
	    int hits=glRenderMode(GL_RENDER);
	    GLuint min_z;
	    GLuint pick_index=0;
	    GLuint pick_pick=0;
	    if(hits >= 1){
		min_z=pick_buffer[1];
		pick_index=pick_buffer[3];
		pick_pick=pick_buffer[4];
		for (int h=1; h<hits; h++) {
		    ASSERT(pick_buffer[h*5]==2);
		    if (pick_buffer[h*5+1] < min_z) {
			min_z=pick_buffer[h*5+1];
			pick_index=pick_buffer[h*5+3];
			pick_pick=pick_buffer[h*5+4];
		    }
		}
	    }
	    geomSelected=(GeomObj *)pick_index;
	    gpick=(GeomPick*)pick_pick;
	    if (geomSelected) {
		for (int i=0; i<geomItemA.size(); i++) {
#ifdef OLDUI
		    if (geomItemA[i]->geom == geomSelected)
			update_mode_string(clString("pick: ")+
					   geomItemA[i]->name);
#endif
		}
		gpick->pick();
	    } else {
#ifdef OLDUI
		update_mode_string("pick: none");
#endif
	    }
#ifdef OLDUI
	    evl->unlock();
#endif
	}
	break;
    case BUTTON_MOTION:
	{
	    if (!geomSelected || !gpick) break;
	    double xmtn=last_x-x;
	    double ymtn=last_y-y;
	    xmtn/=30;
	    ymtn/=30;
#ifdef OLDUI
	    evl->lock();
#endif
	    make_current();
// project the center of the item grabbed onto the screen -- take the z
// component and unprojec the last and current x, y locations to get a 
// vector in object space.
	    BBox itemBB;
	    geomSelected->get_bounds(itemBB);
	    double midz=itemBB.center().z();
	    double mm[16], pm[16];
	    int vp[4];
	    glGetDoublev(GL_MODELVIEW_MATRIX, mm);
	    glGetDoublev(GL_PROJECTION_MATRIX, pm);
	    glGetIntegerv(GL_VIEWPORT, vp);
	    double x0, x1, y0, y1, z0, z1;
	    // unproject the center of the viewport, w/ z-value=.94
	    // to find the point we want to rotate around.
	    if (gluUnProject(last_x, vp[3]-last_y, midz, mm, pm, vp, 
			 &x0, &y0, &z0) == GL_FALSE) 
		cerr << "Error Projecting!\n";
	    if (gluUnProject(x, vp[3]-y, midz, mm, pm, vp, 
			 &x1, &y1, &z1) == GL_FALSE) 
		cerr << "Error Projecting!\n";
	    Vector dir(Point(x1, y1, z1)-Point(x0,y0,z0));
	    double dist=dir.length();
	    dir.normalize();
	    double dot=0;
	    int prin_dir=-1;
	    double currdot;
	    for (int i=0; i<gpick->nprincipal(); i++) {
		if ((currdot=Dot(dir, gpick->principal(i))) >dot){
		    dot=currdot;
		    prin_dir=i;
		}
	    }
	    if(prin_dir != -1){
		Vector mtn(gpick->principal(prin_dir)*dist);
		total_x+=mtn.x();
		total_y+=mtn.y();
		total_z+=mtn.z();
		if (Abs(total_x) < .0001) total_x=0;
		if (Abs(total_y) < .0001) total_y=0;
		if (Abs(total_z) < .0001) total_z=0;
		need_redraw=1;
		for (i=0; i<geomItemA.size(); i++) {
#ifdef OLDUI
		    if (geomItemA[i]->geom == geomSelected)
			update_mode_string(clString("pick: ")+
					   geomItemA[i]->name+
					   " "+to_string(total_x)+
					   ", "+to_string(total_y)+
					   ", "+to_string(total_z));
#endif
		}
		gpick->moved(prin_dir, dist, mtn);
	    } else {
#ifdef OLDUI
		update_mode_string(clString("Bad direction..."));
#endif
	    }
#ifdef OLDUI
	    evl->unlock();
#endif
	    last_x = x;
	    last_y = y;
	}
	break;
    case BUTTON_UP:
	if(gpick){
	    gpick->release();
	} else {
	    geomSelected=0;
	    gpick=0;
	}
#ifdef OLDUI
	update_mode_string("");
#endif
	break;
    }
#endif
}

#ifdef OLDUI
void Roe::update_mode_string(const clString& ms)
{
    mode_string=ms;
    update_modifier_widget();
}

void Roe::update_modifier_widget()
{
#ifdef OLDUI
    evl->lock();
#endif 
    if(!buttons_exposed)return;
    Window w=XtWindow(*buttons);
    Display* dpy=XtDisplay(*buttons);
    XClearWindow(dpy, w);
    XSetForeground(dpy, gc, mod_colors[modifier_mask]->pixel());
    Dimension h;
    buttons->GetHeight(&h);
    buttons->GetValues();
    int fh=h/5;
    if(fh != old_fh || modefont==0){
	modefont=new XFont(fh, XFont::Bold);
	XSetFont(dpy, gc, modefont->font->fid);
	old_fh=fh;
    }
    int fh2=fh/2;
    int fh4=fh2/2;
    int wid=fh2+fh4;
    XFillArc(dpy, w, gc, fh2,       fh2, fh2, fh2, 0, 180*64);
    XFillArc(dpy, w, gc, fh2+wid,   fh2, fh2, fh2, 0, 180*64);
    XFillArc(dpy, w, gc, fh2+2*wid, fh2, fh2, fh2, 0, 180*64);
    XFillArc(dpy, w, gc, fh2,       fh,  fh2, fh2, 180*64, 180*64);
    XFillArc(dpy, w, gc, fh2+wid,   fh,  fh2, fh2, 180*64, 180*64);
    XFillArc(dpy, w, gc, fh2+2*wid, fh,  fh2, fh2, 180*64, 180*64);
    XFillRectangle(dpy, w, gc, fh2,       fh2+fh4, fh2+1, fh2+2);
    XFillRectangle(dpy, w, gc, fh2+wid,   fh2+fh4, fh2+1, fh2+2);
    XFillRectangle(dpy, w, gc, fh2+2*wid, fh2+fh4, fh2+1, fh2+2);

    int toff=wid*3+fh;
    XDrawLine(dpy, w, gc, fh2+fh4, fh+fh4, fh2+fh4, 2*fh);
    XDrawLine(dpy, w, gc, fh2+fh4, 2*fh  , toff-fh4, 2*fh);
    XDrawLine(dpy, w, gc, fh2+wid+fh4, fh+fh4, fh2+wid+fh4, 3*fh);
    XDrawLine(dpy, w, gc, fh2+wid+fh4, 3*fh,   toff-fh4, 3*fh);
    XDrawLine(dpy, w, gc, fh2+2*wid+fh4, fh+fh4, fh2+2*wid+fh4, 4*fh);
    XDrawLine(dpy, w, gc, fh2+2*wid+fh4, 4*fh, toff-fh4, 4*fh);

    XSetForeground(dpy, gc, BlackPixelOfScreen(XtScreen(*buttons)));
    XDrawString(dpy, w, gc, toff, fh+fh2, mode_string(), mode_string.len());
    clString b1_string(mouse_handlers[modifier_mask][0]?
		       mouse_handlers[modifier_mask][0]->title
		       :clString(""));
    clString b2_string(mouse_handlers[modifier_mask][1]?
		       mouse_handlers[modifier_mask][1]->title
		       :clString(""));
    clString b3_string(mouse_handlers[modifier_mask][2]?
		       mouse_handlers[modifier_mask][2]->title
		       :clString(""));
    XDrawString(dpy, w, gc, toff, 2*fh+fh2, b1_string(), b1_string.len());
    XDrawString(dpy, w, gc, toff, 3*fh+fh2, b2_string(), b2_string.len());
    XDrawString(dpy, w, gc, toff, 4*fh+fh2, b3_string(), b3_string.len());
#ifdef OLDUI
    evl->unlock();
#endif
}

void Roe::redraw_buttons(CallbackData*, void*)
{
    buttons_exposed=1;
    update_modifier_widget();
}

void Roe::attach_dials(CallbackData*, void* ud)
{
    int which=(int)ud;
    switch(which){
    case 0:
	Dialbox::attach_dials(dbcontext_st);
	break;
    }
}

void Roe::DBscale(DBContext*, int, double value, double delta, void*)
{
    double oldvalue=value-delta;
    double f=Exp10(value)/Exp10(oldvalue);
    BBox bb;
    get_bounds(bb);
    Point p(0,0,0);
    if(bb.valid())
	p=bb.center();
    mtnScl*=f;
    scale(Vector(f, f, f), p);
    need_redraw=1;
}

void Roe::DBtranslate(DBContext*, int, double, double delta,
		      void* cbdata)
{
    int which=(int)cbdata;
    Vector v(0,0,0);
    switch(which){
    case 0:
	v.x(delta);
	break;
    case 1:
	v.y(delta);
	break;
    case 2:
	v.z(delta);
	break;
    }
    translate(v);
    need_redraw=1;
}

void Roe::DBrotate(DBContext*, int, double, double delta,
		      void* cbdata)
{
    int which=(int)cbdata;
    Vector v(0,0,0);
    switch(which){
    case 0:
	v=Vector(1,0,0);
	break;
    case 1:
	v=Vector(0,1,0);
	break;
    case 2:
	v=Vector(0,0,1);
	break;
    }
    BBox bb;
    get_bounds(bb);
    Point p(0,0,0);
    if(bb.valid())
	p=bb.center();
    rotate_obj(delta, v, p);
    need_redraw=1;
}
#endif

void Roe::redraw_if_needed(int always)
{
    NOT_FINISHED("Roe::redraw_if_needed");
#ifdef OLDUI
    if(need_redraw || always){
	need_redraw=0;
	redrawAll();
    }
    for (int i=0; i<kids.size(); i++)
	kids[i]->redraw_if_needed(always);
#endif
}

#ifdef OLDUI
void Roe::redraw_perf(CallbackData*, void*)
{
    evl->lock();
    Window w=XtWindow(*perf);
    Display* dpy=XtDisplay(*buttons);
    XClearWindow(dpy, w);
    XSetForeground(dpy, gc, fgcolor->pixel());
    XSetFont(dpy, gc, perffont->font->fid);
    int ascent=perffont->font->ascent;
    int height=ascent+perffont->font->descent;
    XDrawString(dpy, w, gc, 0, ascent, perf_string1(), perf_string1.len());
    XDrawString(dpy, w, gc, 0, height+ascent, perf_string2(), perf_string2.len());
    evl->unlock();
}
#endif

void Roe::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Roe needs a minor command");
	return;
    }
    if(args[1] == "setrenderer"){
	if(args.count() != 6){
	    args.error("setrenderer needs a renderer name, etc");
	    return;
	}
	// See if we already have one like that...
	Renderer* r;
	if(!renderers.lookup(args[2], r)){
	    // Create it...
	    r=Renderer::create(args[2]);
	    if(!r){
		args.error("Unknown renderer!");
		return;
	    }
	    renderers.insert(args[2], r);
	}
	if(current_renderer)
	    current_renderer->hide();
	current_renderer=r;
	args.result(r->create_window(args[3], args[4], args[5]));
    } else if(args[1] == "redraw"){
	// We need to dispatch this one to the remote thread
	// We use an ID string instead of a pointer in case this roe
	// gets killed by the time the redraw message gets dispatched.
	manager->mailbox.send(new RedrawMessage(id));
    } else {
	args.error("Unknown minor command for Roe");
    }
}

void Roe::redraw()
{
    current_renderer->redraw(manager, this);
}
