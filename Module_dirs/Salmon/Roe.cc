
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
#include <Geometry/Transform.h>
#include <Geometry/Vector.h>
#include <Geom/Geom.h>
#include <Geom/PointLight.h>
#include <Math/Trig.h>
#include <iostream.h>
#include <stdio.h>
#include <string.h>

#define MouseStart 0
#define MouseEnd 1
#define MouseMove 2

static const int pick_buffer_size = 512;
static const double pick_window = 5.0;

Roe::Roe(Salmon* s, const clString& id)
: id(id), manager(s), view("view", id, this), shading("shading", id, this),
  homeview(Point(.55, .5, 0), Point(.55, .5, .5), Vector(0,1,0), 25)
{
    view.set(homeview);
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


void Roe::mouse_translate(int action, int x, int y)
{
    switch(action){
    case MouseStart:
	last_x=x;
	last_y=y;
	total_x = 0;
	total_y = 0;
	update_mode_string("translate: ");
	break;
    case MouseMove:
	{
	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    double xmtn=double(last_x-x)/double(xres);
	    double ymtn=-double(last_y-y)/double(yres);
	    last_x = x;
	    last_y = y;
	    // Get rid of roundoff error for the display...
	    if (Abs(total_x) < .001) total_x = 0;
	    if (Abs(total_y) < .001) total_y = 0;

	    View tmpview(view.get());
	    double aspect=double(xres)/double(yres);
	    double znear, zfar;
	    if(!current_renderer->compute_depth(this, tmpview, znear, zfar))
		return; // No objects...
	    double zmid=(znear+zfar)/2.;
	    Vector u,v;
	    tmpview.get_viewplane(aspect, zmid, u, v);
	    double ul=u.length();
	    double vl=v.length();
	    Vector trans(u*xmtn+v*ymtn);

	    total_x+=ul*xmtn;
	    total_y+=vl*ymtn;

	    // Translate the view...
	    tmpview.eyep+=trans;
	    tmpview.lookat+=trans;

	    // Put the view back...
	    view.set(tmpview);

	    need_redraw=1;
	    update_mode_string(clString("translate: ")+to_string(total_x)
			       +", "+to_string(total_y));
	}
	break;
    case MouseEnd:
	update_mode_string("");
	break;
    }
}

void Roe::mouse_scale(int action, int x, int y)
{
    switch(action){
    case MouseStart:
	{
	    update_mode_string("scale: ");
	    last_x=x;
	    last_y=y;
	    total_scale=1;
	}
	break;
    case MouseMove:
	{
	    double scl;
	    double xmtn=last_x-x;
	    double ymtn=last_y-y;
	    xmtn/=30;
	    ymtn/=30;
	    last_x = x;
	    last_y = y;
	    if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
	    if (scl<0) scl=1/(1-scl); else scl+=1;
	    total_scale*=scl;

	    View tmpview(view.get());
	    tmpview.fov=RtoD(2*Atan(scl*Tan(DtoR(tmpview.fov/2.))));
	    view.set(tmpview);
	    need_redraw=1;
	    update_mode_string(clString("scale: ")+to_string(total_x*100)+"%");
	}
	break;
    case MouseEnd:
	update_mode_string("");
	break;
    }	
}

void Roe::mouse_rotate(int action, int x, int y)
{
    switch(action){
    case MouseStart:
	{
	    update_mode_string("rotate:");
	    last_x=x;
	    last_y=y;

	    // Find the center of rotation...
	    View tmpview(view.get());
	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    double aspect=double(xres)/double(yres);
	    double znear, zfar;
	    rot_point_valid=0;
	    if(!current_renderer->compute_depth(this, tmpview, znear, zfar))
		return; // No objects...
	    double zmid=(znear+zfar)/2.;

	    Point ep(0, 0, zmid);
	    rot_point=tmpview.eyespace_to_objspace(ep, aspect);
	    rot_view=tmpview;
	    rot_point_valid=1;
	    cerr << "rot_point=" << rot_point << endl;
	}
	break;
    case MouseMove:
	{
	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    double xmtn=double(last_x-x)/double(xres);
	    double ymtn=double(last_y-y)/double(yres);

	    double xrot=xmtn*360.0;
	    double yrot=ymtn*360.0;

	    if(!rot_point_valid)
		break;
	    // Rotate the scene about the rot_point
	    Transform transform;
	    Vector transl(Point(0,0,0)-rot_point);
	    transform.pre_translate(transl);
	    View tmpview(rot_view);
	    Vector u,v;
	    double aspect=double(xres)/double(yres);
	    tmpview.get_viewplane(aspect, 1, u, v);
	    u.normalize();
	    v.normalize();
	    transform.pre_rotate(DtoR(yrot), u);
	    transform.pre_rotate(DtoR(xrot), v);
	    transform.pre_translate(-transl);

	    Point top(tmpview.eyep+tmpview.up);
	    top=transform.project(top);
	    tmpview.eyep=transform.project(tmpview.eyep);
	    tmpview.lookat=transform.project(tmpview.lookat);
	    tmpview.up=top-tmpview.eyep;

	    view.set(tmpview);

	    need_redraw=1;
	    update_mode_string("rotate:");
	}
	break;
    case MouseEnd:
	update_mode_string("");
	break;
    }
}

void Roe::mouse_pick(int action, int x, int y)
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

void Roe::redraw_if_needed()
{
    if(need_redraw){
	need_redraw=0;
	redraw();
    }
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

void Roe::tcl_command(TCLArgs& args, void*)
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
    } else if(args[1] == "mtranslate"){
	do_mouse(&Roe::mouse_translate, args);
    } else if(args[1] == "mrotate"){
	do_mouse(&Roe::mouse_rotate, args);
    } else if(args[1] == "mscale"){
	do_mouse(&Roe::mouse_scale, args);
    } else if(args[1] == "mpick"){
	do_mouse(&Roe::mouse_pick, args);
    } else if(args[1] == "sethome"){
	homeview=view.get();
    } else if(args[1] == "gohome"){
	view.set(homeview);
	manager->mailbox.send(new RedrawMessage(id));
    } else {
	args.error("Unknown minor command for Roe");
    }
}

void Roe::do_mouse(MouseHandler handler, TCLArgs& args)
{
    if(args.count() != 5){
	args.error(args[1]+" needs start/move/end and x y");
	return;
    }
    int action;
    if(args[2] == "start"){
	action=MouseStart;
    } else if(args[2] == "end"){
	action=MouseEnd;
    } else if(args[2] == "move"){
	action=MouseMove;
    } else {
	args.error("Unknown mouse action");
	return;
    }
    int x,y;
    if(!args[3].get_int(x)){
	args.error("error parsing x");
	return;
    }
    if(!args[4].get_int(y)){
	args.error("error parsing y");
	return;
    }
    // We have to send this to the salmon thread...
    manager->mailbox.send(new RoeMouseMessage(id, handler, action, x, y));
}

void Roe::redraw()
{
    need_redraw=0;
    reset_vars();
    current_renderer->redraw(manager, this);
}

void Roe::update_mode_string(const clString& msg)
{
    NOT_FINISHED("Roe::update_mode_string");
}

TCLView::TCLView(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), eyep("eyep", str(), tcl),
  lookat("lookat", str(), tcl), up("up", str(), tcl),
  fov("fov", str(), tcl)
{
}

TCLView::~TCLView()
{
}

View TCLView::get()
{
    return View(eyep.get(), lookat.get(), up.get(), fov.get());
}

void TCLView::set(const View& view)
{
    eyep.set(view.eyep);
    lookat.set(view.lookat);
    up.set(view.up);
    fov.set(view.fov);
}

RoeMouseMessage::RoeMouseMessage(const clString& rid, MouseHandler handler,
				 int action, int x, int y)
: MessageBase(MessageTypes::RoeMouse), rid(rid), handler(handler),
  action(action), x(x), y(y)
{
}

RoeMouseMessage::~RoeMouseMessage()
{
}
