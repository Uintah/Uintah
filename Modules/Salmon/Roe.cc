
/*
 *  Roe.cc:  The Geometry Viewer Window
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
#include <Geom/Pick.h>
#include <Geom/PointLight.h>
#include <Math/Trig.h>
#include <TCL/TCLTask.h>
#include <iostream.h>
#include <stdio.h>
#include <string.h>
#include <strstream.h>

#define MouseStart 0
#define MouseEnd 1
#define MouseMove 2

const int MODEBUFSIZE = 100;

Roe::Roe(Salmon* s, const clString& id)
: id(id), manager(s), view("view", id, this), shading("shading", id, this),
  homeview(Point(.55, .5, 0), Point(.55, .5, .5), Vector(0,1,0), 25)
{
    view.set(homeview);
    TCL::add_command(id, this, 0);
    current_renderer=0;
    modebuf=new char[MODEBUFSIZE];
    modecommand=new char[MODEBUFSIZE];

    // Fill in the visibility database...
    HashTableIter<int,HashTable<int, SceneItem*>*> iter(&manager->portHash);
    for (iter.first(); iter.ok(); ++iter) {
	HashTable<int, SceneItem*>* serHash=iter.get_data();
	HashTableIter<int, SceneItem*> serIter(serHash);
	for (serIter.first(); serIter.ok(); ++serIter) {
	    SceneItem *si=serIter.get_data();
	    itemAdded(si);
	}
    }
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

#endif

void Roe::itemAdded(SceneItem* si)
{
    TCLvarint* vis;
    if(!visible.lookup(si->name, vis)){
	// Make one...
	vis=new TCLvarint(si->name, id, this);
	vis->set(1);
	visible.insert(si->name, vis);
	NOT_FINISHED("Add items to TCL listbox...");
    }
    // invalidate the bounding box
    bb.reset();
    need_redraw=1;
}

void Roe::itemDeleted(SceneItem *si)
{
    NOT_FINISHED("Roe::itemDeleted");
    // invalidate the bounding box
    bb.reset();
    need_redraw=1;
}

// need to fill this in!   
#ifdef OLDUI
void Roe::itemCB(CallbackData*, void *gI) {
    GeomItem *g = (GeomItem *)gI;
    g->vis = !g->vis;
    need_redraw=1;
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
    delete[] modebuf;
    delete[] modecommand;
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
    bbox.reset();
    HashTableIter<int,HashTable<int, SceneItem*>*> iter(&manager->portHash);
    for (iter.first(); iter.ok(); ++iter) {
	HashTable<int, SceneItem*>* serHash=iter.get_data();
	HashTableIter<int, SceneItem*> serIter(serHash);
	for (serIter.first(); serIter.ok(); ++serIter) {
	    SceneItem *si=serIter.get_data();
	    // Look up the name to see if it should be drawn...
	    TCLvarint* vis;
	    if(visible.lookup(si->name, vis)){
		if(vis->get())
		    si->obj->get_bounds(bbox);
	    } else {
		cerr << "Warning: object " << si->name << " not in visibility database...\n";
		si->obj->get_bounds(bbox);
	    }
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
	    ostrstream str(modebuf, MODEBUFSIZE);
	    str << "translate: " << total_x << ", " << total_y;
	    update_mode_string(str.str());
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
	    ostrstream str(modebuf, MODEBUFSIZE);
	    str << "scale: " << total_x*100 << "%";
	    update_mode_string(str.str());
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
    switch(action){
    case MouseStart:
	{
	    total_x=0;
	    total_y=0;
	    total_z=0;
	    last_x=x;
	    last_y=y;
	    current_renderer->get_pick(manager, this, x, y,
				       pick_obj, pick_pick);

	    if (pick_obj){
		NOT_FINISHED("update mode string for pick");
		pick_pick->pick();
	    } else {
		update_mode_string("pick: none");
	    }
	}
	break;
    case MouseMove:
	{
	    if (!pick_obj || !pick_pick) break;
	    double xmtn=last_x-x;
	    double ymtn=last_y-y;
// project the center of the item grabbed onto the screen -- take the z
// component and unprojec the last and current x, y locations to get a 
// vector in object space.
	    BBox itemBB;
	    pick_obj->get_bounds(itemBB);
	    View tmpview(view.get());
	    Point cen(itemBB.center());
	    double depth=tmpview.depth(cen);

	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    Point p1(2*x/double(xres)-1, 2*y/double(yres)-1, depth);
	    Point p0(2*last_x/double(xres)-1, 2*last_y/double(yres)-1, depth);
	    double aspect=double(xres)/double(yres);
	    p1=tmpview.eyespace_to_objspace(p1, aspect);
	    p0=tmpview.eyespace_to_objspace(p0, aspect);
	    Vector dir(p1-p0);
	    double dist=dir.normalize();

	    double maxdot=0;
	    int prin_dir=-1;
	    for (int i=0; i<pick_pick->nprincipal(); i++) {
		double pdot=Dot(dir, pick_pick->principal(i));
		if(pdot > maxdot){
		    maxdot=pdot;
		    prin_dir=i;
		}
	    }
	    if(prin_dir != -1){
		Vector mtn(pick_pick->principal(prin_dir)*dist);
		total_x+=mtn.x();
		total_y+=mtn.y();
		total_z+=mtn.z();
		if (Abs(total_x) < .0001) total_x=0;
		if (Abs(total_y) < .0001) total_y=0;
		if (Abs(total_z) < .0001) total_z=0;
		need_redraw=1;
		update_mode_string("picked someting...");
		pick_pick->moved(prin_dir, dist, mtn);
	    } else {
		update_mode_string("Bad direction...");
	    }
	    last_x = x;
	    last_y = y;
	}
	break;
    case MouseEnd:
	if(pick_pick){
	    pick_pick->release();
	}
	pick_pick=0;
	pick_obj=0;
	update_mode_string("");
	break;
    }
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
	if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	    cerr << "Redraw event dropped, mailbox full!\n";
	} else {
	    manager->mailbox.send(new RedrawMessage(id));
	}
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
    if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	cerr << "Mouse event dropped, mailbox full!\n";
    } else {
	manager->mailbox.send(new RoeMouseMessage(id, handler, action, x, y));
    }
}

void Roe::redraw()
{
    need_redraw=0;
    reset_vars();
    current_renderer->redraw(manager, this);
}

void Roe::update_mode_string(const char* msg)
{
    ostrstream str(modecommand, MODEBUFSIZE);    
    str << "updateMode " << id << " \"" << msg << "\"";
    TCL::execute(str.str());
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
    TCLTask::lock();
    View v(eyep.get(), lookat.get(), up.get(), fov.get());
    TCLTask::unlock();
    return v;
}

void TCLView::set(const View& view)
{
    TCLTask::lock();
    eyep.set(view.eyep);
    lookat.set(view.lookat);
    up.set(view.up);
    fov.set(view.fov);
    TCLTask::unlock();
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
