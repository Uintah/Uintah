

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
#include <Modules/Salmon/Ball.h>
#include <Modules/Salmon/BallMath.h>
#include <Classlib/Debug.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <Classlib/Pstreams.h>
#include <Geometry/BBox.h>
#include <Geometry/Transform.h>
#include <Geometry/Vector.h>
#include <Geom/Geom.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/Pick.h>
#include <Geom/PointLight.h>
#include <Geom/Scene.h>
#include <Geom/Sphere.h>
#include <Malloc/Allocator.h>
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
static DebugSwitch autoview_sw("Roe", "autoview");

Roe::Roe(Salmon* s, const clString& id)
: manager(s),
  view("view", id, this),
  homeview(Point(.55, .5, 0), Point(.0, .0, .0), Vector(0,1,0), 25),
  bgcolor("bgcolor", id, this), shading("shading", id, this),
  do_stereo("do_stereo", id, this), drawimg("drawimg", id, this),
  tracker_state("tracker_state", id, this),
  id(id)
{
    inertia_mode=0;
    bgcolor.set(Color(0,0,0));
    view.set(homeview);
    TCL::add_command(id+"-c", this, 0);
    current_renderer=0;
    modebuf=scinew char[MODEBUFSIZE];
    modecommand=scinew char[MODEBUFSIZE];
    maxtag=0;
    tracker=0;
    mouse_obj=0;
    ball = new BallData();
    ball->Init();
}

void Roe::itemAdded(GeomSalmonItem* si)
{
    ObjTag* vis;
    if(!visible.lookup(si->name, vis)){
	// Make one...
	vis=scinew ObjTag;
	vis->visible=scinew TCLvarint(si->name, id, this);
	vis->visible->set(1);
	vis->tagid=maxtag++;
	visible.insert(si->name, vis);
	char buf[1000];
	ostrstream str(buf, 1000);
	str << id << " addObject " << vis->tagid << " \"" << si->name << "\"" << '\0';
	TCL::execute(str.str());
    } else {
	char buf[1000];
	ostrstream str(buf, 1000);
	str << id << " addObject2 " << vis->tagid << '\0';
	TCL::execute(str.str());
    }
    // invalidate the bounding box
    bb.reset();
    need_redraw=1;
}

void Roe::itemDeleted(GeomSalmonItem *si)
{
    ObjTag* vis;
    if(!visible.lookup(si->name, vis)){
	cerr << "Where did that object go???" << endl;
    } else {
	char buf[1000];
	ostrstream str(buf, 1000);
	str << id << " removeObject " << vis->tagid << '\0';
	TCL::execute(str.str());
    }
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

  kids.add(scinew Roe(manager, mat, mtnScl));
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


void Roe::get_bounds(BBox& bbox)
{
    bbox.reset();
//    HashTableIter<int, PortInfo*> iter(&manager->portHash);
    HashTableIter<int,GeomObj*> iter = manager->ports.getIter();
    for (iter.first(); iter.ok(); ++iter) {
//	HashTable<int, SceneItem*>* serHash=iter.get_data()->objs;
//	HashTableIter<int, SceneItem*> serIter(serHash);
//	HashTable<int,GeomObj*>* serHash = 
//	    ((GeomSalmonPort*)iter.get_data())->getHashPtr();
	HashTableIter<int,GeomObj*> serIter = 
	    ((GeomSalmonPort*)iter.get_data())->getIter();
	// items in the scen are all GeomSalmonItem's...
	for (serIter.first(); serIter.ok(); ++serIter) {
	    GeomSalmonItem *si=(GeomSalmonItem*)serIter.get_data();
	    // Look up the name to see if it should be drawn...
	    ObjTag* vis;
	    if(visible.lookup(si->name, vis)){
		if(vis->visible->get()){
		    if(si->lock)
			si->lock->read_lock();
		    si->get_bounds(bbox);
		    if(si->lock)
			si->lock->read_unlock();
		}
	    } else {
		cerr << "Warning: object " << si->name << " not in visibility database...\n";
		si->get_bounds(bbox);
	    }
	}
    }
}

void Roe::rotate(double /*angle*/, Vector /*v*/, Point /*c*/)
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

void Roe::rotate_obj(double /*angle*/, const Vector& /*v*/, const Point& /*c*/)
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

void Roe::translate(Vector /*v*/)
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

void Roe::scale(Vector /*v*/, Point /*c*/)
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


void Roe::mouse_translate(int action, int x, int y, int, int, int)
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
	    tmpview.eyep(tmpview.eyep()+trans);
	    tmpview.lookat(tmpview.lookat()+trans);

	    // Put the view back...
	    view.set(tmpview);

	    need_redraw=1;
	    ostrstream str(modebuf, MODEBUFSIZE);
	    str << "translate: " << total_x << ", " << total_y << '\0';
	    update_mode_string(str.str());
	}
	break;
    case MouseEnd:
	update_mode_string("");
	break;
    }
}

void Roe::mouse_scale(int action, int x, int y, int, int, int)
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
	    tmpview.fov(RtoD(2*Atan(scl*Tan(DtoR(tmpview.fov()/2.)))));

	    view.set(tmpview);
	    need_redraw=1;
	    ostrstream str(modebuf, MODEBUFSIZE);
	    str << "scale: " << total_x*100 << "%" << '\0';
	    update_mode_string(str.str());
	}
	break;
    case MouseEnd:
	update_mode_string("");
	break;
    }	
}

void Roe::mouse_rotate(int action, int x, int y, int, int, int time)
{
    switch(action){
    case MouseStart:
	{
	    if(inertia_mode){
		inertia_mode=0;
		redraw();
	    }
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

	    rot_point = tmpview.lookat();
	    rot_view=tmpview;
	    rot_point_valid=1;

	    double rad = 0.8;
	    HVect center(0,0,0,1.0);
	
	    // we also want to keep the old transform information
	    // around (so stuff correlates correctly)
	    // OGL uses left handed coordinate system!
	
	    Vector z_axis,y_axis,x_axis;

	    y_axis = tmpview.up();
	    z_axis = tmpview.eyep() - tmpview.lookat();
	    eye_dist = z_axis.normalize();
	    x_axis = Cross(y_axis,z_axis);
	    x_axis.normalize();
	    y_axis = Cross(z_axis,x_axis);
	    y_axis.normalize();
	    tmpview.up(y_axis); // having this correct could fix something?

	    prev_trans.load_frame(Point(0.0,0.0,0.0),x_axis,y_axis,z_axis);

	    ball->Init();
	    ball->Place(center,rad);
	    HVect mouse((2.0*x)/xres - 1.0,2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
	    ball->Mouse(mouse);
	    ball->BeginDrag();

	    prev_time[0] = time;
	    prev_quat[0] = mouse;
	    prev_time[1] = prev_time[2] = -100;
	    ball->Update();
	    last_time=time;
	    inertia_mode=0;
	    need_redraw = 1;
	}
	break;
    case MouseMove:
	{
	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    //double aspect=double(xres)/double(yres);

	    if(!rot_point_valid)
		break;

	    HVect mouse((2.0*x)/xres - 1.0,2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
	    prev_time[2] = prev_time[1];
	    prev_time[1] = prev_time[0];
	    prev_time[0] = time;
	    ball->Mouse(mouse);
	    ball->Update();

	    prev_quat[2] = prev_quat[1];
	    prev_quat[1] = prev_quat[0];
	    prev_quat[0] = mouse;

	    // now we should just sendthe view points through
	    // the rotation (after centerd around the ball)
	    // eyep lookat and up

	    View tmpview(rot_view);

	    Transform tmp_trans;
	    HMatrix mNow;
	    ball->Value(mNow);
	    tmp_trans.set(&mNow[0][0]);

	    Transform prv = prev_trans;
	    prv.post_trans(tmp_trans);

	    HMatrix vmat;
	    prv.get(&vmat[0][0]);

	    Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
	    Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);

	    tmpview.up(y_a.vector());
	    tmpview.eyep((z_a*(eye_dist)) + tmpview.lookat().vector());

	    view.set(tmpview);
	    need_redraw=1;
	    update_mode_string("rotate:");

	    last_time=time;
	    inertia_mode=0;
	}
	break;
    case MouseEnd:
	if(time-last_time < 20){
	    // now setup the normalized quaternion
 

	    View tmpview(rot_view);
	    
	    Transform tmp_trans;
	    HMatrix mNow;
	    ball->Value(mNow);
	    tmp_trans.set(&mNow[0][0]);
	    
	    Transform prv = prev_trans;
	    prv.post_trans(tmp_trans);
	    
	    HMatrix vmat;
	    prv.get(&vmat[0][0]);
	    
	    Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
	    Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	    
	    tmpview.up(y_a.vector());
	    tmpview.eyep((z_a*(eye_dist)) + tmpview.lookat().vector());
	    
	    view.set(tmpview);
	    prev_trans = prv;

	    // now you need to use the history to 
	    // set up the arc you want to use...

	    ball->Init();
	    double rad = 0.8;
	    HVect center(0,0,0,1.0);

	    ball->Place(center,rad);

	    int index=2;

	    if (prev_time[index] == -100)
		index = 1;

	    ball->vDown = prev_quat[index];
	    ball->vNow  = prev_quat[0];
	    ball->dragging = 1;
	    ball->Update();
	    
	    ball->qNorm = ball->qNow.Conj();
	    double mag = ball->qNow.VecMag();

	    // Go into inertia mode...
	    inertia_mode=1;
	    need_redraw=1;

	    if (mag < 0.00001) { // arbitrary ad-hoc threshold
		inertia_mode = 0;
		need_redraw = 1;
		cerr << mag << " " << prev_time[0] - prev_time[index] << endl;
	    }
	    else {
		double c = 1.0/mag;
		double dt = prev_time[0] - prev_time[index];// time between last 2 events
		ball->qNorm.x *= c;
		ball->qNorm.y *= c;
		ball->qNorm.z *= c;
		angular_v = 2*acos(ball->qNow.w)*1000.0/dt;
		cerr << dt << endl;
	    }
	} else {
	    inertia_mode=0;
	}
	ball->EndDrag();
	rot_point_valid = 0; // so we don't have to draw this...
	need_redraw = 1;     // always update this...
	update_mode_string("");
	break;
    }
}

void Roe::mouse_pick(int action, int x, int y, int state, int btn, int)
{
    BState bs;
    bs.shift=1; // Always for widgets...
    bs.control= ((state&4)!=0);
    bs.alt= ((state&8)!=0);
    bs.btn=btn;
    switch(action){
    case MouseStart:
	{
	    total_x=0;
	    total_y=0;
	    total_z=0;
	    last_x=x;
	    last_y=current_renderer->yres-y;
	    current_renderer->get_pick(manager, this, x, y,
				       pick_obj, pick_pick);

	    if (pick_obj){
		NOT_FINISHED("update mode string for pick");
		pick_pick->pick(this,bs);
		need_redraw=1;
	    } else {
		update_mode_string("pick: none");
	    }
	}
	break;
    case MouseMove:
	{
	    if (!pick_obj || !pick_pick) break;
// project the center of the item grabbed onto the screen -- take the z
// component and unprojec the last and current x, y locations to get a 
// vector in object space.
	    y=current_renderer->yres-y;
	    BBox itemBB;
	    pick_obj->get_bounds(itemBB);
	    View tmpview(view.get());
	    Point cen(itemBB.center());
	    double depth=tmpview.depth(cen);
	    Vector u,v;
	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    double aspect=double(xres)/double(yres);
	    tmpview.get_viewplane(aspect, depth, u, v);
	    int dx=x-last_x;
	    int dy=y-last_y;
	    double ndx=(2*dx/(double(xres)-1));
	    double ndy=(2*dy/(double(yres)-1));
	    Vector motionv(u*ndx+v*ndy);

	    double maxdot=0;
	    int prin_dir=-1;
	    for (int i=0; i<pick_pick->nprincipal(); i++) {
		double pdot=Dot(motionv, pick_pick->principal(i));
		if(pdot > maxdot){
		    maxdot=pdot;
		    prin_dir=i;
		}
	    }
	    if(prin_dir != -1){
		double dist=motionv.length();
		Vector mtn(pick_pick->principal(prin_dir)*dist);
		total_x+=mtn.x();
		total_y+=mtn.y();
		total_z+=mtn.z();
		if (Abs(total_x) < .0001) total_x=0;
		if (Abs(total_y) < .0001) total_y=0;
		if (Abs(total_z) < .0001) total_z=0;
		need_redraw=1;
		update_mode_string("picked someting...");
		pick_pick->moved(prin_dir, dist, mtn, bs);
		need_redraw=1;
	    } else {
		update_mode_string("Bad direction...");
	    }
	    last_x = x;
	    last_y = y;
	}
	break;
    case MouseEnd:
	if(pick_pick){
	    pick_pick->release(bs);
	    need_redraw=1;
	}
	pick_pick=0;
	pick_obj=0;
	update_mode_string("");
	break;
    }
}

void Roe::redraw_if_needed()
{
    if(need_redraw){
	need_redraw=0;
	redraw();
    }
}

void Roe::tcl_command(TCLArgs& args, void*)
{
    if(args.count() < 2){
	args.error("Roe needs a minor command");
	return;
    }
    if(args[1] == "dump_roe"){
	if(args.count() != 3){
	    args.error("Roe::dump_roe needs an output file name!");
	    return;
	}
	// We need to dispatch this one to the remote thread
	// We use an ID string instead of a pointer in case this roe
	// gets killed by the time the redraw message gets dispatched.
	if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	    cerr << "Redraw event dropped, mailbox full!\n";
	} else {
	    manager->mailbox.send(scinew SalmonMessage(MessageTypes::RoeDumpImage, id, args[2]));
	}
    } else if(args[1] == "startup"){
	// Fill in the visibility database...
	//HashTableIter<int, PortInfo*> iter(&manager->portHash);
	HashTableIter<int,GeomObj*> iter = manager->ports.getIter();
	for (iter.first(); iter.ok(); ++iter) {
//	    HashTable<int, SceneItem*>* serHash=iter.get_data()->objs;
//	    HashTableIter<int, SceneItem*> serIter(serHash);
	    HashTableIter<int,GeomObj*> serIter = 
		((GeomSalmonPort*)iter.get_data())->getIter();	    
	    for (serIter.first(); serIter.ok(); ++serIter) {
		GeomSalmonItem *si=(GeomSalmonItem*)serIter.get_data();
		itemAdded(si);
	    }
	}
    } else if(args[1] == "setrenderer"){
	if(args.count() != 6){
	    args.error("setrenderer needs a renderer name, etc");
	    return;
	}
	Renderer* r=get_renderer(args[2]);
	if(!r){
	    args.error("Unknown renderer!");
	    return;
	}
	if(current_renderer)
	    current_renderer->hide();
	current_renderer=r;
	args.result(r->create_window(this, args[3], args[4], args[5]));
    } else if(args[1] == "redraw"){
	// We need to dispatch this one to the remote thread
	// We use an ID string instead of a pointer in case this roe
	// gets killed by the time the redraw message gets dispatched.
	if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	    cerr << "Redraw event dropped, mailbox full!\n";
	} else {
	    manager->mailbox.send(scinew SalmonMessage(id));
	}
    } else if(args[1] == "anim_redraw"){
	// We need to dispatch this one to the remote thread
	// We use an ID string instead of a pointer in case this roe
	// gets killed by the time the redraw message gets dispatched.
	if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	    cerr << "Redraw event dropped, mailbox full!\n";
	} else {
	    if(args.count() != 6){
		args.error("anim_redraw wants tbeg tend nframes framerate");
		return;
	    }
	    double tbeg;
	    if(!args[2].get_double(tbeg)){
		args.error("Can't figure out tbeg");
		return;
	    } 
	    double tend;
	    if(!args[3].get_double(tend)){
		args.error("Can't figure out tend");
		return;
	    }
	    int nframes;
	    if(!args[4].get_int(nframes)){
		args.error("Can't figure out nframes");
		return;
	    }
	    double framerate;
	    if(!args[5].get_double(framerate)){
		args.error("Can't figure out framerate");
		return;
	    }	    
	    manager->mailbox.send(scinew SalmonMessage(id, tbeg, tend,
						    nframes, framerate));
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
	manager->mailbox.send(scinew SalmonMessage(id)); // Redraw
    } else if(args[1] == "autoview"){
	BBox bbox;
	get_bounds(bbox);
	autoview(bbox);
    } else if(args[1] == "dolly"){
	if(args.count() != 3){
	    args.error("dolly needs an amount");
	    return;
	}
	double amount;
	if(!args[2].get_double(amount)){
	    args.error("Can't figure out amount");
	    return;
	}
	View cv(view.get());
	Vector lookdir(cv.eyep()-cv.lookat());
	lookdir*=amount;
	cv.eyep(cv.lookat()+lookdir);
	animate_to_view(cv, 1.0);
    } else if(args[1] == "tracker"){
	if(tracker_state.get()){
	    // Turn the tracker on...
	    if(!tracker){
		tracker=new Tracker(&manager->mailbox, (void*)this);
	    }
	    have_trackerdata=0;
	} else {
	    // Turn the tracker off...
	    if(tracker){
		delete tracker;
		tracker=0;
	    }
	}
    } else if(args[1] == "reset_tracker"){
	have_trackerdata=0;
    } else if(args[1] == "saveobj") {
	if(args.count() != 4){
	    args.error("Roe::dump_roe needs an output file name and format!");
	    return;
	}
	// We need to dispatch this one to the remote thread
	// We use an ID string instead of a pointer in case this roe
	// gets killed by the time the redraw message gets dispatched.
	if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	    cerr << "Redraw event dropped, mailbox full!\n";
	} else {
	    manager->mailbox.send(scinew SalmonMessage(MessageTypes::RoeDumpObjects,
						       id, args[2], args[3]));
	}
    } else {
	args.error("Unknown minor command for Roe");
    }
}

void Roe::do_mouse(MouseHandler handler, TCLArgs& args)
{
    if(args.count() != 5 && args.count() != 7 && args.count() != 8 && args.count() != 6){
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
    int state;
    int btn;
    if(args.count() == 7){
       if(!args[5].get_int(state)){
	  args.error("error parsing state");
	  return;

       }
       if(!args[6].get_int(btn)){
	  args.error("error parsing btn");
	  return;
       }
    }
    int time;
    if(args.count() == 8){
	if(!args[7].get_int(time)){
	   args.error("err parsing time");
	   return;
       }
    }
    if(args.count() == 6){
	if(!args[5].get_int(time)){
	   args.error("err parsing time");
	   return;
       }
    }

    // We have to send this to the salmon thread...
    if(manager->mailbox.nitems() >= manager->mailbox.size()-1){
	cerr << "Mouse event dropped, mailbox full!\n";
    } else {
	manager->mailbox.send(scinew RoeMouseMessage(id, handler, action, x, y, state, btn, time));
    }
}

void Roe::autoview(const BBox& bbox)
{
    if(bbox.valid()){
        View cv(view.get());
        // Animate lookat point to center of BBox...
        cv.lookat(bbox.center());
        animate_to_view(cv, 2.0);
        
        // Move forward/backwards until entire view is in scene...

	// change this a little, make it so that the FOV must
	// be 60 deg...

        Vector diag(bbox.diagonal());
	double w=diag.length();
	Vector lookdir(cv.lookat()-cv.eyep()); 
	lookdir.normalize();
	const double scale = 1.0/(2*Tan(DtoR(60.0/2.0)));
	double length = w*scale;
	cv.fov(60.0);
	cv.eyep(cv.lookat() - lookdir*length);
        animate_to_view(cv, 2.0);

    }
}

void Roe::redraw()
{
    need_redraw=0;
    reset_vars();

    // Get animation variables
    double ct;
    if(!get_tcl_doublevar(id, "current_time", ct)){
	manager->error("Error reading current_time");
	return;
    }
    current_renderer->redraw(manager, this, ct, ct, 1, 0);
}

void Roe::redraw(double tbeg, double tend, int nframes, double framerate)
{
    need_redraw=0;
    reset_vars();

    // Get animation variables
    current_renderer->redraw(manager, this, tbeg, tend, nframes, framerate);
}

void Roe::update_mode_string(const char* msg)
{
    ostrstream str(modecommand, MODEBUFSIZE);    
    str << id << " updateMode \"" << msg << "\"" << '\0';
    TCL::execute(str.str());
}

TCLView::TCLView(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), eyep("eyep", str(), tcl),
  lookat("lookat", str(), tcl), up("up", str(), tcl),
  fov("fov", str(), tcl), eyep_offset("eyep_offset", str(), tcl)
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
    eyep.set(view.eyep());
    lookat.set(view.lookat());
    up.set(view.up());
    fov.set(view.fov());
    TCLTask::unlock();
}

void TCLView::emit(ostream& out)
{
    eyep.emit(out);
    lookat.emit(out);
    up.emit(out);
    fov.emit(out);
}

RoeMouseMessage::RoeMouseMessage(const clString& rid, MouseHandler handler,
				 int action, int x, int y, int state, int btn,
				 int time)
: MessageBase(MessageTypes::RoeMouse), rid(rid), handler(handler),
  action(action), x(x), y(y), state(state), btn(btn), time(time)
{
}

RoeMouseMessage::~RoeMouseMessage()
{
}

void Roe::animate_to_view(const View& v, double /*time*/)
{
    NOT_FINISHED("Roe::animate_to_view");
    view.set(v);
    manager->mailbox.send(scinew SalmonMessage(id));
}

Renderer* Roe::get_renderer(const clString& name)
{
    // See if we already have one like that...
    Renderer* r;
    if(!renderers.lookup(name, r)){
	// Create it...
	r=Renderer::create(name);
	if(r)
	    renderers.insert(name, r);
    }
    return r;
}

void Roe::force_redraw()
{
    need_redraw=1;
}

void Roe::do_for_visible(Renderer* r, RoeVisPMF pmf)
{
    // Do internal objects first...
    for(int i=0;i<roe_objs.size();i++){
	(r->*pmf)(manager, this, roe_objs[i]);
    }

//    HashTableIter<int, PortInfo*> iter(&manager->portHash);
    HashTableIter<int,GeomObj*> iter = manager->ports.getIter();
    for (iter.first(); iter.ok(); ++iter) {
//	HashTable<int, SceneItem*>* serHash=iter.get_data()->objs;
//	HashTableIter<int, SceneItem*> serIter(serHash);
	HashTableIter<int,GeomObj*> serIter = 
	    ((GeomSalmonPort*)iter.get_data())->getIter();
	
	for (serIter.first(); serIter.ok(); ++serIter) {
	    GeomSalmonItem *si=(GeomSalmonItem*)serIter.get_data();
	    // Look up the name to see if it should be drawn...
	    ObjTag* vis;
	    if(visible.lookup(si->name, vis)){
		if(vis->visible->get()){
		    if(si->lock)
			si->lock->read_lock ();
		    (r->*pmf)(manager, this, si);
		    if(si->lock)
			si->lock->read_unlock();
		}
	    } else {
		cerr << "Warning: object " << si->name << " not in visibility database...\n";
	    }
	}
    }
}

void Roe::set_current_time(double time)
{
    set_tclvar(id, "current_time", to_string(time));
}

// The tracker gives us measurements in 1/1000th of an inch
#define TRACKER_RESOLUTION 0.001
#define TRACKER_DIST 6

void Roe::flyingmouse_moved(const TrackerPosition& pos)
{
    if(pos.out)
	return;
    if(!have_trackerdata)
	return;
    double xdist=(pos.x-old_head_pos.x)*TRACKER_RESOLUTION/TRACKER_DIST;
    double ydist=(pos.y-old_head_pos.y)*TRACKER_RESOLUTION/TRACKER_DIST;
    double zdist=(pos.z-old_head_pos.z)*TRACKER_RESOLUTION/TRACKER_DIST;
    cerr << "MOUSE: xdist=" << xdist << ", ydist=" << ydist << ", zdist=" << zdist << endl;
    mousep=(orig_eye-frame_right*xdist-frame_up*ydist-frame_front*zdist);
    if(!mouse_obj){
	mouse_obj=new GeomSphere(mousep, .1);
	roe_objs.add(mouse_obj);
    } else {
	mouse_obj->move(mousep, .1);
    }
    need_redraw=1;
}


void Roe::head_moved(const TrackerPosition& pos)
{
    if(pos.out)
	return;
    if(!have_trackerdata){
	old_head_pos=pos;
	if(pos.x == 0 && pos.y == 0 && pos.z == 0)
	    return;
	have_trackerdata=1;
	View old_view(view.get());

	Renderer* r=current_renderer;
	double znear;
	double zfar;
	View tview(view.get());
	if(!r->compute_depth(this, tview, znear, zfar))
	    return;
	double zmid=(znear+zfar)/2.;
	double aspect=double(r->xres)/double(r->yres);
	old_view.get_viewplane(aspect, zmid, frame_right, frame_up);
	frame_front=old_view.lookat()-old_view.eyep();
	frame_front.normalize();
	frame_front*=zmid;
	orig_eye=old_view.eyep();
	cerr << "x=" << pos.x << ", y=" << pos.y << ", z=" << pos.z << endl;
	return;
    }
    
    double xdist=(pos.x-old_head_pos.x)*TRACKER_RESOLUTION/TRACKER_DIST;
    double ydist=(pos.y-old_head_pos.y)*TRACKER_RESOLUTION/TRACKER_DIST;
    double zdist=(pos.z-old_head_pos.z)*TRACKER_RESOLUTION/TRACKER_DIST;
    cerr << "HEAD: xdist=" << xdist << ", ydist=" << ydist << ", zdist=" << zdist << endl;
    View tview(view.get());
    Point eyep(orig_eye-frame_right*xdist-frame_up*ydist-frame_front*zdist);
    Point lookat(eyep+frame_front);
    tview.eyep(eyep);
    tview.lookat(lookat);
    tview.up(frame_up);
    view.set(tview);
    need_redraw=1;
}

void Roe::dump_objects(const clString& filename, const clString& format)
{
    if(format == "scirun_binary" || format == "scirun_ascii"){
	Piostream* stream;
	if(format == "scirun_binary")
	    stream=new BinaryPiostream(filename, Piostream::Write);
	else
	    stream=new TextPiostream(filename, Piostream::Write);
	if(stream->error()){
	    delete stream;
	    return;
	}
	manager->geomlock.read_lock();
	GeomScene scene(bgcolor.get(), view.get(), &manager->lighting,
			&manager->ports);
	Pio(*stream, scene);
	if(stream->error()){
	    cerr << "Error writing geom file: " << filename << endl;
	} else {
	    cerr << "Done writing geom file: " << filename << endl;
	}
	delete stream;
	manager->geomlock.read_unlock();
    } else {
	cerr << "WARNING: format " << format << " not supported!\n";
    }
}

#ifdef __GNUG__

#include <Classlib/HashTable.cc>

template class HashTable<clString, Renderer*>;
template class HashKey<clString, Renderer*>;

template class HashTable<clString, ObjTag*>;
template class HashKey<clString, ObjTag*>;

template class HashTable<clString, int>;
template class HashKey<clString, int>;

#endif
