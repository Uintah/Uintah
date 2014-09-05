//static char *id="@(#) $Id$";

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

#include <PSECommon/Modules/Salmon/Salmon.h>
#include <PSECommon/Modules/Salmon/Roe.h>
#include <PSECommon/Modules/Salmon/Renderer.h>
#include <PSECommon/Modules/Salmon/Ball.h>
#include <PSECommon/Modules/Salmon/BallMath.h>
#include <SCICore/Util/Debug.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Util/Timer.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/PointLight.h>
#include <SCICore/Geom/GeomScene.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <SCICore/Thread/FutureValue.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <string.h>
#include <sstream>
using std::ostringstream;

#define MouseStart 0
#define MouseEnd 1
#define MouseMove 2

namespace PSECommon {
namespace Modules {

using SCICore::Math::Abs;
using SCICore::Geometry::Cross;
using SCICore::Geometry::Dot;
using SCICore::Containers::to_string;
using SCICore::GeomSpace::BState;
using SCICore::GeomSpace::GeomScene;
using SCICore::PersistentSpace::BinaryPiostream;
using SCICore::PersistentSpace::TextPiostream;

//static DebugSwitch autoview_sw("Roe", "autoview");
static Roe::MapClStringObjTag::iterator viter;

Roe::Roe(Salmon* s, const clString& id)
  : manager(s),
  view("view", id, this),
  homeview(Point(.55, .5, 0), Point(.0, .0, .0), Vector(0,1,0), 25),
  bgcolor("bgcolor", id, this), shading("shading", id, this),
  do_stereo("do_stereo", id, this),
  sbase("sbase", id, this),
    sr("sr", id, this),
  // >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
  do_bawgl("do_bawgl", id, this),
  // <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<
  drawimg("drawimg", id, this),
  saveprefix("saveprefix", id, this),
  id(id),doingMovie(false),makeMPEG(false),
  curFrame(0),curName("movie")
  {
    sr.set(1);
    inertia_mode=0;
    bgcolor.set(Color(0,0,0));
    view.set(homeview);
    TCL::add_command(id+"-c", this, 0);
    current_renderer=0;
    maxtag=0;
    mouse_obj=0;
    ball = new BallData();
    ball->Init();
    // >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
    bawgl = new SCIBaWGL();
    // <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<
  }

clString Roe::set_id(const clString& new_id)
{
  clString old_id=id;
  id=new_id;
  return old_id;
}

void Roe::itemAdded(GeomSalmonItem* si)
{
    ObjTag* vis;
    
    viter = visible.find(si->name);
    if(viter==visible.end()){
      // Make one...
      vis=scinew ObjTag;
      vis->visible=scinew TCLvarint(si->name, id, this);
      vis->visible->set(1);
      vis->tagid=maxtag++;
      visible[si->name] = vis;
      ostringstream str;
      str << id << " addObject " << vis->tagid << " \"" << si->name << "\"";
      TCL::execute(str.str().c_str());
    } else {
      vis = (*viter).second;
      ostringstream str;
      str << id << " addObject2 " << vis->tagid;
      TCL::execute(str.str().c_str());
    }
    // invalidate the bounding box
    bb.reset();
    need_redraw=1;
}

void Roe::itemDeleted(GeomSalmonItem *si)
{
  ObjTag* vis;
    
  viter = visible.find(si->name);
  if (viter == visible.end()) { // if not found
    cerr << "Where did that object go???" << endl;
  }
  else {
    vis = (*viter).second;
    ostringstream str;
    str << id << " removeObject " << vis->tagid;
    TCL::execute(str.str().c_str());
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
    TCL::delete_command( id+"-c" );
}

void Roe::get_bounds(BBox& bbox)
{
  bbox.reset();

  GeomIndexedGroup::IterIntGeomObj iter = manager->ports.getIter();
    
  for ( ; iter.first != iter.second; iter.first++) {
	
    GeomIndexedGroup::IterIntGeomObj serIter =
      ((GeomSalmonPort*)((*iter.first).second))->getIter();

				// items in the scen are all
				// GeomSalmonItem's...
    for ( ; serIter.first != serIter.second; serIter.first++) {
      GeomSalmonItem *si=(GeomSalmonItem*)((*serIter.first).second);
	    
				// Look up the name to see if it
				// should be drawn...
      ObjTag* vis;
	    
      viter = visible.find(si->name);
	    
      if (viter != visible.end()) { // if found
	vis = (*viter).second;
	if (vis->visible->get()) {
	  if(si->lock) si->lock->readLock();
	  si->get_bounds(bbox);
	  if(si->lock) si->lock->readUnlock();
	}
      }
      else {
	cerr << "Warning: object " << si->name
	     << " not in visibility database...\n";
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
	    ostringstream str;
	    str << "translate: " << total_x << ", " << total_y;
	    update_mode_string(str.str().c_str());
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
	    ostringstream str;
	    str << "scale: " << total_x*100 << "%";
	    update_mode_string(str.str().c_str());
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

void Roe::mouse_rotate_eyep(int action, int x, int y, int, int, int time)
{
    switch(action){
    case MouseStart:
	{
	    if(inertia_mode){
		inertia_mode=0;
		redraw();
	    }
	    update_mode_string("rotate lookatP:");
	    last_x=x;
	    last_y=y;

	    // The center of rotation is eye point...
	    View tmpview(view.get());
	    int xres=current_renderer->xres;
	    int yres=current_renderer->yres;
	    
	    rot_point_valid=0;

	    rot_point = tmpview.eyep();
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
	    HVect mouse((2.0*x)/xres-1.0, 2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
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
	    
	    if(!rot_point_valid)
		break;

	    HVect mouse((2.0*x)/(xres) - 1.0, 2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
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
	    tmpview.lookat(tmpview.eyep()-(z_a*(eye_dist)).vector());
	    view.set(tmpview);
	    need_redraw=1;
	    update_mode_string("rotate lookatP:");

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
	    tmpview.lookat((tmpview.eyep()-(z_a*(eye_dist)).vector()));
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
	    inertia_mode=2;
	    need_redraw=1;

	    if (mag < 0.00001) { // arbitrary ad-hoc threshold
		inertia_mode = 0;
		need_redraw = 1;
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

//>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
static int prevPrinc;
void Roe::bawgl_pick(int action, int iv[3], GLfloat fv[3])
{
    BState bs;
    switch(action) {
    case BAWGL_PICK_START:
	{
	    
	    current_renderer->get_pick(manager, this, iv[0], iv[1],
				       pick_obj, pick_pick, pick_n); 
	    if (pick_obj){
		update_mode_string(pick_obj);
		pick_pick->set_picked_obj(pick_obj);
		pick_pick->pick(this,bs);
		total_x=0;
		total_y=0;
		total_z=0;
		//need_redraw=1;
	    } else {
		update_mode_string("pick: none");
	    }

	}
    break;
    case BAWGL_PICK_MOVE:
	{
	    if (!pick_obj || !pick_pick) break;
	    Vector dir(fv[0],fv[1],fv[2]);
	    //float dv= sqrt(fv[0]*fv[0]+fv[1]*fv[1]+fv[2]*fv[2]);
	    //pick_pick->moved(0, dv, dir, bs);

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
		prevPrinc= prin_dir;
		double dist=dir.length();
		Vector mtn(pick_pick->principal(prin_dir)*dist);
		total_x+=mtn.x();
		total_y+=mtn.y();
		total_z+=mtn.z();
		if (Abs(total_x) < .0001) total_x=0;
		if (Abs(total_y) < .0001) total_y=0;
		if (Abs(total_z) < .0001) total_z=0;
		update_mode_string(pick_obj);
		pick_pick->moved(prin_dir, dist, mtn, bs);
	    } else {
		update_mode_string("pick: Bad direction...");
	    }
	}
    break;
    case BAWGL_PICK_END:
	{
	    if(pick_pick){
		pick_pick->release( bs );
	    }
	    pick_pick=0;
	    pick_obj=0;
	    update_mode_string("");
	}
    break;
    }
}
//<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<

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
				       pick_obj, pick_pick, pick_n);

	    if (pick_obj){
		update_mode_string(pick_obj);
		pick_pick->set_picked_obj(pick_obj);
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
		update_mode_string(pick_obj);
		pick_pick->moved(prin_dir, dist, mtn, bs);
		need_redraw=1;
	    } else {
		update_mode_string("pick: Bad direction...");
	    }
	    last_x = x;
	    last_y = y;
	}
	break;
    case MouseEnd:
	if(pick_pick){
	    pick_pick->release( bs );
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
  if (args.count() < 2) {
    args.error("Roe needs a minor command");
    return;
  }
  
  if (args[1] == "sgi_defined") {
    clString result("");
#ifdef __sgi
#if (_MIPS_SZPTR == 64)
    result += "2";
#else
    result += "1";
#endif
#else
    result += "0";
#endif
    args.result( result );
  } else if (args[1] == "dump_roe") {
    if (args.count() != 4) {
      args.error("Roe::dump_roe needs an output file name and type");
      return;
    }
				// We need to dispatch this one to the
				// remote thread.  We use an ID string
				// instead of a pointer in case this
				// roe gets killed by the time the
				// redraw message gets dispatched.
    manager->mailbox.send(scinew
	     SalmonMessage(MessageTypes::RoeDumpImage, id, args[2], args[3]));
  }else if (args[1] == "startup") {
				// Fill in the visibility database...
    GeomIndexedGroup::IterIntGeomObj iter = manager->ports.getIter();
    
    for ( ; iter.first != iter.second; iter.first++) {
      
      GeomIndexedGroup::IterIntGeomObj serIter =
	((GeomSalmonPort*)((*iter.first).second))->getIter();
      
      for ( ; serIter.first != serIter.second; serIter.first++) {
	GeomSalmonItem *si =
	  (GeomSalmonItem*)((*serIter.first).second);
	itemAdded(si);
      }
    }
  }
  else if (args[1] == "setrenderer") {
    if (args.count() != 6) {
      args.error("setrenderer needs a renderer name, etc");
      return;
    }
    Renderer* r=get_renderer(args[2]);
    if (!r) {
      args.error("Unknown renderer!");
      return;
    }
    if (current_renderer) current_renderer->hide();
    current_renderer=r;
    args.result(r->create_window(this, args[3], args[4], args[5]));
  } else if (args[1] == "redraw") {
				// We need to dispatch this one to the
				// remote thread We use an ID string
				// instead of a pointer in case this
				// roe gets killed by the time the
				// redraw message gets dispatched.
    if(!manager->mailbox.trySend(scinew SalmonMessage(id)))
      cerr << "Redraw event dropped, mailbox full!\n";
  } else if(args[1] == "destroy"){
    manager->delete_roe(this);
  } else if(args[1] == "anim_redraw"){
				// We need to dispatch this one to the
				// remote thread We use an ID string
				// instead of a pointer in case this
				// roe gets killed by the time the
				// redraw message gets dispatched.
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
    if(!manager->mailbox.trySend(scinew SalmonMessage(id, tbeg, tend,
      nframes, framerate)))
      cerr << "Redraw event dropped, mailbox full!\n";
  } else if(args[1] == "mtranslate"){
    do_mouse(&Roe::mouse_translate, args);
  } else if(args[1] == "mrotate"){
    do_mouse(&Roe::mouse_rotate, args);
  } else if(args[1] == "mrotate_eyep"){
    do_mouse(&Roe::mouse_rotate_eyep, args);
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
  } else if(args[1] == "dolly2"){
    if(args.count() != 3){
      args.error("dolly2 needs an amount");
      return;
    }
    double amount;
    if(!args[2].get_double(amount)){
      args.error("Can't figure out amount");
      return;
    }
    View cv(view.get());
    Vector lookdir(cv.eyep()-cv.lookat());
    amount = amount-1;
    lookdir*=amount;
    cv.eyep(cv.eyep()+lookdir);
    cv.lookat(cv.lookat()+lookdir);
    animate_to_view(cv, 1.0);
  } else if(args[1] == "saveobj") {
    if(args.count() != 4){
      args.error("Roe::dump_roe needs an output file name and format!");
      return;
    }
				// We need to dispatch this one to the
				// remote thread We use an ID string
				// instead of a pointer in case this
				// roe gets killed by the time the
				// redraw message gets dispatched.
    manager->mailbox.send(scinew SalmonMessage(MessageTypes::RoeDumpObjects,
      id, args[2], args[3]));
  } else if(args[1] == "listvisuals"){
    current_renderer->listvisuals(args);
  } else if(args[1] == "switchvisual"){
    if(args.count() != 6){
      args.error("switchvisual needs a window name, a visual index, a width and a height");
      return;
    }
    int idx;
    if(!args[3].get_int(idx)){
      args.error("bad index for switchvisual");
      return;
    }
    int width;
    if(!args[4].get_int(width)){
      args.error("Bad width");
      return;
    }
    int height;
    if(!args[5].get_int(height)){
      args.error("Bad height");
      return;
    }
    current_renderer->setvisual(args[2], idx, width, height);
    // >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
  } else if(args[1] == "startbawgl") {
    if( bawgl->start(this, "bench.config")  == 0 )
      {
	bawgl_error = 0;
      }
    else
      {
	do_bawgl.set(0);
	bawgl_error = 1;
	args.error("Bummer!\n Check if the device daemons are alive!");
      }
  } else if(args[1] == "stopbawgl"){
    if( !bawgl_error ) bawgl->stop();
    // <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<
  } else {
    args.error("Unknown minor command '" + args[1] + "' for Roe");
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
    if(!manager->mailbox.trySend(scinew RoeMouseMessage(id, handler, action, x, y, state, btn, time)))
	cerr << "Mouse event dropped, mailbox full!\n";
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

	// I'm changing this to be 20 degrees - Dave

	double myfov=20.0;

        Vector diag(bbox.diagonal());
	double w=diag.length();
	Vector lookdir(cv.lookat()-cv.eyep()); 
	lookdir.normalize();
	const double scale = 1.0/(2*Tan(DtoR(myfov/2.0)));
	double length = w*scale;
	cv.fov(myfov);
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

void Roe::update_mode_string(GeomObj* pick_obj)
{
    clString ms="pick: ";
    GeomSalmonItem* si=dynamic_cast<GeomSalmonItem*>(pick_obj);
    if(!si){
	ms+="not a GeomSalmonItem?";
    } else {
	ms+=si->name;
    }
    if(pick_n != 0x12345678)
	ms+=", index="+to_string(pick_n);
    update_mode_string(ms);
}

void Roe::update_mode_string(const clString& msg)
{
    ostringstream str;
    str << id << " updateMode \"" << msg << "\"";
    TCL::execute(str.str().c_str());
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
  MapClStringRenderer::iterator riter;

  riter = renderers.find(name);
  if (riter == renderers.end()) { // if not found
    // Create it...
    r = Renderer::create(name);
    if (r) {
      renderers[name] = r;
    }
  }
  else {
    r = (*riter).second;
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
  int i;
  for (i = 0; i < roe_objs.size(); i++){
    (r->*pmf)(manager, this, roe_objs[i]);
  }

  Array1<GeomSalmonItem*> transp_objs; // transparent objects - drawn last

  GeomIndexedGroup::IterIntGeomObj iter = manager->ports.getIter();
  
  for ( ; iter.first != iter.second; iter.first++) {
      
    GeomIndexedGroup::IterIntGeomObj serIter = 
      ((GeomSalmonPort*)((*iter.first).second))->getIter();
    
    for ( ; serIter.first != serIter.second; serIter.first++) {
	    
      GeomSalmonItem *si =
	(GeomSalmonItem*)((*serIter.first).second);
      
      // Look up the name to see if it should be drawn...
      ObjTag* vis;
      
      viter = visible.find(si->name);
      if (viter != visible.end()) { // if found
	vis = (*viter).second;
	if (vis->visible->get()) {
	  if (strstr(si->name(),"TransParent")) { // delay drawing
	    transp_objs.add(si);
	  }
	  else {
	    if(si->lock)
	      si->lock->readLock();
	    (r->*pmf)(manager, this, si);
	    if(si->lock)
	      si->lock->readUnlock();
	  }
	}
      }
      else {
	cerr << "Warning: object " << si->name << " not in visibility database...\n";
      }
    }
  }

  // now run through the transparent objects...

  for(i=0;i<transp_objs.size();i++) {
    GeomSalmonItem *si = transp_objs[i];    

    if(si->lock)
      si->lock->readLock();
    (r->*pmf)(manager, this, si);
    if(si->lock)
      si->lock->readUnlock();
  }

  // now you are done...

}

void Roe::set_current_time(double time)
{
    set_tclvar(id, "current_time", to_string(time));
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
	manager->geomlock.readLock();
	GeomScene scene(bgcolor.get(), view.get(), &manager->lighting,
			&manager->ports);
	SCICore::PersistentSpace::Pio(*stream, scene);
	if(stream->error()){
	    cerr << "Error writing geom file: " << filename << endl;
	} else {
	    cerr << "Done writing geom file: " << filename << endl;
	}
	delete stream;
	manager->geomlock.readUnlock();
   } else {
	cerr << "WARNING: format " << format << " not supported!\n";
    }
}

void Roe::getData(int datamask, FutureValue<GeometryData*>* result)
{
    if(current_renderer){
	cerr << "calling current_renderer->getData\n";
	current_renderer->getData(datamask, result);
	cerr << "current_renderer...\n";
    } else {
	result->send(0);
    }
}

void Roe::setView(View newView) {
    view.set(newView);
    manager->mailbox.send(scinew SalmonMessage(id)); // Redraw
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.14.2.2  2000/10/26 10:03:40  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.14.2.1  2000/09/28 03:16:06  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.19  2000/10/08 05:42:38  samsonov
// Added rotation around eye point and corresponding inertia mode; to use the mode , use ALT key and middle mouse button
//
// Revision 1.18  2000/09/29 08:06:59  samsonov
// Changes in stereo implementation
//
// Revision 1.17  2000/08/11 15:51:22  bigler
// Removed set_index(int) calls to GeomPick class and replaced them with
// set_picked_obj(GeomObj*).
//
// Revision 1.16  2000/06/09 17:50:18  kuzimmer
// Hopefully everything is fixed so that you can use -lifl on SGI's and you can use -lcl on SGI's in32bit mode.
//
// Revision 1.15  2000/06/07 20:59:26  kuzimmer
// Modifications to make the image save menu item work on SGIs
//
// Revision 1.14  2000/03/17 09:27:17  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.13  2000/03/11 00:39:52  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.12  1999/12/03 00:28:59  dmw
// added setView message for Salmon/Roe
//
// Revision 1.11  1999/11/19 17:50:11  ikits
// Put in __sgi to make it compile for linux. Replaced int errcode w/ GLenum errcode in OpenGL.cc.
//
// Revision 1.10  1999/11/19 05:44:15  dmw
// commented out performance reporting so we dont get so many printouts from Salmon
//
// Revision 1.9  1999/11/16 00:47:26  yarden
// put "#ifdef __sgi" around code for BAWGL
//
// Revision 1.8  1999/10/21 22:39:06  ikits
// Put bench.config into PSE/src (where the executable gets invoked from). Fixed bug in the bawgl code and added preliminary navigation and picking.
//
// Revision 1.7  1999/10/16 20:50:59  jmk
// forgive me if I break something -- this fixes picking and sets up sci
// bench - go to /home/sci/u2/VR/PSE for the latest sci bench technology
// gota getup to get down.
//
// Revision 1.6  1999/10/07 02:06:56  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 22:04:33  sparker
// Fixed picking
// Added messages for pick mode
//
// Revision 1.4  1999/08/29 00:46:42  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.3  1999/08/18 20:19:53  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:38  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:52  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:28  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/13 20:35:16  dav
// update
//
// Revision 1.2  1999/05/13 18:24:09  dav
// Removed TCLView from Roe.cc and uncommented a number of pick things
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
