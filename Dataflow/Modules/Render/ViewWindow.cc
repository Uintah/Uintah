/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#define protected public
/*
 *  ViewWindow.cc:  The Geometry Viewer Window
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */
#include <sci_defs.h>

#include <Dataflow/Modules/Render/Viewer.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Modules/Render/Ball.h>
#include <Dataflow/Modules/Render/BallMath.h>
#include <Core/Util/Debug.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/Timer.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/PointLight.h>
#include <Core/Geom/GeomScene.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomCone.h>      
#include <Core/Geom/GeomCylinder.h>  
#include <Core/Geom/GeomGroup.h>     
#include <Core/Geom/GeomSticky.h>     
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/FutureValue.h>
#include "OpenGL.h"
#include <iostream>
using std::cerr;
#include <stdio.h>
#include <string.h>
#include <sstream>
using std::ostringstream;

// CollabVis code begin
#ifdef HAVE_COLLAB_VIS
#include <Dataflow/Network/Network.h>
#endif
// CollabVis code end

#define MouseStart 0
#define MouseEnd 1
#define MouseMove 2

namespace SCIRun {
  
//static DebugSwitch autoview_sw("ViewWindow", "autoview");
static ViewWindow::MapStringObjTag::iterator viter;

#if 0
static void
add_pt( ViewWindow *viewwindow, Point p, double s=.2 )
{
  GeomSphere *obj = scinew GeomSphere(p, s);
  viewwindow->viewwindow_objs.push_back(obj);
  viewwindow->viewwindow_objs_draw.push_back(true);
  viewwindow->need_redraw = 1;
}
#endif

ViewWindow::ViewWindow(Viewer* s, GuiInterface* gui, GuiContext* ctx)
  : gui(gui), ctx(ctx), manager(s),
    pos(ctx->subVar("pos")),
    caxes(ctx->subVar("caxes")),
    raxes(ctx->subVar("raxes")),
    iaxes(ctx->subVar("iaxes")),
    // CollabVis code begin
    HaveCollabVis_(ctx->subVar("have_collab_vis")),
    // CollabVis code end
    doingMovie(false),
    makeMPEG(false),
    curFrame(0),
    curName("movie"),
    dolly_throttle(0),
    // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
    groupInfo( NULL ),
    groupInfoLock( "GroupInfoLock" ),
    handlingOneTimeRequest( false ),
    viewStateLock("ViewStateLock"),
#endif
    // CollabVis code end
    show_rotation_axis(false),
    id(ctx->getfullname()),
    need_redraw( false ),
    view(ctx->subVar("view")),
    homeview(Point(2.1, 1.6, 11.5), Point(.0, .0, .0), Vector(0,1,0), 20),
    lightColors(ctx->subVar("lightColors")),
    lightVectors(ctx->subVar("lightVectors")),
    bgcolor(ctx->subVar("bgcolor")), 
    shading(ctx->subVar("shading")),
    do_stereo(ctx->subVar("do_stereo")), 
    ambient_scale(ctx->subVar("ambient-scale")),
    diffuse_scale(ctx->subVar("diffuse-scale")),
    specular_scale(ctx->subVar("specular-scale")),
    emission_scale(ctx->subVar("emission-scale")),
    shininess_scale(ctx->subVar("shininess-scale")),
    polygon_offset_factor(ctx->subVar("polygon-offset-factor")),
    polygon_offset_units(ctx->subVar("polygon-offset-units")),
    point_size(ctx->subVar("point-size")),
    line_width(ctx->subVar("line-width")),
    sbase(ctx->subVar("sbase")),
    sr(ctx->subVar("sr")),
    // --  BAWGL -- 
    do_bawgl(ctx->subVar("do_bawgl")),  
    // --  BAWGL -- 
    drawimg(ctx->subVar("drawimg")),
    saveprefix(ctx->subVar("saveprefix")),
    file_resx(ctx->subVar("resx")),
    file_resy(ctx->subVar("resy")),
    file_aspect(ctx->subVar("aspect")),
    file_aspect_ratio(ctx->subVar("aspect_ratio")),
    gui_global_light_(ctx->subVar("global-light")),
    gui_global_fog_(ctx->subVar("global-fog")),
    gui_global_debug_(ctx->subVar("global-debug")),
    gui_global_clip_(ctx->subVar("global-clip")),
    gui_global_cull_(ctx->subVar("global-cull")),
    gui_global_dl_(ctx->subVar("global-dl")),
    gui_global_type_(ctx->subVar("global-type")),
    gui_ortho_view_(ctx->subVar("ortho-view"))
{
  inertia_mode=0;
  bgcolor.set(Color(0,0,0));

  view.set(homeview);

  gui->add_command(id+"-c", this, 0);
  current_renderer=new OpenGL(gui);
  maxtag=0;
  ball = new BallData();
  ball->Init();
  // --  BAWGL -- 
  bawgl = new SCIBaWGL();
  // --  BAWGL -- 
  
  // 0 - Axes, visible
  viewwindow_objs.push_back( createGenAxes() );
  viewwindow_objs_draw.push_back(true);              

  // 1 - Unicam control sphere, not visible by default.
  focus_sphere = scinew GeomSphere;
  Color c(0.0, 0.0, 1.0);
  MaterialHandle focus_color = scinew Material(c);
  viewwindow_objs.push_back(scinew GeomMaterial(focus_sphere, focus_color));
  viewwindow_objs_draw.push_back(false);

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  //cerr << "[HAVE_COLLAB_VIS] (ViewWindow::ViewWindow) 1\n";

  // Allocate memory for server object
  server = scinew ViewServer();
  
  // Add the view server to the network. All further module adds/subs/mods
  // will now be noted.
  s->getNetwork()->write_lock();
  //s->getNetwork()->server = &server;
  s->getNetwork()->server = server;
  //server.network = s->getNetwork();
  server->network = s->getNetwork();
  s->getNetwork()->write_unlock();

  //cout << "[HAVE_COLLAB_VIS] (ViewWindow::ViewWindow) Setting HaveCollabVis_ to 1\n";
  // Set have_collab_vis variable to true in the GUI
  HaveCollabVis_.set(1);
  
#else

  // Set have_collab_vis variable to false in the GUI
  HaveCollabVis_.set(0);
  
#endif
  // CollabVis code end

}


string ViewWindow::set_id(const string& new_id)
{
  string old_id=id;
  id=new_id;
  return old_id;
}

void ViewWindow::itemAdded(GeomViewerItem* si)
{
  ObjTag* vis;
    
  viter = visible.find(si->name_);
  if(viter==visible.end()){
    // Make one...
    vis=scinew ObjTag;
    vis->visible=scinew GuiInt(ctx->subVar(si->name_), 1);
    vis->tagid=maxtag++;
    visible[si->name_] = vis;
    ostringstream str;
    str << id << " addObject " << vis->tagid << " \"" << si->name_ << "\"";
    gui->execute(str.str());
  } else {
    vis = (*viter).second;
    ostringstream str;
    str << id << " addObject2 " << vis->tagid;
    gui->execute(str.str());
  }
  // invalidate the bounding box
  bb.reset();
  need_redraw=true;
}

void ViewWindow::itemDeleted(GeomViewerItem *si)
{
  ObjTag* vis;
    
  viter = visible.find(si->name_);
  if (viter == visible.end()) { // if not found
    cerr << "Where did that object go???" << "\n";
  }
  else {
    vis = (*viter).second;
    ostringstream str;
    str << id << " removeObject " << vis->tagid;
    gui->execute(str.str());
  }
				// invalidate the bounding box
  bb.reset();
  need_redraw=true;
}

void ViewWindow::itemRenamed(GeomViewerItem *si, string newname)
{
  // Remove old
  {
    ObjTag *vis;
    viter = visible.find(si->name_);
    if (viter == visible.end()) { // if not found
      cerr << "Where did that object go???" << "\n";
    }
    else {
      vis = (*viter).second;
      ostringstream str;
      str << id << " removeObject " << vis->tagid;
      gui->execute(str.str());
    }
  }

  // Rename.
  si->name_ = newname;

  // Reinsert.
  {
    ObjTag* vis;
    
    viter = visible.find(si->name_);
    if(viter==visible.end()){
      // Make one...
      vis=scinew ObjTag;
      vis->visible=scinew GuiInt(ctx->subVar(si->name_), 1);
      vis->tagid=maxtag++;
      visible[si->name_] = vis;
      ostringstream str;
      str << id << " addObject " << vis->tagid << " \"" << si->name_ << "\"";
      gui->execute(str.str());
    } else {
      vis = (*viter).second;
      ostringstream str;
      str << id << " addObject2 " << vis->tagid;
      gui->execute(str.str());
    }
  }
}


// need to fill this in!   
#ifdef OLDUI
void ViewWindow::itemCB(CallbackData*, void *gI) {
  GeomItem *g = (GeomItem *)gI;
  g->vis = !g->vis;
  need_redraw=true;
}

void ViewWindow::spawnChCB(CallbackData*, void*)
{
  double mat[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, mat);

  kids.add(scinew ViewWindow(manager, mat, mtnScl));
  kids[kids.size()-1]->SetParent(this);
  for (int i=0; i<geomItemA.size(); i++)
    kids[kids.size()-1]->itemAdded(geomItemA[i]->geom, geomItemA[i]->name);
}
#endif

ViewWindow::~ViewWindow()
{
  delete current_renderer;
  delete ball;
  delete bawgl;

  gui->delete_command( id+"-c" );
}

void ViewWindow::get_bounds(BBox& bbox)
{
  bbox.reset();

  GeomIndexedGroup::IterIntGeomObj iter = manager->ports_.getIter();
    
  for ( ; iter.first != iter.second; iter.first++) {
	
    GeomIndexedGroup::IterIntGeomObj serIter =
      ((GeomViewerPort*)((*iter.first).second.get_rep()))->getIter();

				// items in the scen are all
				// GeomViewerItem's...
    for ( ; serIter.first != serIter.second; serIter.first++) {
      GeomViewerItem *si=(GeomViewerItem*)((*serIter.first).second.get_rep());
	    
				// Look up the name to see if it
				// should be drawn...
      ObjTag* vis;
	    
      viter = visible.find(si->name_);
	    
      if (viter != visible.end()) { // if found
	vis = (*viter).second;
	if (vis->visible->get()) {
	  if(si->crowd_lock_) si->crowd_lock_->readLock();
	  si->get_bounds(bbox);
	  if(si->crowd_lock_) si->crowd_lock_->readUnlock();
	}
      }
      else {
	cerr << "Warning: object " << si->name_
	     << " not in visibility database...\n";
	si->get_bounds(bbox);
      }
    }
  }

  // XXX - START - ASF ADDED FOR UNICAM
  //   cerr << "viewwindow_objs.size() = " << viewwindow_objs.size() << "\n";
  //int objs_size = viewwindow_objs.size();
  unsigned int draw_size = viewwindow_objs_draw.size();
  for(unsigned int i=0;i<viewwindow_objs.size();i++) {
    
    if (i<draw_size && viewwindow_objs_draw[i])
      viewwindow_objs[i]->get_bounds(bbox);
  }
  // XXX - END   - ASF ADDED FOR UNICAM

  // If the bounding box is empty, make it default to sane view.
  if (!bbox.valid())
  {
    bbox.extend(Point(-1.0, -1.0, -1.0));
    bbox.extend(Point(1.0, 1.0, 1.0));
  }
}

// CollabVis code begin
#ifdef HAVE_COLLAB_VIS

#define _sub(a,x,y) a[ (x) + (y) * xres * 3 ]

void ViewWindow::sendImageToServer( char * image, int xres, int yres ) {
  //cout << "[HAVE_COLLAB_VIS] (ViewWindow::sendImageToServer) 0\n";
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) entered\n" );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) locking groupInfoLock\n" );

  //groupInfoLock.lock();
  if( !groupInfoLock.tryLock() )
  {
    Log::log( SemotusVisum::Logging::WARNING, "(ViewWindow::sendImageToServer) failed to lock groupInfoLock\n" );
    return;
  }

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) before ImageRenderer *ir\n" );
  
  ImageRenderer *ir = (ImageRenderer *)(groupInfo->group->getRenderer());

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) before getSubimage\n" );
  
  if ( ir->getSubimage() ) {
    WallClockTimer t;
    t.start();
    // Try this - see how much of the image is non-background.
    int left = 0, top = 0;
    int right = xres-1, bottom = yres-1;
    char THRESHOLD = 4; // if it's within this, it's background!
    
    Color bg(bgcolor.get());
    char r = (char)(bg.r() * 255) % 255;
    char g = (char)(bg.g() * 255) % 255;
    char b = (char)(bg.b() * 255) % 255;
    
    
    // Top
    for ( int i = top; i < bottom; i++ )
      for ( int j = left; j < right*3; j+=3 )
	if ( Abs( _sub( image, j, i ) - r ) > THRESHOLD ||
	     Abs( _sub( image, j+1, i ) - g ) > THRESHOLD ||
	     Abs( _sub( image, j+2, i ) - b ) > THRESHOLD ) {
	  top = i;
	  goto left;
	}
  left:
    for ( int i = left; i < right*3; i+=3 )
      for ( int j = top; j < bottom; j++ )
	if ( Abs( _sub( image, i, j ) - r ) > THRESHOLD ||
	     Abs( _sub( image, i+1, j ) - g ) > THRESHOLD ||
	     Abs( _sub( image, i+2, j ) - b ) > THRESHOLD ) {
	  left = i/3;
	  goto bottom;
	}
  bottom:
    for ( int i = bottom; i > top; i-- )
      for ( int j = left; j < right*3; j+=3 )
	if ( Abs( _sub( image, j, i ) - r ) > THRESHOLD ||
	     Abs( _sub( image, j+1, i ) - g ) > THRESHOLD ||
	     Abs( _sub( image, j+2, i ) - b ) > THRESHOLD ) {
	  bottom = i;
	  goto right;
	}
  right:
    for ( int i = right*3; i > left; i-=3 )
      for ( int j = top; j < bottom; j++ )    
	if ( Abs( _sub( image, i, j ) - r ) > THRESHOLD ||
	     Abs( _sub( image, i+1, j ) - g ) > THRESHOLD ||
	     Abs( _sub( image, i+2, j ) - b ) > THRESHOLD ) {
	  right = i/3;
	  goto done;
	}
  done:
    t.stop();
    cerr << "Subimage Took " << t.time() * 1000.0 << " ms." << endl;
    
    int xdim = right-left;
    int ydim = bottom-top;
    
    char *newImage = scinew char[ xdim * ydim * 3 ];
    for ( int i = 0, y = top; i < ydim; i++, y++ )
      memcpy( &newImage[ i*xdim*3 ],
	      &image[ left*3 + y*xres*3 ],
	      xdim*3 );
    //server.sendImage( newImage, xdim, ydim, left, top, xres, yres, bg,
    //	      groupInfo->group->getRenderer() );
    Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) before server->sendImage 1\n" );
    server->sendImage( newImage, xdim, ydim, left, top, xres, yres, bg,
    	      groupInfo->group->getRenderer() );
    delete[] image;
  }
  else
  {
    //server.sendImage( image, xres, yres, groupInfo->group->getRenderer() );
    Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) before server->sendImage 2\n" );
    server->sendImage( image, xres, yres, groupInfo->group->getRenderer() );
  }

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) unlocking groupInfoLock\n" );
  groupInfoLock.unlock();

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::sendImageToServer) leaving\n" );
  cerr << "End of ViewWindow::sendImageToServer\n";
}

#undef _sub
	   
void  ViewWindow::getViewState( ViewWindowState &state ) {
  //cout << "[HAVE_COLLAB_VIS] (ViewWindow::getViewState) 0" << endl;
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) entered\n" );
  char buffer[1000];
  snprintf( buffer, 1000,
	    "(ViewWindow::getViewState) Getting view state for state 0x%x, window 0x%x",
	    (void *)(&state), (void *)this );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) calling viewStateLock.readLock()" );
  viewStateLock.readLock();
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) finished calling viewStateLock.readLock()" );
  state.angular_v = angular_v;
  state.ball = *ball;
  state.dolly_total = dolly_total;
  state.dolly_vector = dolly_vector;
  state.dolly_throttle = dolly_throttle;
  state.eye_dist = eye_dist;
  state.inertia_mode = inertia_mode;
  state.last_x = last_x;
  state.last_y = last_y;
  state.last_time = last_time;
  state.prev_quat[0] = prev_quat[0];
  state.prev_quat[1] = prev_quat[1];
  state.prev_quat[2] = prev_quat[2];
  state.prev_time[0] = prev_time[0];
  state.prev_time[1] = prev_time[1];
  state.prev_time[2] = prev_time[2];
  state.prev_trans = prev_trans;
  state.rot_point = rot_point;
  state.rot_point_valid = rot_point_valid;
  state.rot_view = rot_view;
  state.total_scale = total_scale;
  state.total_x = total_x; 
  state.total_y = total_y; 
  state.total_z = total_z;
  // New 5-25-02

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) checking state.view" );
  
  if ( state.view == NULL )
  {
    //state.view = scinew GuiView(view); // this function isn't actually implemented
    state.view = scinew GuiView(ctx);
    *state.view = view;
  }
  else
  {
    *state.view = view;
  }


  /*
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) before logging statements" );
  
  // Old state.view = view;

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) before state.view->get()" );
  View v = state.view->get();
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) before eye" );
  snprintf( buffer, 1000, "Eye for get state 0x%x: (%f,%f,%f)",
	    (void *)(&state),
	    v.eyep().x(),
	    v.eyep().y(),
	    v.eyep().z() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) before lookat" );
  snprintf( buffer, 1000, "Lookat for get state 0x%x: (%f,%f,%f)",
	    (void *)(&state),
	    v.lookat().x(),
	    v.lookat().y(),
	    v.lookat().z() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) before up" );
  snprintf( buffer, 1000, "Up for get state 0x%x: (%f,%f,%f)",
	    (void *)(&state),
	    v.up().x(),
	    v.up().y(),
	    v.up().z() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) before fov" );
  snprintf( buffer, 1000, "Fov for get state 0x%x: %f",
	    (void *)(&state),
	    v.fov() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  */
  
	    
  // GUI parameters 
  //get_gui_stringvar( id, "global-light", state.lighting ); 
  //get_gui_stringvar( id, "global-fog", state.fog ); 
  //get_gui_stringvar( id, "global-type", state.shading );
 
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) calling viewStateLock.readUnlock()" );
  
  viewStateLock.readUnlock();
  snprintf( buffer, 1000,
	    "Got view state for state 0x%x, window 0x%x",
	    (void *)(&state), (void *)this );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::getViewState) leaving\n" );
}

void  ViewWindow::setViewState( const ViewWindowState &state ) {
  //cout << "[HAVE_COLLAB_VIS] (ViewWindow::setViewState) 0" << endl;
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::setViewState) entered" );
  char buffer[1000];
  snprintf( buffer, 1000,
	    "Setting view state for state 0x%x, window 0x%x",
	    (void *)(&state), (void *)this );

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::setViewState) before setting state" );
  
  viewStateLock.writeLock();
  angular_v = state.angular_v;
  *ball = state.ball;
  dolly_total = state.dolly_total;
  dolly_vector = state.dolly_vector;
  dolly_throttle = state.dolly_throttle;
  eye_dist = state.eye_dist;
  inertia_mode = state.inertia_mode;
  last_x = state.last_x;
  last_y = state.last_y;
  last_time = state.last_time;
  prev_quat[0] = state.prev_quat[0];
  prev_quat[1] = state.prev_quat[1];
  prev_quat[2] = state.prev_quat[2];
  prev_time[0] = state.prev_time[0];
  prev_time[1] = state.prev_time[1];
  prev_time[2] = state.prev_time[2];
  prev_trans = state.prev_trans;
  rot_point = state.rot_point;
  rot_point_valid = state.rot_point_valid;
  rot_view = state.rot_view;
  total_scale = state.total_scale;
  total_x = state.total_x;
  total_y = state.total_y;
  total_z = state.total_z;

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::setViewState) before checking state view" );
   
  // New 5-25-2
  if ( state.view == NULL ) {
    Log::log( WARNING, "Loading a view window with a NULL view!" );
  }
  else
    view = *state.view;
    // Old view = state.view;

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::setViewState) before logging state" );

  /*
  View v = view.get();
  snprintf( buffer, 1000, "Eye for set state 0x%x: (%f,%f,%f)",
	    (void *)(&state),
	    v.eyep().x(),
	    v.eyep().y(),
	    v.eyep().z() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  snprintf( buffer, 1000, "Lookat for set state 0x%x: (%f,%f,%f)",
	    (void *)(&state),
	    v.lookat().x(),
	    v.lookat().y(),
	    v.lookat().z() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  snprintf( buffer, 1000, "Up for set state 0x%x: (%f,%f,%f)",
	    (void *)(&state),
	    v.up().x(),
	    v.up().y(),
	    v.up().z() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  snprintf( buffer, 1000, "Fov for set state 0x%x: %f",
	    (void *)(&state),
	    v.fov() );
  Log::log( SemotusVisum::Logging::DEBUG, buffer );
  */
  
  // GUI parameters
  //set_guivar( id, "global-light", state.lighting );
  //set_guivar( id, "global-fog", state.fog );
  //set_guivar( id, "global-type", state.shading );

  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::setViewState) before writeUnlock" );
   
  viewStateLock.writeUnlock();
  snprintf( buffer, 1000,
	    "Set view state for state 0x%x, window 0x%x",
	    (void *)(&state), (void *)this );
  Log::log( SemotusVisum::Logging::DEBUG, "(ViewWindow::setViewState) leaving" );
}


Array1<GeomObj*> ViewWindow::getGeometry() {
  //cout << "[HAVE_COLLAB_VIS] (ViewWindow::getGeometry) 0" << endl;

}

// CollabVis code end
#endif


void ViewWindow::rotate(double /*angle*/, Vector /*v*/, Point /*c*/)
{
  NOT_FINISHED("ViewWindow::rotate");
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
  need_redraw=true;
}

void ViewWindow::rotate_obj(double /*angle*/, const Vector& /*v*/, const Point& /*c*/)
{
  NOT_FINISHED("ViewWindow::rotate_obj");
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
  need_redraw=true;
}

void ViewWindow::translate(Vector /*v*/)
{
  NOT_FINISHED("ViewWindow::translate");
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
  need_redraw=true;
}

void ViewWindow::scale(Vector /*v*/, Point /*c*/)
{
  NOT_FINISHED("ViewWindow::scale");
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
  need_redraw=true;
}

void ViewWindow::mouse_translate(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw();
      }
      last_x=x;
      last_y=y;
      total_x = 0;
      total_y = 0;
      update_mode_string("translate: ");
    }
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

      need_redraw=true;
      ostringstream str;
      str << "translate: " << total_x << ", " << total_y;
      update_mode_string(str.str());
    }
    break;
  case MouseEnd:
    update_mode_string("");
    break;
  }
}


// Dolly into and out-of the scene
// -- moving left/right decreases/increases the speed of motion (throttle)
// -- moving down/up moves the user into/out-of the scene
// -- throttle is *not* reset on mouse release
// -- throttle is reset on autoview
// -- throttle is normalized by the length of the diagonal of the scene's bbox 

void ViewWindow::mouse_dolly(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw();
      }
      if (dolly_throttle == 0) {
	BBox bbox;
	get_bounds(bbox);
	dolly_throttle_scale=bbox.diagonal().length()/
	  Max(current_renderer->xres, current_renderer->yres);
	dolly_throttle=1;
      }

      last_x=x;
      last_y=y;

      View tmpview(view.get());
      float curpt[2];
      NormalizeMouseXY(x, y, &curpt[0], &curpt[1]);
      dolly_vector = tmpview.lookat() - tmpview.eyep();

      // if the user clicked near the center of the screen, just move
      //   towards the lookat point, since read the z-values is sorta slow
      if (fabs(curpt[0])>.2 || fabs(curpt[1])>.2) {

	// gather the buffer's z-values
	extern int CAPTURE_Z_DATA_HACK;
	CAPTURE_Z_DATA_HACK = 1;
	redraw();

	Point pick_pt;
	if (current_renderer->pick_scene(x, y, &pick_pt)) {
	  dolly_vector = pick_pt - tmpview.eyep();
	}
      }

      dolly_vector.normalize();
      dolly_total=0;
      char str[100];
      sprintf(str, "dolly: %.3g (th=%.3g)", dolly_total,
	      dolly_throttle);
      update_mode_string(str);
    }
    break;
  case MouseMove:
    {
      double dly;
      double xmtn=last_x-x;
      double ymtn=last_y-y;
      last_x = x;
      last_y = y;

      if (Abs(xmtn)>Abs(ymtn)) {
	double scl=-xmtn/200;
	if (scl<0) scl=1/(1-scl); else scl+=1;
	dolly_throttle *= scl;
      } else {
	dly=-ymtn*(dolly_throttle*dolly_throttle_scale);
	dolly_total+=dly;
	View tmpview(view.get());
	tmpview.lookat(tmpview.lookat()+dolly_vector*dly);
	tmpview.eyep(tmpview.eyep()+dolly_vector*dly);
	view.set(tmpview);
	need_redraw=true;
      }
      char str[100];
      sprintf(str, "dolly: %.3g (th=%.3g)", dolly_total, 
	      dolly_throttle);
      update_mode_string(str);
    }
    break;
  case MouseEnd:
    update_mode_string("");
    break;
  }
}

void ViewWindow::mouse_scale(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw();
      }
      update_mode_string("scale: ");
      last_x=x;
      last_y=y;
      total_scale=1;
    }
    break;
  case MouseMove:
    {
      double scl;
      const double xmtn = (last_x-x) * 6.0 / current_renderer->xres;
      const double ymtn = (y-last_y) * 6.0 / current_renderer->yres;
      last_x = x;
      last_y = y;
      const double len = sqrt(xmtn * xmtn + ymtn * ymtn);
      if (Abs(xmtn)>Abs(ymtn)) scl = xmtn; else scl = ymtn;
      if (scl<0) scl = 1.0 / (1.0 + len); else scl = len + 1.0;
      total_scale*=scl;

      View tmpview(view.get());
      tmpview.eyep(tmpview.lookat() + (tmpview.eyep() - tmpview.lookat())*scl);

      view.set(tmpview);
      need_redraw=true;
      ostringstream str;
      str << "scale: " << 100.0/total_scale << "%";
      update_mode_string(str.str());
    }
    break;
  case MouseEnd:
    update_mode_string("");
    break;
  }
}

float ViewWindow::WindowAspect()
{
  float w = current_renderer->xres;
  float h = current_renderer->yres;

  return w/h;
}

void ViewWindow::MyTranslateCamera(Vector offset)
{
  View tmpview(view.get());

  tmpview.eyep  (tmpview.eyep  () + offset);
  tmpview.lookat(tmpview.lookat() + offset);

  view.set(tmpview);
  need_redraw=true;
}

// surprised this wasn't definied somewhere in Core?
Point operator*(Transform &t, const Point &d)
{
  float result[4], tmp[4];
  result[0] = result[1] = result[2] = result[3] = 0;
  tmp[0] = d(0);
  tmp[1] = d(1);
  tmp[2] = d(2);
  tmp[3] = 1.0;

  double mat[16];
  t.get(mat);

  for(int i=0;i<4;i++) {
    for(int j=0;j<4;j++) {
      result[i] += mat[4*i + j] * tmp[j];
    }
  }

  return Point(result[0], result[1], result[2]);
}

Vector operator*(Transform &t, const Vector &d)
{
  float result[4], tmp[4];
  result[0] = result[1] = result[2] = result[3] = 0;
  tmp[0] = d.x();
  tmp[1] = d.y();
  tmp[2] = d.z();
  tmp[3] = 0.0;

  double mat[16];
  t.get(mat);

  for(int i=0;i<4;i++) {
    for(int j=0;j<4;j++) {
      result[i] += mat[4*i + j] * tmp[j];
    }
  }

  return Vector(result[0], result[1], result[2]);
}

void ViewWindow::MyRotateCamera( Point  center,
                          Vector axis,
                          double angle )  // radians
{
  View tmpview(view.get());

  Point Origin(0,0,0);

  Transform mat;
  mat.load_identity();
  mat.pre_translate(Origin - center);
  mat.pre_rotate   (angle, axis);
  mat.pre_translate(center - Origin);

  Point  p = tmpview.eyep();
  Point  a = tmpview.lookat();
  Vector u = tmpview.up();

  tmpview.eyep  (mat * p);
  tmpview.lookat(mat * a);
  tmpview.up    (mat * u);

  view.set(tmpview);
  need_redraw=true;
}

void ViewWindow::NormalizeMouseXY( int X, int Y, float *NX, float *NY )
{
  double w = current_renderer->xres;
  double h = current_renderer->yres;

  *NX = double(X) / w;
  *NY = double(Y) / h;

  *NY = 1.0 - *NY;

  *NX = -1.0 + 2.0 * (*NX);
  *NY = -1.0 + 2.0 * (*NY);
}

Vector ViewWindow::CameraToWorld(Vector v)
{
  View tmpview(view.get());

  Vector z_axis,y_axis,x_axis;

  y_axis = tmpview.up();
  z_axis = tmpview.eyep() - tmpview.lookat();
  z_axis.normalize();
  x_axis = Cross(y_axis,z_axis);
  x_axis.normalize();
  y_axis = Cross(z_axis,x_axis);
  y_axis.normalize();

  Transform mat(tmpview.eyep(), x_axis, y_axis, z_axis);

  return mat * v;
}

void ViewWindow::unicam_choose(int X, int Y)
{
  //   MyTranslateCamera(Vector(0.1,0,0));
  //   BBox bbox;
  //   get_bounds(bbox);
  //   Point ctr = (bbox.valid() ? bbox.center() : Point(0,0,0));
  //   MyRotateCamera(ctr,
  //                  Vector(0,1,0),
  //                  1 * M_PI/180.0);

  //   cerr << CameraToWorld(Vector(1,0,0)) << "\n";
  //   cerr << CameraToWorld(Vector(0,1,0)) << "\n";
  //   cerr << CameraToWorld(Vector(0,0,1)) << "\n";

  //   float nx, ny;
  //   NormalizeMouseXY(x, y, &nx, &ny);
  //   cerr << nx << "\t" << ny << "\n";

  int   te[2];  // pixel location
  te[0] = X;
  te[1] = Y;

  float curpt[2];
  NormalizeMouseXY(X, Y, &curpt[0], &curpt[1]);
  
  float delta[2];
  delta[0] = curpt[0] - _last_pos[0];
  delta[1] = curpt[1] - _last_pos[1];
  _last_pos[0] = te[0];
  _last_pos[1] = te[1];

  double tdelt(the_time() - _dtime);

  _dist += sqrt(delta[0] * delta[0] + delta[1] * delta[1]);

  float sdelt[2];
  sdelt[0] = te[0] - _start_pix[0];
  sdelt[1] = te[1] - _start_pix[1];

  int xa=0,ya=1;
  if (getenv("FLIP_CAM_MANIP")) {
    int tmp = xa;
    xa = ya;
    ya = tmp;
  }
     
  float len = sqrt(sdelt[0] * sdelt[0] + sdelt[1] * sdelt[1]);
  if (fabs(sdelt[ya])/len > 0.9 && tdelt > 0.05) {
    unicam_state = UNICAM_ZOOM;
    //     ptr->set_old(_start_pix);
  } else if (tdelt < 0.1 && _dist < 0.03)
    return;
  else {
    if (fabs(sdelt[xa])/len > 0.6 )
      unicam_state = UNICAM_PAN;
    else unicam_state = UNICAM_ZOOM;
    //     ptr->set_old(_start_pix);
  }
}

void ViewWindow::unicam_rot(int x, int y)
{
  //  float myTEST = X;
  //  cerr << "myTEST = " << myTEST << "\t" << "X = " << X << "\n";
  Point center = focus_sphere->cen;

  //   this->ComputeWorldToDisplay(center[0], center[1], center[2], cpt);
  // XXX - this code did not seem to work to return normalized window
  // XXX - coordinates.
  // //   float cpt[3];
  // //   View tmpview(view.get());
  // //   Point tmp = tmpview.objspace_to_eyespace(center, WindowAspect());
  // //   cpt[0] = tmp(0);
  // //   cpt[1] = tmp(1);
  // //   NormalizeMouseXY(cpt[0], cpt[1], &cpt[0], &cpt[1]);
  float cpt[3];
  NormalizeMouseXY(_down_x, _down_y, &cpt[0], &cpt[1]);

  double radsq = pow(1.0+fabs(cpt[0]),2); // squared rad of virtual cylinder

  //   XYpt        tp    = ptr->old(); 
  //   XYpt        te    = ptr->cur();
  float tp[2], te[2];
  NormalizeMouseXY((int)(_last_pix[0]), (int)(_last_pix[1]), &tp[0], &tp[1]);
  NormalizeMouseXY(x, y, &te[0], &te[1]);
  _last_pix[0] = x;
  _last_pix[1] = y;

  //    Wvec   op  (tp[0], 0, 0);             // get start and end X coordinates
  //    Wvec   oe  (te[0], 0, 0);             //    of cursor motion
  float op[3], oe[3];
  op[0] = tp[0];
  op[1] = 0;
  op[2] = 0;
  oe[0] = te[0];
  oe[1] = 0;
  oe[2] = 0;

  //   double opsq = op * op, oesq = oe * oe;
  double opsq = op[0] * op[0], oesq = oe[0] * oe[0];

  double lop  = opsq > radsq ? 0 : sqrt(radsq - opsq);
  double loe  = oesq > radsq ? 0 : sqrt(radsq - oesq);

  //   Wvec   nop  = Wvec(op[0], 0, lop).normalize();
  //   Wvec   noe  = Wvec(oe[0], 0, loe).normalize();
  Vector nop = Vector(op[0], 0, lop).normal();
  Vector noe = Vector(oe[0], 0, loe).normal();

  //   double dot  = nop * noe;
  double dot = Dot(nop, noe);

  if (fabs(dot) > 0.0001) {
    //       data->rotate(Wline(data->center(), Wvec::Y),
    //                    -2*acos(clamp(dot,-1.,1.)) * Sign(te[0]-tp[0]));

    double angle = -2*acos(clamp(dot,-1.,1.)) * Sign(te[0]-tp[0]);
    MyRotateCamera(center, Vector(0,1,0), angle);


    // 2nd part of rotation
    //      View tmpview(view.get());

    //       Wvec   dvec  = data->from() - data->center();
     
    double rdist = te[1]-tp[1];
    //      Point  from = tmpview.eyep();
    //      Vector dvec = (from - center);
    //      double tdist = acos(Wvec::Y * dvec.normalize());
    //Vector Yvec(0,1,0);

    //    double tdist = acos(clamp(Dot(Yvec, dvec.normal()), -1., 1.));

    //       CAMdataptr   dd = new CAMdata(*data);
    //       Wline raxe(data->center(),data->right_v());
    //       data->rotate(raxe, rdist);
    Vector right_v = (film_pt(1, 0) - film_pt(0, 0)).normal();

    MyRotateCamera(center, right_v, rdist);

    View tmpview = view.get(); // update tmpview params given last rotation
    tmpview.up(Vector(0,1,0));
    view.set(tmpview);

    //       if (data->right_v() * dd->right_v() < 0)
    //          *data = *dd;
  }
}

void ViewWindow::unicam_zoom(int X, int Y)
{
  float cn[2], ln[2];
  NormalizeMouseXY(X, Y, &cn[0], &cn[1]);
  NormalizeMouseXY((int)(_last_pix[0]), (int)(_last_pix[1]), &ln[0], &ln[1]);

  float delta[2];
  delta[0] = cn[0] - ln[0];
  delta[1] = cn[1] - ln[1];
  _last_pix[0] = X;
  _last_pix[1] = Y;

  // PART A: Zoom in/out
  // (assume perspective projection for now..)
  View tmpview(view.get());
  
  Vector movec   = (_down_pt - tmpview.eyep());
  Vector movec_n = movec.normal(); // normalized movec

  Vector trans1  = movec_n * (movec.length() * delta[1] * -4);

  MyTranslateCamera(trans1);


  // PART B: Pan left/right.
  // Camera has moved, update tmpview..
  tmpview = view.get();

  movec   = (_down_pt - tmpview.eyep());  // (recompute since cam changed)
  Vector at_v  = film_dir(0,0);
  double depth = Dot(movec, at_v);

  Vector right_v = film_pt(1, 0, depth) - film_pt(-1, 0,depth);
  //Vector up_v    = film_pt(0, 1, depth) - film_pt( 0,-1,depth);

  Vector trans2  = right_v * (-delta[0]/2);

  MyTranslateCamera(trans2);
}

void ViewWindow::unicam_pan(int X, int Y)
{
  float cn[2], ln[2];
  NormalizeMouseXY(X, Y, &cn[0], &cn[1]);
  NormalizeMouseXY((int)(_last_pix[0]), (int)(_last_pix[1]), &ln[0], &ln[1]);

  float delta[2];
  delta[0] = cn[0] - ln[0];
  delta[1] = cn[1] - ln[1];
  _last_pix[0] = X;
  _last_pix[1] = Y;

  View tmpview(view.get());

  Vector movec   = (_down_pt - tmpview.eyep());

  Vector at_v  = film_dir(0,0);
  double depth = Dot(movec, at_v);

  Vector right_v = film_pt(1, 0, depth) - film_pt(-1, 0,depth);
  Vector up_v    = film_pt(0, 1, depth) - film_pt( 0,-1,depth);

  // add_pt(this, film_pt(0,0,depth), .01);
  // add_pt(this, film_pt(1,0,depth), .01);
  // add_pt(this, film_pt(0,1,depth), .01);

  Vector trans = (right_v * (-delta[0]/2) +
                  up_v    * (-delta[1]/2));

  MyTranslateCamera(trans);
}

void
ViewWindow::ShowFocusSphere()
{
  viewwindow_objs_draw[1] = true;
}

void
ViewWindow::HideFocusSphere()
{
  viewwindow_objs_draw[1] = false;
}

// XXX - obsolete-- delete this method below.
// This method returns the world space point under the pixel <x, y>
// XXX - in addition to drawing the simple sphere 'focus sphere', picking
// was difficult to do too.  initially the problem was understanding how
// the example in 'mouse_pick(..)' worked (parameters, what was returned).
Point ViewWindow::GetPointUnderCursor( int /* x */, int /* y */)
{
  return Point(0,0,0);
}

Vector ViewWindow::film_dir   (double x, double y)
{
  View tmpview(view.get());
  Point at = tmpview.eyespace_to_objspace(Point( x, y, 1), WindowAspect());
  return (at - tmpview.eyep()).normal();
}

Point ViewWindow::film_pt    (double x, double y, double z)
{
  View tmpview(view.get());

  Vector dir = film_dir(x,y);

  return tmpview.eyep() + dir * z;
}

void ViewWindow::mouse_unicam(int action, int x, int y, int, int, int)
{
  //   static int first=1;
  //   if (first) {
  //     first = 0;
  //     need_redraw = 1;

  //     Point l = film_pt(-1, 0, 5);
  //     Point r = film_pt( 1, 0, 5);
  //     double s = (l - r).length() / 20.0;

  //     for(int i=0;i<5;i++) {
  //       double u = double(i) / 4.0;

  //       Point p = film_pt(-1.0 + u * 2.0, 0, 5);

  //       GeomSphere *obj = scinew GeomSphere;
  //       obj->move(p, s);
  //       viewwindow_objs.add(obj);
  //     }
  //   }

  //   if (action == MouseStart) {
  //     Point p;

  //     current_renderer->pick_scene(x, y, &p);

  //     Vector at_v = (view.get().lookat() - view.get().eyep()).normal();
  //     Vector vec  = (p - view.get().eyep()) * at_v;
  //     double s = 0.008 * vec.length();

  //     GeomSphere *obj = scinew GeomSphere;
  //     obj->move(p, s);
  //     viewwindow_objs.add(obj);
    
  //     need_redraw = 1;
  //   }
  // return;    

  switch(action){
  case MouseStart:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw();
      }
      extern int CAPTURE_Z_DATA_HACK;
      CAPTURE_Z_DATA_HACK = 1;
      redraw();

      update_mode_string("unicam: ");
      last_x=x;
      last_y=y;

      _dtime    = the_time();
      _dist     = 0;

      // cam manip init
      float curpt[2];
      NormalizeMouseXY(x, y, &curpt[0], &curpt[1]);
      _last_pos[0] = curpt[0];
      _last_pos[1] = curpt[1];

      // XXX - erroneously had 'x' be a capital 'X', which was a
      // bug, but the compiler didn't catch it.  Innocent
      // mistake was not caught by the compiler for some reason,
      // caused bad behavior in user interaction, and eventually
      // was debugged.
      //             _start_pix[0] = _last_pix[0] = X; // doesn't produce error!?
      //             _start_pix[1] = _last_pix[1] = Y; // doesn't produce error!?
      _start_pix[0] = _last_pix[0] = x;
      _start_pix[1] = _last_pix[1] = y;

      // find '_down_pt'  (point in world space under the cursor tip)
      current_renderer->pick_scene(x, y, &_down_pt);
      _down_x = x;
      _down_y = y;
      //             cerr << "_down_x = " << _down_x << "\n";
      //             cerr << "_down_y = " << _down_y << "\n";
            
      // if someone has already clicked to make a dot and
      // they're not clicking on it now, OR if the user is
      // clicking on the perimeter of the screen, then we want
      // to go into rotation mode.
      if ((fabs(curpt[0]) > .85 || fabs(curpt[1]) > .9) || viewwindow_objs_draw[1]) {
	if (viewwindow_objs_draw[1])
	  _center = focus_sphere->cen;
              
	unicam_state = UNICAM_ROT;
      } else {
	unicam_state = UNICAM_CHOOSE;
      }
    }
    break;
  case MouseMove:
    {
      switch (unicam_state) {
      case UNICAM_CHOOSE:   unicam_choose(x, y); break;
      case UNICAM_ROT:      unicam_rot(x, y); break;
      case UNICAM_PAN:      unicam_pan(x, y); break;
      case UNICAM_ZOOM:     unicam_zoom(x, y); break;
      }

      need_redraw=true;

      ostringstream str;
      char *unicamMode[] = {"Choose", "Rotate", "Pan", "Zoom"};
      str << "unicam: " << unicamMode[unicam_state];
      update_mode_string(str.str());
    }
    break;

  case MouseEnd:
    if (unicam_state == UNICAM_ROT && viewwindow_objs_draw[1] ) {
      HideFocusSphere();
    } else if (unicam_state == UNICAM_CHOOSE) {
      if (viewwindow_objs_draw[1]) {
	HideFocusSphere();
      } else {
	// XXX - need to select 's' to make focus_sphere 1/4 or so
	// inches on the screen always...  how?
	Vector at_v = (view.get().lookat() - view.get().eyep()).normal();
	Vector vec  = (_down_pt - view.get().eyep()) * at_v;
	double s = 0.008 * vec.length();

	focus_sphere->move(_down_pt, s);
	ShowFocusSphere();
      }
    }
        
    need_redraw = true;

    update_mode_string("");
    break;
  }	
}

void ViewWindow::mouse_rotate(int action, int x, int y, int, int, int time)
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
      //             cerr << "zmid = " << zmid << "\n";

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
//	cerr << dt << "\n";
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

void ViewWindow::mouse_rotate_eyep(int action, int x, int y, int, int, int time)
{
  switch(action){
  case MouseStart:
    {
      if(inertia_mode){
	inertia_mode=0;
	redraw();
      }
      update_mode_string("rotate lookatp:");
      last_x=x;
      last_y=y;

      // Find the center of rotation...
      View tmpview(view.get());
      int xres=current_renderer->xres;
      int yres=current_renderer->yres;

      rot_point_valid=0;

      rot_point = tmpview.eyep();
      rot_view=tmpview;
      rot_point_valid=1;
      
      double rad = 12;
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
      HVect mouse(2.0*(xres-x)/xres - 1.0, 2.0*y/yres - 1.0, 0.0, 1.0);
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

      HVect mouse(2.0*(xres-x)/xres - 1.0, 2.0*y/yres - 1.0, 0.0, 1.0);
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
      update_mode_string("rotate lookatp:");

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
      tmpview.lookat(tmpview.eyep()-(z_a*(eye_dist)).vector());
      view.set(tmpview);
      prev_trans = prv;

      // now you need to use the history to 
      // set up the arc you want to use...

      ball->Init();
      double rad = 12;
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
//	cerr << dt << "\n";
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

// -- BAWGL -- 
//static int prevPrinc;
void ViewWindow::bawgl_pick(int action, int iv[3], GLfloat fv[3])
{
  BState bs;
  switch(action) {
  case BAWGL_PICK_START:
    {
	    
      current_renderer->get_pick(manager, this, iv[0], iv[1],
				 pick_obj, pick_pick, pick_n); 
      if (pick_obj.get_rep()){
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
      if (!pick_obj.get_rep() || !pick_pick.get_rep()) break;
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
	//prevPrinc= prin_dir;
	double dist=dir.length();
	Vector mtn(pick_pick->principal(prin_dir)*dist);
	total_x+=mtn.x();
	total_y+=mtn.y();
	total_z+=mtn.z();
	if (Abs(total_x) < .0001) total_x=0;
	if (Abs(total_y) < .0001) total_y=0;
	if (Abs(total_z) < .0001) total_z=0;
	update_mode_string(pick_obj);
	// TODO: Verify that this is the correct pick offset.
	Vector pick_offset(total_x, total_y, total_z);
	pick_pick->moved(prin_dir, dist, mtn, bs, pick_offset);
      } else {
	update_mode_string("pick: Bad direction...");
      }
    }
    break;
  case BAWGL_PICK_END:
    {
      if(pick_pick.get_rep()){
	pick_pick->release( bs );
      }
      pick_pick=0;
      pick_obj=0;
      update_mode_string("");
    }
    break;
  }
}
// -- BAWGL --

void ViewWindow::mouse_pick(int action, int x, int y, int state, int btn, int)
{
  BState bs;
  bs.shift=1; // Always for widgets...
  bs.control= ((state&4)!=0);
  bs.alt= ((state&8)!=0);
  bs.btn=btn;
  switch(action){
  case MouseStart:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw();
      }
      total_x=0;
      total_y=0;
      total_z=0;
      last_x=x;
      last_y=current_renderer->yres-y;
      pick_x = last_x;
      pick_y = last_y;
      current_renderer->get_pick(manager, this, x, y,
				 pick_obj, pick_pick, pick_n);

      if (pick_obj.get_rep()){
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
      if (!pick_obj.get_rep() || !pick_pick.get_rep()) break;
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
      const double ndx = 2.0 * (x - last_x) / (xres - 1.0);
      const double ndy = 2.0 * (y - last_y) / (yres - 1.0);
      Vector motionv(u*ndx+v*ndy);

      const double pdx = (x - pick_x) / (xres - 1.0);
      const double pdy = (y - pick_y) / (yres - 1.0);
      Vector pmotionv(u*pdx + v*pdy);

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
//	double dist=motionv.length2()/maxdot;
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
	pick_pick->moved(prin_dir, dist, mtn, bs, pmotionv);
	need_redraw=1;
      } else {
	update_mode_string("pick: Bad direction...");
      }
      last_x = x;
      last_y = y;
    }
    break;
  case MouseEnd:
    if(pick_pick.get_rep()){
      pick_pick->release( bs );
      need_redraw=1;
    }
    pick_pick=0;
    pick_obj=0;
    update_mode_string("");
    break;
  }
}

void ViewWindow::redraw_if_needed()
{
  if(need_redraw){
    need_redraw=0;
    redraw();
  }

}

void ViewWindow::tcl_command(GuiArgs& args, void*)
{
  if (args.count() < 2) {
    args.error("ViewWindow needs a minor command");
    return;
  }
  if (args[1] == "have_mpeg") {
#ifdef HAVE_MPEG
    args.result("1");
#else
    args.result("0");
#endif // HAVE_MPEG
  } else if (args[1] == "dump_viewwindow") {
    if (args.count() != 6) {
      args.error("ViewWindow::dump_viewwindow needs an output file name and type");
      return;
    }
    // We need to dispatch this one to the
    // remote thread.  We use an ID string
    // instead of a pointer in case this
    // viewwindow gets killed by the time the
    // redraw message gets dispatched.
    manager->
      mailbox.send(scinew  ViewerMessage(MessageTypes::ViewWindowDumpImage,
					 id, args[2], args[3],args[4],args[5]));
  } else if (args[1] == "startup") {
    // Fill in the visibility database...
    GeomIndexedGroup::IterIntGeomObj iter = manager->ports_.getIter();
    
    for ( ; iter.first != iter.second; iter.first++) {
      
      GeomIndexedGroup::IterIntGeomObj serIter =
	((GeomViewerPort*)((*iter.first).second.get_rep()))->getIter();
      
      for ( ; serIter.first != serIter.second; serIter.first++) {
	GeomViewerItem *si =
	  (GeomViewerItem*)((*serIter.first).second.get_rep());
	itemAdded(si);
      }
    }
  } else if (args[1] == "redraw") {
    // We need to dispatch this one to the
    // remote thread We use an ID string
    // instead of a pointer in case this
    // viewwindow gets killed by the time the
    // redraw message gets dispatched.
    if(!manager->mailbox.trySend(scinew ViewerMessage(id)))
      cerr << "Redraw event dropped, mailbox full!\n";
  } else if(args[1] == "anim_redraw"){
    // We need to dispatch this one to the
    // remote thread We use an ID string
    // instead of a pointer in case this
    // viewwindow gets killed by the time the
    // redraw message gets dispatched.
    if(args.count() != 6){
      args.error("anim_redraw wants tbeg tend nframes framerate");
      return;
    }
    double tbeg;
    if(!string_to_double(args[2], tbeg)){
      args.error("Can't figure out tbeg");
      return;
    } 
    double tend;
    if(!string_to_double(args[3], tend)){
      args.error("Can't figure out tend");
      return;
    }
    int nframes;
    if(!string_to_int(args[4], nframes)){
      args.error("Can't figure out nframes");
      return;
    }
    double framerate;
    if(!string_to_double(args[5], framerate)){
      args.error("Can't figure out framerate");
      return;
    }
    if(!manager->mailbox.trySend(scinew ViewerMessage(id, tbeg, tend,
						      nframes, framerate)))
      cerr << "Redraw event dropped, mailbox full!\n";
  } else if(args[1] == "mtranslate"){
    do_mouse(&ViewWindow::mouse_translate, args);
  } else if(args[1] == "mdolly"){
    do_mouse(&ViewWindow::mouse_dolly, args);
  } else if(args[1] == "mrotate"){
    do_mouse(&ViewWindow::mouse_rotate, args);
  } else if(args[1] == "mrotate_eyep"){
    do_mouse(&ViewWindow::mouse_rotate_eyep, args);
  } else if(args[1] == "mscale"){
    do_mouse(&ViewWindow::mouse_scale, args);
  } else if(args[1] == "municam"){
    do_mouse(&ViewWindow::mouse_unicam, args);
  } else if(args[1] == "mpick"){
    do_mouse(&ViewWindow::mouse_pick, args);
  } else if(args[1] == "sethome"){
    homeview=view.get();
  } else if(args[1] == "gohome"){
    inertia_mode=0;
    view.set(homeview);
    manager->mailbox.send(scinew ViewerMessage(id)); // Redraw
  } else if(args[1] == "autoview"){
    BBox bbox;
    inertia_mode=0;
    get_bounds(bbox);
    autoview(bbox);
  } else if(args[1] == "Snap") {
    inertia_mode=0;
    View sv(view.get());
    
    // determine closest eyep position
    Vector lookdir(sv.eyep() - sv.lookat());
    cerr << "lookdir = " << lookdir << "\n";
    double distance = lookdir.length();
    
    double x = lookdir.x();
    double y = lookdir.y();
    double z = lookdir.z();
    
    // determine closest up vector position
    if( fabs(x) > fabs(y)) {
      if( fabs(x) > fabs(z)) {
	if(lookdir.x() < 0.0) {
	  distance *= -1;
	  sv.eyep(Point(distance, 0.0, 0.0, 1.0));
	} else {
	  sv.eyep(Point(distance, 0.0, 0.0, 1.0));
	}
      } else if (fabs(z) > fabs(y)) {
	if(lookdir.z() < 0.0) {
	  distance *= -1;
	  sv.eyep(Point(0.0, 0.0, distance, 1.0));
	} else {
	  sv.eyep(Point(0.0, 0.0, distance, 1.0)); 
	}
      }
    } else if( fabs(y) > fabs(z)) {
      if(lookdir.y() < 0.0) {
	distance *= -1;
	sv.eyep(Point(0.0, distance, 0.0, 1.0));
      } else {
	sv.eyep(Point(0.0, distance, 0.0, 1.0));
      }
    } else {
      if(lookdir.z() < 0.0) {
	distance *= -1;
        sv.eyep(Point(0.0, 0.0, distance, 1.0)); 
      } else {
	sv.eyep(Point(0.0, 0.0, distance, 1.0));   
      }
    }
    
    x = sv.up().x();
    y = sv.up().y();
    z = sv.up().z();
    Vector v;
    
    // determine closest up vector position
    if( fabs(x) > fabs(y)) {
      if( fabs(x) > fabs(z)) {
	if(sv.up().x() < 0.0) {
	  v = Vector(-1.0, 0.0, 0.0);
	} else {
	  v = Vector(1.0, 0.0, 0.0);
	}
      } else if( fabs(z) > fabs(y)) {
	if(sv.up().z() < 0.0) {
	  v = Vector(0.0, 0.0, -1.0);
	} else {
	  v = Vector(0.0, 0.0, 1.0);
	}
      }
    } else if( fabs(y) > fabs(z)) {
      if(sv.up().y() < 0.0) {
	v = Vector(0.0, -1.0, 0.0);
      } else {
	v = Vector(0.0, 1.0, 0.0);
      }
    } else {
      if(sv.up().z() < 0.0) {
	v = Vector(0.0, 0.0, -1.0);
      } else {
	v = Vector(0.0, 0.0, 1.0);
      }
    }
    Vector lookdir2(sv.eyep() - sv.lookat());
    cerr << "lookdir = " << lookdir2 << "\n";
    sv.up(v);   // set the up vector
    animate_to_view(sv, 2.0); 
  } else if(args[1] == "Views") {
    View df(view.get());
    // position tells first which axis to look down 
    // (with x1 being the positive x axis and x0 being
    // the negative x axis) and then which axis is up
    // represented the same way
    string position = pos.get();
    Vector lookdir(df.eyep()-df.lookat()); 
    double distance = lookdir.length();
    if(position == "x1_y1") {
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, 1.0, 0.0));
    } else if(position == "x1_y0") {
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, -1.0, 0.0));
    } else if(position == "x1_z1") {
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, 0.0, 1.0));
    } else if(position == "x1_z0") {
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, 0.0, -1.0));
    } else if(position == "x0_y1") {
      distance *= -1;
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, 1.0, 0.0));
    } else if(position == "x0_y0") {
      distance *= -1;
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, -1.0, 0.0));
    } else if(position == "x0_z1") {
      distance *= -1;
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, 0.0, 1.0));
    } else if(position == "x0_z0") {
      distance *= -1;
      df.eyep(Point(distance, 0.0, 0.0));
      df.up(Vector(0.0, 0.0, -1.0));
    } else if(position == "y1_x1") {
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(1.0, 0.0, 0.0));
    } else if(position == "y1_x0") {
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(-1.0, 0.0, 0.0));
    } else if(position == "y1_z1") {
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(0.0, 0.0, 1.0));
    } else if(position == "y1_z0") {
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(0.0, 0.0, -1.0));
    } else if(position == "y0_x1") {
      distance *= -1;
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(1.0, 0.0, 0.0));
    } else if(position == "y0_x0") {
      distance *= -1;
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(-1.0, 0.0, 0.0));
    } else if(position == "y0_z1") {
      distance *= -1;
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(0.0, 0.0, 1.0));
    } else if(position == "y0_z0") {
      distance *= -1;
      df.eyep(Point(0.0, distance, 0.0));
      df.up(Vector(0.0, 0.0, -1.0));
    } else if(position == "z1_x1") {
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(1.0, 0.0, 0.0));
    } else if(position == "z1_x0") {
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(-1.0, 0.0, 0.0));
    } else if(position == "z1_y1") {
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(0.0, 1.0, 0.0));
    } else if(position == "z1_y0") {
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(0.0, -1.0, 0.0));
    } else if(position == "z0_x1") {
      distance *= -1;
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(1.0, 0.0, 0.0));
    } else if(position == "z0_x0") {
      distance *= -1;
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(-1.0, 0.0, 0.0));
    } else if(position == "z0_y1") {
      distance *= -1;
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(0.0, 1.0, 0.0));
    } else if(position == "z0_y0") {
      distance *= -1;
      df.eyep(Point(0.0, 0.0, distance));
      df.up(Vector(0.0, -1.0, 0.0));
    }
    animate_to_view(df, 2.0);
  } else if (args[1] == "edit_light" ){
    if (args.count() != 6) {
      args.error("ViewWindow::switch_light  needs light num, val and vector");
      return;
    }
    // We need to dispatch a message to the remote viewer thread
    // via the manager.
    bool on;
    int on_int;
    int lightNo;
    float x = 0,y = 0,z = 0; 
    float r = 0,g = 0,b = 0;

    sscanf(args[2].c_str(), "%d", &lightNo);
    sscanf(args[3].c_str(), "%d", &on_int);  on = (bool)on_int;
    sscanf(args[4].c_str(), "%f%f%f", &x, &y, &z);
    sscanf(args[5].c_str(), "%f%f%f", &r, &g, &b);

//     cerr<<"light vector "<< x <<", "<<y<<", "<<z<<endl;
//     cerr<<"color  "<< r <<", "<<g<<", "<<b<<endl;
    manager->
      mailbox.send(scinew ViewerMessage(MessageTypes::ViewWindowEditLight,
					id, lightNo, on, Vector(x,y,z),
					Color(r,g,b)));

  } else if (args[1] == "killwindow") {
    current_renderer->kill_helper();
    inertia_mode=0;
    manager->delete_viewwindow(this);
    return;
  } else if(args[1] == "saveobj") {
    if(args.count() != 6){
      args.error("ViewWindow::dump_viewwindow needs an output file name and format!");
      return;
    }
    // We need to dispatch this one to the
    // remote thread We use an ID string
    // instead of a pointer in case this
    // viewwindow gets killed by the time the
    // redraw message gets dispatched.
    manager->mailbox.send(scinew ViewerMessage(MessageTypes::ViewWindowDumpObjects,
					       id, args[2], args[3],args[4],args[5]));
  } else if(args[1] == "listvisuals"){
    current_renderer->listvisuals(args);
  } else if(args[1] == "switchvisual"){
    if(args.count() != 6){
      args.error("switchvisual needs a window name, a visual index, a width and a height");
      return;
    }
    int idx;
    if(!string_to_int(args[3], idx)){
      args.error("bad index for switchvisual");
      return;
    }
    int width;
    if(!string_to_int(args[4], width)){
      args.error("Bad width");
      return;
    }
    int height;
    if(!string_to_int(args[5], height)){
      args.error("Bad height");
      return;
    }
    current_renderer->setvisual(args[2], idx, width, height);
    // --  BAWGL -- 
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
    // --  BAWGL -- 
  } else if(args[1] == "centerGenAxes") { 
    // have to do this here, as well as in redraw() so the axes can be
    // turned on/off even while spinning with inertia
    if(caxes.get() == 1) {
      viewwindow_objs_draw[0] = 1;
    } else {
      viewwindow_objs_draw[0] = 0;
    }
  } else if(args[1] == "rotateGenAxes") { 
    // have to do this here, as well as in redraw() so the axes can be
    // turned on/off even while spinning with inertia
    show_rotation_axis = raxes.get();
  } else if(args[1] == "iconGenAxes") {    
    if(iaxes.get() == 1) {
    } else {    
    }
    // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
    //cerr << "[HAVE_COLLAB_VIS] (tcl_command) 0\n";
  } else if (args[1] == "doServer") {
    //cerr << "GET: " << view_server.get() << endl;
    if ( server->existsDataViewWindow( this ) ) {
      //cerr << "Removing data window: " << this << endl;
      server->removeDataViewWindow( this );
    }
    else {
      //cerr << "Adding data window: " << this << endl;
      server->addDataViewWindow( this );
      //cerr << "DONE!" << endl;
    }
    // CollabVis code end  
#endif
  }else
    args.error("Unknown minor command '" + args[1] + "' for ViewWindow");
}


void ViewWindow::do_mouse(MouseHandler handler, GuiArgs& args)
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
  if(!string_to_int(args[3], x)){
    args.error("error parsing x");
    return;
  }
  if(!string_to_int(args[4], y)){
    args.error("error parsing y");
    return;
  }
  int state;
  int btn;
  if(args.count() == 7){
    if(!string_to_int(args[5], state)){
      args.error("error parsing state");
      return;

    }
    if(!string_to_int(args[6], btn)){
      args.error("error parsing btn");
      return;
    }
  }
  int time;
  if(args.count() == 8){
    if(!string_to_int(args[7], time)){
      args.error("err parsing time");
      return;
    }
  }
  if(args.count() == 6){
    if(!string_to_int(args[5], time)){
      args.error("err parsing time");
      return;
    }
  }

  // We have to send this to the Viewer thread...
  if(!manager->mailbox.trySend(scinew ViewWindowMouseMessage(id, handler, action, x, y, state, btn, time)))
    cerr << "Mouse event dropped, mailbox full!\n";
}

void ViewWindow::autoview(const BBox& bbox)
{
  dolly_throttle=0;
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

void ViewWindow::redraw()
{
  need_redraw=0;
  ctx->reset();
  // Get animation variables
  double ct;
  if(!ctx->getSub("current_time", ct)){
    manager->error("Error reading current_time");
    return;
  }
  // Find out whether to draw the axes or not.  Generally, this is handled
  //  in the centerGenAxes case of the tcl_command, but for the first redraw
  //  it's needed here (can't check it in the constructor since the variable
  //  might not exist on the tcl side yet)
  viewwindow_objs_draw[0] = caxes.get();
  show_rotation_axis = raxes.get();

  current_renderer->redraw(manager, this, ct, ct, 1, 0);
}

void ViewWindow::redraw(double tbeg, double tend, int nframes, double framerate)
{
  need_redraw=0;
  ctx->reset();

  // Get animation variables
  current_renderer->redraw(manager, this, tbeg, tend, nframes, framerate);
}

void ViewWindow::update_mode_string(GeomHandle pick_obj)
{
  string ms="pick: ";
  GeomViewerItem* si=dynamic_cast<GeomViewerItem*>(pick_obj.get_rep());
  if(!si){
    ms+="not a GeomViewerItem?";
  } else {
    ms+=si->name_;
  }
  if(pick_n != 0x12345678)
    ms+=", index="+to_string(pick_n);
  update_mode_string(ms);
}

void ViewWindow::update_mode_string(const string& msg)
{
  ostringstream str;
  str << id << " updateMode \"" << msg << "\"";
  gui->execute(str.str());
}

ViewWindowMouseMessage::ViewWindowMouseMessage(const string& rid, MouseHandler handler,
				 int action, int x, int y, int state, int btn,
				 int time)
  : MessageBase(MessageTypes::ViewWindowMouse), rid(rid), handler(handler),
    action(action), x(x), y(y), state(state), btn(btn), time(time)
{
}

ViewWindowMouseMessage::~ViewWindowMouseMessage()
{
}

void ViewWindow::animate_to_view(const View& v, double /*time*/)
{
  //NOT_FINISHED("ViewWindow::animate_to_view");  // Quiet please.
  view.set(v);
  manager->mailbox.send(scinew ViewerMessage(id));
}

void ViewWindow::force_redraw()
{
  need_redraw=1;
}

void ViewWindow::do_for_visible(OpenGL* r, ViewWindowVisPMF pmf)
{
				// Do internal objects first...
  unsigned int i;
  for (i = 0; i < viewwindow_objs.size(); i++){
    if (viewwindow_objs_draw[i] == 1) {
      (r->*pmf)(manager, this, viewwindow_objs[i].get_rep());
    }
  }
  
  
  for (int pass=0; pass < 4; pass++)
  {

    GeomIndexedGroup::IterIntGeomObj iter = manager->ports_.getIter();
  
    for ( ; iter.first != iter.second; iter.first++) {
      
      GeomIndexedGroup::IterIntGeomObj serIter = 
	((GeomViewerPort*)((*iter.first).second.get_rep()))->getIter();
    
      for ( ; serIter.first != serIter.second; serIter.first++) {
	    
	GeomViewerItem *si =
	  (GeomViewerItem*)((*serIter.first).second.get_rep());
      
	// Look up the name to see if it should be drawn...
	ObjTag* vis;
      
	viter = visible.find(si->name_);
	if (viter != visible.end()) // if found
	{
	  vis = (*viter).second;
	  if (vis->visible->get())
	  {
	    const bool transparent = strstr(si->name_.c_str(), "TransParent");
	    const bool culledtext = strstr(si->name_.c_str(), "Culled Text");
	    const bool sticky = strstr(si->name_.c_str(), "Sticky");
	    if ((pass == 0 && !transparent && !culledtext && !sticky) ||
		(pass == 1 && transparent && !culledtext && !sticky) ||
		(pass == 2 && culledtext && !sticky) ||
		(pass == 3 && sticky))
	    {
	      if(si->crowd_lock_)
		si->crowd_lock_->readLock();
	      (r->*pmf)(manager, this, si);
	      if(si->crowd_lock_)
		si->crowd_lock_->readUnlock();
	    }
	  }
	}
	else
	{
	  cerr << "Warning: Object " << si->name_ <<
	    " not in visibility database.\n";
	}
      }
    }
  }
}

void ViewWindow::set_current_time(double time)
{
  ctx->setSub("current_time", to_string(time));
}

void ViewWindow::dump_objects(const string& filename, const string& format)
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
    manager->geomlock_.readLock();
    GeomScene scene(bgcolor.get(), view.get(), &manager->lighting_,
		    &manager->ports_);
    Pio(*stream, scene);
    if(stream->error()){
      cerr << "Error writing geom file: " << filename << "\n";
    } else {
      cerr << "Done writing geom file: " << filename << "\n";
    }
    delete stream;
    manager->geomlock_.readUnlock();
  } else {
    cerr << "WARNING: format " << format << " not supported!\n";
  }
}

void ViewWindow::getData(int datamask, FutureValue<GeometryData*>* result)
{
  if(current_renderer){
    cerr << "calling current_renderer->getData\n";
    current_renderer->getData(datamask, result);
    cerr << "current_renderer...\n";
  } else {
    result->send(0);
  }
}

void ViewWindow::setView(View newView) {
  view.set(newView);
  manager->mailbox.send(scinew ViewerMessage(id)); // Redraw
}

GeomHandle ViewWindow::createGenAxes() {     
  MaterialHandle dk_red = scinew Material(Color(0,0,0), Color(.2,0,0),
					  Color(.5,.5,.5), 20);
  MaterialHandle dk_green = scinew Material(Color(0,0,0), Color(0,.2,0),
					    Color(.5,.5,.5), 20);
  MaterialHandle dk_blue = scinew Material(Color(0,0,0), Color(0,0,.2),
					   Color(.5,.5,.5), 20);
  MaterialHandle lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
					  Color(.5,.5,.5), 20);
  MaterialHandle lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
					    Color(.5,.5,.5), 20);
  MaterialHandle lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
					   Color(.5,.5,.5), 20);

  GeomGroup* xp = scinew GeomGroup; 
  GeomGroup* yp = scinew GeomGroup;
  GeomGroup* zp = scinew GeomGroup;
  GeomGroup* xn = scinew GeomGroup;
  GeomGroup* yn = scinew GeomGroup;
  GeomGroup* zn = scinew GeomGroup;

  const double sz = 1.0;
  xp->add(scinew GeomCylinder(Point(0,0,0), Point(sz, 0, 0), sz/20));
  xp->add(scinew GeomCone(Point(sz, 0, 0), Point(sz+sz/5, 0, 0), sz/10, 0));
  yp->add(scinew GeomCylinder(Point(0,0,0), Point(0, sz, 0), sz/20));
  yp->add(scinew GeomCone(Point(0, sz, 0), Point(0, sz+sz/5, 0), sz/10, 0));
  zp->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, sz), sz/20));
  zp->add(scinew GeomCone(Point(0, 0, sz), Point(0, 0, sz+sz/5), sz/10, 0));
  xn->add(scinew GeomCylinder(Point(0,0,0), Point(-sz, 0, 0), sz/20));
  xn->add(scinew GeomCone(Point(-sz, 0, 0), Point(-sz-sz/5, 0, 0), sz/10, 0));
  yn->add(scinew GeomCylinder(Point(0,0,0), Point(0, -sz, 0), sz/20));
  yn->add(scinew GeomCone(Point(0, -sz, 0), Point(0, -sz-sz/5, 0), sz/10, 0));
  zn->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, -sz), sz/20));
  zn->add(scinew GeomCone(Point(0, 0, -sz), Point(0, 0, -sz-sz/5), sz/10, 0));
  GeomGroup* all=scinew GeomGroup;
  all->add(scinew GeomMaterial(xp, lt_red));
  all->add(scinew GeomMaterial(yp, lt_green));
  all->add(scinew GeomMaterial(zp, lt_blue));
  all->add(scinew GeomMaterial(xn, dk_red));
  all->add(scinew GeomMaterial(yn, dk_green));
  all->add(scinew GeomMaterial(zn, dk_blue));
  
  return all;
}

void ViewWindow::emit_vars(std::ostream& out, const std::string& midx)
{
  ctx->emit(out, midx);
}

} // End namespace SCIRun
