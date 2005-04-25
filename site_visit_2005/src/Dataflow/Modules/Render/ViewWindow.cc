/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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

#include <sci_defs/collab_vis_defs.h>
#include <sci_defs/chromium_defs.h>

#include <Dataflow/Modules/Render/Viewer.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Modules/Render/OpenGL.h>
#include <Dataflow/Modules/Render/Ball.h>
#include <Dataflow/Modules/Render/BallMath.h>
#include <Dataflow/Network/Network.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/Environment.h>
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
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/FutureValue.h>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>

using std::cerr;
using std::ostringstream;

#define MouseStart 0
#define MouseEnd 1
#define MouseMove 2

namespace SCIRun {

ViewWindow::ViewWindow(Viewer* viewer, GuiInterface* gui, GuiContext* ctx)
  : id_(ctx->getfullname()),
    inertia_mode_(0),
    ball_(scinew BallData()),
    angular_v_(0.0),
    rot_view_(),
    prev_trans_(),
    eye_dist_(0.0),
    gui_view_(ctx->subVar("view")),
    gui_sr_(ctx->subVar("sr")),
    gui_do_stereo_(ctx->subVar("do_stereo")), 
    gui_ortho_view_(ctx->subVar("ortho-view")),
    gui_track_view_window_0_(ctx->subVar("trackViewWindow0")),
    gui_raxes_(ctx->subVar("raxes")),
    gui_ambient_scale_(ctx->subVar("ambient-scale")),
    gui_diffuse_scale_(ctx->subVar("diffuse-scale")),
    gui_specular_scale_(ctx->subVar("specular-scale")),
    gui_emission_scale_(ctx->subVar("emission-scale")),
    gui_shininess_scale_(ctx->subVar("shininess-scale")),
    gui_polygon_offset_factor_(ctx->subVar("polygon-offset-factor")),
    gui_polygon_offset_units_(ctx->subVar("polygon-offset-units")),
    gui_point_size_(ctx->subVar("point-size")),
    gui_line_width_(ctx->subVar("line-width")),
    gui_sbase_(ctx->subVar("sbase")),
    gui_bgcolor_(ctx->subVar("bgcolor")), 
    gui_fogusebg_(ctx->subVar("fogusebg")),
    gui_fogcolor_(ctx->subVar("fogcolor")), 
    gui_fog_start_(ctx->subVar("fog-start")), 
    gui_fog_end_(ctx->subVar("fog-end")),
    gui_fog_visibleonly_(ctx->subVar("fog-visibleonly")),
    gui_total_frames_(ctx->subVar("total_frames", 0), 0),
    // Private Variables
    viewer_(viewer),
    renderer_(new OpenGL(gui, viewer, this)),
    gui_(gui), 
    ctx_(ctx),
    visible_(),
    obj_tag_(),
    need_redraw_(false),
    pick_n_(0),
    maxtag_(0),
    last_x_(0),
    last_y_(0),
    last_time_(0),
    mouse_action_(false),
    total_x_(0.0),
    total_y_(0.0),
    total_z_(0.0),
    total_scale_(.01),
    homeview_(Point(2.1, 1.6, 11.5), Point(.0, .0, .0), Vector(0,1,0), 20),
    rotate_valid_p_(0),
    pick_pick_(0),
    pick_obj_(0),
    viewwindow_objs_(),
    viewwindow_objs_draw_(),
    dolly_total_(0.0),
    dolly_vector_(0.0, 0.0, 0.0),
    dolly_throttle_(0.0),
    dolly_throttle_scale_(0.0),
    unicam_state_(0),
    down_x_(0),
    down_y_(0),
    down_pt_(),
    dtime_(0.0),
    uni_dist_(0.0),
    focus_sphere_(scinew GeomSphere),
    gui_current_time_(ctx->subVar("current_time",false)),
    gui_currentvisual_(ctx->subVar("currentvisual")),
    gui_caxes_(ctx->subVar("caxes")),
    gui_pos_(ctx->subVar("pos"))
{
  gui_->add_command(id_ + "-c", this, 0);

  gui_bgcolor_.set(Color(0,0,0));
  gui_view_.set(homeview_);

  ball_->Init();
  
  // 0 - Axes, visible
  viewwindow_objs_.push_back( createGenAxes() );
  viewwindow_objs_draw_.push_back(true);              

  // 1 - Unicam control sphere, not visible by default.
  MaterialHandle focus_color = scinew Material(Color(0.0, 0.0, 1.0));
  viewwindow_objs_.push_back(scinew GeomMaterial(focus_sphere_, focus_color));
  viewwindow_objs_draw_.push_back(false);

  // Clip plane variables, declare them so that they are saved out
  ctx_->subVar("clip-num");
  ctx_->subVar("clip-visible");
  ctx_->subVar("clip-selected");
  for (unsigned int i = 0; i < 6; i++)
  {
    const string istr = to_string(i+1);
    ctx_->subVar("clip-visible-" + istr);
    ctx_->subVar("clip-normal-x-" + istr);
    ctx_->subVar("clip-normal-y-" + istr);
    ctx_->subVar("clip-normal-z-" + istr);
    ctx_->subVar("clip-normal-d-" + istr);
  }

  // Lighting Variables, declare them so that they are saved out
  ctx_->subVar("global-light0");
  ctx_->subVar("global-light1");
  ctx_->subVar("global-light2");
  ctx_->subVar("global-light3");
  ctx_->subVar("lightColors");
  ctx_->subVar("lightVectors");

  // Global DrawInfo Varialbes, declare them so that they are saved
  ctx_->subVar("global-light");
  ctx_->subVar("global-fog");
  ctx_->subVar("global-debug");
  ctx_->subVar("global-clip");
  ctx_->subVar("global-cull");
  ctx_->subVar("global-dl");
  ctx_->subVar("global-type");
}

// Destructor
ViewWindow::~ViewWindow()
{
  delete renderer_;
  delete ball_;
  gui_->delete_command(id_+"-c" ); 
}

void
ViewWindow::itemAdded(GeomViewerItem* si)
{
  const string &name = si->name_;
  map<string,GuiInt*>::iterator gui_iter = visible_.find(name);
  if(gui_iter==visible_.end()){
    visible_.insert(make_pair(name,scinew GuiInt(ctx_->subVar(name))));
    if (!visible_[name]->valid()) 
      visible_[name]->set(1);
    obj_tag_[name] = maxtag_++;
  }
  gui_->eval(id_+" addObject "+to_string(obj_tag_[name])+" {"+name+"}");
  need_redraw_ = true;
}

void
ViewWindow::itemDeleted(GeomViewerItem *si)
{
  const string &name = si->name_;
  map<string,GuiInt*>::iterator gui_iter = visible_.find(name);
  if (gui_iter == visible_.end()) { // if not found
    cerr << name << " has dissappeared from the viewer\n";
  } else {
    gui_->eval(id_+" removeObject "+to_string(obj_tag_[name]));
  }
  need_redraw_=true;
}

void
ViewWindow::itemRenamed(GeomViewerItem *si, string newname)
{
  const int need_redraw_cache = need_redraw_;
  itemDeleted(si);
  si->name_ = newname;
  itemAdded(si);
  need_redraw_ = need_redraw_cache;
}

void
ViewWindow::get_bounds(BBox& bbox)
{
  bbox.reset();
  GeomIndexedGroup::IterIntGeomObj iter = viewer_->ports_.getIter();
  for ( ; iter.first != iter.second; iter.first++) {
    GeomIndexedGroup::IterIntGeomObj serIter =
      ((GeomViewerPort*)((*iter.first).second.get_rep()))->getIter();
    // items in the scene are all GeomViewerItem's...
    for ( ; serIter.first != serIter.second; serIter.first++) {
      GeomViewerItem *si=(GeomViewerItem*)((*serIter.first).second.get_rep());
      // Look up the name to see if it should be drawn...
      map<string,GuiInt*>::iterator gui_iter = visible_.find(si->name_);
      if (gui_iter != visible_.end()) { // if found
	if ((*gui_iter).second->get()) {
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
  const unsigned int objs_size = viewwindow_objs_.size();
  const unsigned int draw_size = viewwindow_objs_draw_.size();
  for(unsigned int i = 0; i < objs_size; i++) {
    if (i < draw_size && viewwindow_objs_draw_[i])
      viewwindow_objs_[i]->get_bounds(bbox);
  }
  // XXX - END   - ASF ADDED FOR UNICAM

  // If the bounding box is empty, make it default to sane view.
  if (!bbox.valid())
  {
    bbox.extend(Point(-1.0, -1.0, -1.0));
    bbox.extend(Point(1.0, 1.0, 1.0));
  }
}



// Get bounds for all the objects, not just the visible ones.  Used
// for fog computation.

void
ViewWindow::get_bounds_all(BBox& bbox)
{
  bbox.reset();
  GeomIndexedGroup::IterIntGeomObj iter = viewer_->ports_.getIter();
  for ( ; iter.first != iter.second; iter.first++) {
    GeomIndexedGroup::IterIntGeomObj serIter =
      ((GeomViewerPort*)((*iter.first).second.get_rep()))->getIter();
    // items in the scene are all GeomViewerItem's...
    for ( ; serIter.first != serIter.second; serIter.first++) {
      GeomViewerItem *si=(GeomViewerItem*)((*serIter.first).second.get_rep());
      if(si->crowd_lock_) si->crowd_lock_->readLock();
      si->get_bounds(bbox);
      if(si->crowd_lock_) si->crowd_lock_->readUnlock();
    }
  }

  // XXX - START - ASF ADDED FOR UNICAM
  const unsigned int objs_size = viewwindow_objs_.size();
  const unsigned int draw_size = viewwindow_objs_draw_.size();
  for(unsigned int i = 0; i < objs_size; i++) {
    if (i < draw_size && viewwindow_objs_draw_[i])
      viewwindow_objs_[i]->get_bounds(bbox);
  }
  // XXX - END   - ASF ADDED FOR UNICAM

  // If the bounding box is empty, make it default to sane view.
  if (!bbox.valid())
  {
    bbox.extend(Point(-1.0, -1.0, -1.0));
    bbox.extend(Point(1.0, 1.0, 1.0));
  }
}


void
ViewWindow::mouse_translate(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode_) {
	inertia_mode_=0;
	redraw();
      }
      last_x_=x;
      last_y_=y;
      total_x_ = 0;
      total_y_ = 0;
      update_mode_string("translate: ");
    }
    break;
  case MouseMove:
    {
      int xres=renderer_->xres_;
      int yres=renderer_->yres_;
      double xmtn=double(last_x_-x)/double(xres);
      double ymtn=-double(last_y_-y)/double(yres);
      last_x_ = x;
      last_y_ = y;
      // Get rid of roundoff error for the display...
      if (Abs(total_x_) < .001) total_x_ = 0;
      if (Abs(total_y_) < .001) total_y_ = 0;

      View tmpview(gui_view_.get());
      double aspect=double(xres)/double(yres);
      double znear, zfar;
      if(!renderer_->compute_depth(tmpview, znear, zfar))
	return; // No objects...
      double zmid=(znear+zfar)/2.;
      Vector u,v;
      tmpview.get_viewplane(aspect, zmid, u, v);
      double ul=u.length();
      double vl=v.length();
      Vector trans(u*xmtn+v*ymtn);

      total_x_+=ul*xmtn;
      total_y_+=vl*ymtn;

      // Translate the view...
      tmpview.eyep(tmpview.eyep()+trans);
      tmpview.lookat(tmpview.lookat()+trans);

      // Put the view back...
      gui_view_.set(tmpview);

      need_redraw_=true;
      ostringstream str;
      str << "translate: " << total_x_ << ", " << total_y_;
      update_mode_string(str.str());
    }
    break;
  case MouseEnd:
    update_mode_string("");
    need_redraw_ = 1; // this is needed for mouse-adaptive rendering
    break;
  }
}


// Dolly into and out-of the scene
// -- moving left/right decreases/increases the speed of motion (throttle)
// -- moving down/up moves the user into/out-of the scene
// -- throttle is *not* reset on mouse release
// -- throttle is reset on autoview
// -- throttle is normalized by the length of the diagonal of the scene's bbox 
void
ViewWindow::mouse_dolly(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode_) {
	inertia_mode_=0;
	redraw();
      }
      if (dolly_throttle_ == 0) {
	BBox bbox;
	get_bounds(bbox);
	dolly_throttle_scale_=bbox.diagonal().length()/
	  Max(renderer_->xres_, renderer_->yres_);
	dolly_throttle_=1;
      }

      last_x_ = x;
      last_y_ = y;

      View tmpview(gui_view_.get());
      float curpt[2];
      NormalizeMouseXY(x, y, &curpt[0], &curpt[1]);
      dolly_vector_ = tmpview.lookat() - tmpview.eyep();

      // if the user clicked near the center of the screen, just move
      //   towards the lookat point, since read the z-values is sorta slow
      if (fabs(curpt[0])>.2 || fabs(curpt[1])>.2) {

	// gather the buffer's z-values
	extern int CAPTURE_Z_DATA_HACK;
	CAPTURE_Z_DATA_HACK = 1;
	redraw();

	Point pick_pt;
	if (renderer_->pick_scene(x, y, &pick_pt)) {
	  dolly_vector_ = pick_pt - tmpview.eyep();
	}
      }

      dolly_vector_.normalize();
      dolly_total_=0;
      char str[100];
      sprintf(str, "dolly: %.3g (th=%.3g)", dolly_total_, dolly_throttle_);
      update_mode_string(str);
    }
    break;
  case MouseMove:
    {
      double dly;
      double xmtn=last_x_-x;
      double ymtn=last_y_-y;
      last_x_ = x;
      last_y_ = y;

      if (Abs(xmtn)>Abs(ymtn)) {
	double scl=-xmtn/200;
	if (scl<0) scl=1/(1-scl); else scl+=1;
	dolly_throttle_ *= scl;
      } else {
	dly=-ymtn*(dolly_throttle_*dolly_throttle_scale_);
	dolly_total_+=dly;
	View tmpview(gui_view_.get());
	tmpview.lookat(tmpview.lookat()+dolly_vector_*dly);
	tmpview.eyep(tmpview.eyep()+dolly_vector_*dly);
	gui_view_.set(tmpview);
	need_redraw_=true;
      }
      char str[100];
      sprintf(str, "dolly: %.3g (th=%.3g)", dolly_total_, dolly_throttle_);
      update_mode_string(str);
    }
    break;
  case MouseEnd:
    update_mode_string("");
    need_redraw_ = 1;
    break;
  }
}

void
ViewWindow::mouse_scale(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode_) {
	inertia_mode_=0;
	redraw();
      }
      update_mode_string("scale: ");
      last_x_=x;
      last_y_=y;
      total_scale_=1.0;
    }
    break;
  case MouseMove:
    {
      double scl;
      const double xmtn = (last_x_-x) * 6.0 / renderer_->xres_;
      const double ymtn = (last_y_-y) * 6.0 / renderer_->yres_;
      last_x_ = x;
      last_y_ = y;
      const double len = sqrt(xmtn * xmtn + ymtn * ymtn);
      if (Abs(xmtn)>Abs(ymtn)) scl = xmtn; else scl = ymtn;
      if (scl<0) scl = 1.0 / (1.0 + len); else scl = len + 1.0;
      total_scale_*=scl;

      View tmpview(gui_view_.get());
      tmpview.eyep(tmpview.lookat() + (tmpview.eyep() - tmpview.lookat())*scl);

      gui_view_.set(tmpview);
      need_redraw_=true;
      ostringstream str;
      str << "scale: " << 100.0/total_scale_ << "%";
      update_mode_string(str.str());
    }
    break;
  case MouseEnd:
    update_mode_string("");
    need_redraw_ = 1;
    break;
  }
}

float
ViewWindow::WindowAspect()
{
  float w = renderer_->xres_;
  float h = renderer_->yres_;

  return w/h;
}

void
ViewWindow::MyTranslateCamera(Vector offset)
{
  View tmpview(gui_view_.get());

  tmpview.eyep  (tmpview.eyep  () + offset);
  tmpview.lookat(tmpview.lookat() + offset);

  gui_view_.set(tmpview);
  need_redraw_=true;
}

void
ViewWindow::MyRotateCamera(Point  center, Vector axis, double angle) // radians
{
  View tmpview(gui_view_.get());

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

  gui_view_.set(tmpview);
  need_redraw_=true;
}

void
ViewWindow::NormalizeMouseXY(int X, int Y, float *NX, float *NY)
{
  *NX = -1.0 + 2.0 * double(X) / double(renderer_->xres_);
  *NY =  1.0 - 2.0 * double(Y) / double(renderer_->yres_);
}

void
ViewWindow::UnNormalizeMouseXY(float NX, float NY, int *X, int *Y )
{
  *X = Round((NX + 1.0) * double(renderer_->xres_) / 2.0);
  *Y = Round((1.0 - NY) * double(renderer_->yres_) / 2.0);
}

Vector
ViewWindow::CameraToWorld(Vector v)
{
  View tmpview(gui_view_.get());

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

void
ViewWindow::unicam_choose(int X, int Y)
{
  int   te[2];  // pixel location
  te[0] = X;
  te[1] = Y;

  float curpt[2];
  NormalizeMouseXY(X, Y, &curpt[0], &curpt[1]);
  
  float delta[2];
  delta[0] = curpt[0] - last_pos_[0];
  delta[1] = curpt[1] - last_pos_[1];
  last_pos_[0] = te[0];
  last_pos_[1] = te[1];

  double tdelt(the_time() - dtime_);

  uni_dist_ += sqrt(delta[0] * delta[0] + delta[1] * delta[1]);

  float sdelt[2];
  sdelt[0] = te[0] - start_pix_[0];
  sdelt[1] = te[1] - start_pix_[1];

  int xa=0,ya=1;
  if (sci_getenv("FLIP_CAM_MANIP")) {
    int tmp = xa;
    xa = ya;
    ya = tmp;
  }
     
  float len = sqrt(sdelt[0] * sdelt[0] + sdelt[1] * sdelt[1]);
  if (fabs(sdelt[ya])/len > 0.9 && tdelt > 0.05) {
    unicam_state_ = UNICAM_ZOOM;
    //     ptr->set_old(start_pix_);
  } else if (tdelt < 0.1 && uni_dist_ < 0.03)
    return;
  else {
    if (fabs(sdelt[xa])/len > 0.6 )
      unicam_state_ = UNICAM_PAN;
    else unicam_state_ = UNICAM_ZOOM;
    //     ptr->set_old(start_pix_);
  }
}

void
ViewWindow::unicam_rot(int x, int y)
{
  Point center = focus_sphere_->cen;
  float cpt[3];
  NormalizeMouseXY(down_x_, down_y_, &cpt[0], &cpt[1]);
  double radsq = pow(1.0+fabs(cpt[0]),2); // squared rad of virtual cylinder
  float tp[2], te[2];
  NormalizeMouseXY((int)(last_pix_[0]), (int)(last_pix_[1]), &tp[0], &tp[1]);
  NormalizeMouseXY(x, y, &te[0], &te[1]);
  last_pix_[0] = x;
  last_pix_[1] = y;
  float op[3], oe[3];
  op[0] = tp[0];
  op[1] = 0;
  op[2] = 0;
  oe[0] = te[0];
  oe[1] = 0;
  oe[2] = 0;
  double opsq = op[0] * op[0], oesq = oe[0] * oe[0];
  double lop  = opsq > radsq ? 0 : sqrt(radsq - opsq);
  double loe  = oesq > radsq ? 0 : sqrt(radsq - oesq);
  Vector nop = Vector(op[0], 0, lop).normal();
  Vector noe = Vector(oe[0], 0, loe).normal();
  double dot = Dot(nop, noe);

  if (fabs(dot) > 0.0001) {
    double angle = -2*acos(Clamp(dot,-1.,1.)) * Sign(te[0]-tp[0]);
    MyRotateCamera(center, Vector(0,1,0), angle);
    double rdist = te[1]-tp[1];
    Vector right_v = (film_pt(1, 0) - film_pt(0, 0)).normal();
    MyRotateCamera(center, right_v, rdist);
    View tmpview = gui_view_.get(); // update tmpview given last rotation
    tmpview.up(Vector(0,1,0));
    gui_view_.set(tmpview);
  }
}

void
ViewWindow::unicam_zoom(int X, int Y)
{
  float cn[2], ln[2];
  NormalizeMouseXY(X, Y, &cn[0], &cn[1]);
  NormalizeMouseXY((int)(last_pix_[0]), (int)(last_pix_[1]), &ln[0], &ln[1]);
  float delta[2];
  delta[0] = cn[0] - ln[0];
  delta[1] = cn[1] - ln[1];
  last_pix_[0] = X;
  last_pix_[1] = Y;

  // PART A: Zoom in/out (assume perspective projection for now..)
  View tmpview(gui_view_.get());
  Vector movec   = (down_pt_ - tmpview.eyep());
  Vector movec_n = movec.normal(); // normalized movec
  Vector trans1  = movec_n * (movec.length() * delta[1] * -4);
  MyTranslateCamera(trans1);

  // PART B: Pan left/right. Camera has moved, update tmpview..
  tmpview = gui_view_.get();
  movec   = (down_pt_ - tmpview.eyep());  // (recompute since cam changed)
  Vector at_v  = film_dir(0,0);
  double depth = Dot(movec, at_v);
  Vector right_v = film_pt(1, 0, depth) - film_pt(-1, 0,depth);
  Vector trans2  = right_v * (-delta[0]/2);
  MyTranslateCamera(trans2);
}

void
ViewWindow::unicam_pan(int X, int Y)
{
  float cn[2], ln[2];
  NormalizeMouseXY(X, Y, &cn[0], &cn[1]);
  NormalizeMouseXY((int)(last_pix_[0]), (int)(last_pix_[1]), &ln[0], &ln[1]);
  float delta[2];
  delta[0] = cn[0] - ln[0];
  delta[1] = cn[1] - ln[1];
  last_pix_[0] = X;
  last_pix_[1] = Y;
  View tmpview(gui_view_.get());
  Vector movec   = (down_pt_ - tmpview.eyep());
  Vector at_v  = film_dir(0,0);
  double depth = Dot(movec, at_v);
  Vector right_v = film_pt(1, 0, depth) - film_pt(-1, 0,depth);
  Vector up_v    = film_pt(0, 1, depth) - film_pt( 0,-1,depth);
  Vector trans = (right_v * (-delta[0]/2) + up_v*(-delta[1]/2));
  MyTranslateCamera(trans);
}

void
ViewWindow::ShowFocusSphere()
{
  viewwindow_objs_draw_[1] = true;
}

void
ViewWindow::HideFocusSphere()
{
  viewwindow_objs_draw_[1] = false;
}

Vector
ViewWindow::film_dir(double x, double y)
{
  View tmpview(gui_view_.get());
  Point at = tmpview.eyespace_to_objspace(Point( x, y, 1), WindowAspect());
  return (at - tmpview.eyep()).normal();
}

Point
ViewWindow::film_pt(double x, double y, double z)
{
  View tmpview(gui_view_.get());
  Vector dir = film_dir(x,y);
  return tmpview.eyep() + dir * z;
}

void
ViewWindow::mouse_unicam(int action, int x, int y, int, int, int)
{
  switch(action){
  case MouseStart:
    {
      if (inertia_mode_) {
	inertia_mode_=0;
	redraw();
      }
      extern int CAPTURE_Z_DATA_HACK;
      CAPTURE_Z_DATA_HACK = 1;
      redraw();

      update_mode_string("unicam: ");
      last_x_ = x;
      last_y_ = y;
      dtime_    = the_time();

      // cam manip init
      float curpt[2];
      NormalizeMouseXY(x, y, &curpt[0], &curpt[1]);
      last_pos_[0] = curpt[0];
      last_pos_[1] = curpt[1];
      start_pix_[0] = last_pix_[0] = x;
      start_pix_[1] = last_pix_[1] = y;
      // find 'down_pt_'  (point in world space under the cursor tip)
      renderer_->pick_scene(x, y, &down_pt_);
      down_x_ = x;
      down_y_ = y;
      // if someone has already clicked to make a dot and
      // they're not clicking on it now, OR if the user is
      // clicking on the perimeter of the screen, then we want
      // to go into rotation mode.
      if ((fabs(curpt[0]) > .85 || fabs(curpt[1]) > .9) || 
	  viewwindow_objs_draw_[1]) {
	unicam_state_ = UNICAM_ROT;
      } else {
	unicam_state_ = UNICAM_CHOOSE;
      }
    }
    break;
  case MouseMove:
    {
      switch (unicam_state_) {
      case UNICAM_CHOOSE:   unicam_choose(x, y); break;
      case UNICAM_ROT:      unicam_rot(x, y); break;
      case UNICAM_PAN:      unicam_pan(x, y); break;
      case UNICAM_ZOOM:     unicam_zoom(x, y); break;
      }
      need_redraw_=true;
      ostringstream str;
      char *unicamMode[] = {"Choose", "Rotate", "Pan", "Zoom"};
      str << "unicam: " << unicamMode[unicam_state_];
      update_mode_string(str.str());
    }
    break;

  case MouseEnd:
    if (unicam_state_ == UNICAM_ROT && viewwindow_objs_draw_[1] ) {
      HideFocusSphere();
    } else if (unicam_state_ == UNICAM_CHOOSE) {
      if (viewwindow_objs_draw_[1]) {
	HideFocusSphere();
      } else {
	// XXX - need to select 's' to make focus_sphere_ 1/4 or so
	// inches on the screen always...  how?
	Vector at_v=(gui_view_.get().lookat()-gui_view_.get().eyep()).normal();
	Vector vec  = (down_pt_ - gui_view_.get().eyep()) * at_v;
	double s = 0.008 * vec.length();
	focus_sphere_->move(down_pt_, s);
	ShowFocusSphere();
      }
    }
    need_redraw_ = true;
    update_mode_string("");
    break;
  }	
}

void
ViewWindow::mouse_rotate(int action, int x, int y, int, int, int time)
{
  switch(action){
  case MouseStart:
    {
      if(inertia_mode_){
	inertia_mode_=0;
	redraw();
      }
      update_mode_string("rotate:");
      last_x_ = x;
      last_y_ = y;

      // Find the center of rotation...
      View tmpview(gui_view_.get());
      int xres=renderer_->xres_;
      int yres=renderer_->yres_;
      double znear, zfar;
      rotate_valid_p_=0;
      if(!renderer_->compute_depth(tmpview, znear, zfar))
	return; // No objects...
      //double zmid=(znear+zfar)/2.;

      rot_view_=tmpview;
      rotate_valid_p_=1;

      double rad = 0.8;
      HVect center(0,0,0,1.0);
	
      // we also want to keep the old transform information
      // around (so stuff correlates correctly)
      // OGL uses left handed coordinate system!
      Vector z_axis,y_axis,x_axis;

      y_axis = tmpview.up();
      z_axis = tmpview.eyep() - tmpview.lookat();
      eye_dist_ = z_axis.normalize();
      x_axis = Cross(y_axis,z_axis);
      x_axis.normalize();
      y_axis = Cross(z_axis,x_axis);
      y_axis.normalize();
      tmpview.up(y_axis); // having this correct could fix something?

      prev_trans_.load_frame(Point(0.0,0.0,0.0),x_axis,y_axis,z_axis);

      ball_->Init();
      ball_->Place(center,rad);
      HVect mouse((2.0*x)/xres - 1.0,2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
      ball_->Mouse(mouse);
      ball_->BeginDrag();

      prev_time_[0] = time;
      prev_quat_[0] = mouse;
      prev_time_[1] = prev_time_[2] = -100;
      ball_->Update();
      last_time_=time;
      inertia_mode_=0;
      need_redraw_ = 1;
    }
    break;
  case MouseMove:
    {
      int xres=renderer_->xres_;
      int yres=renderer_->yres_;

      if(!rotate_valid_p_)
	break;

      HVect mouse((2.0*x)/xres - 1.0,2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
      prev_time_[2] = prev_time_[1];
      prev_time_[1] = prev_time_[0];
      prev_time_[0] = time;
      ball_->Mouse(mouse);
      ball_->Update();

      prev_quat_[2] = prev_quat_[1];
      prev_quat_[1] = prev_quat_[0];
      prev_quat_[0] = mouse;

      // now we should just send the view points through the rotation
      // (after centerd around the ball), eyep, lookat, and up
      View tmpview(rot_view_);

      Transform tmp_trans;
      HMatrix mNow;
      ball_->Value(mNow);
      tmp_trans.set(&mNow[0][0]);

      Transform prv = prev_trans_;
      prv.post_trans(tmp_trans);

      HMatrix vmat;
      prv.get(&vmat[0][0]);

      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);

      tmpview.up(y_a.vector());
      tmpview.eyep((z_a*(eye_dist_)) + tmpview.lookat().vector());

      gui_view_.set(tmpview);
      need_redraw_=1;
      update_mode_string("rotate:");

      last_time_=time;
      inertia_mode_=0;
    }
    break;
  case MouseEnd:
    if(time-last_time_ < 20){
      // now setup the normalized quaternion
      View tmpview(rot_view_);
	    
      Transform tmp_trans;
      HMatrix mNow;
      ball_->Value(mNow);
      tmp_trans.set(&mNow[0][0]);
	    
      Transform prv = prev_trans_;
      prv.post_trans(tmp_trans);
	    
      HMatrix vmat;
      prv.get(&vmat[0][0]);
	    
      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	    
      tmpview.up(y_a.vector());
      tmpview.eyep((z_a*(eye_dist_)) + tmpview.lookat().vector());
	    
      gui_view_.set(tmpview);
      prev_trans_ = prv;

      // now you need to use the history to 
      // set up the arc you want to use...
      ball_->Init();
      double rad = 0.8;
      HVect center(0,0,0,1.0);

      ball_->Place(center,rad);

      int index=2;

      if (prev_time_[index] == -100)
	index = 1;

      ball_->vDown = prev_quat_[index];
      ball_->vNow  = prev_quat_[0];
      ball_->dragging = 1;
      ball_->Update();
	    
      ball_->qNorm = ball_->qNow.Conj();
      double mag = ball_->qNow.VecMag();

      // Go into inertia mode...
      inertia_mode_=1;
      need_redraw_=1;

      if (mag < 0.00001) { // arbitrary ad-hoc threshold
	inertia_mode_ = 0;
	need_redraw_ = 1;
      }
      else {
	double c = 1.0/mag;
	// time between last 2 events
	double dt = prev_time_[0] - prev_time_[index];
	ball_->qNorm.x *= c;
	ball_->qNorm.y *= c;
	ball_->qNorm.z *= c;
	angular_v_ = 2*acos(ball_->qNow.w)*1000.0/dt;
      }
    } else {
      inertia_mode_=0;
    }
    ball_->EndDrag();
    rotate_valid_p_ = 0; // so we don't have to draw this...
    need_redraw_ = 1;     // always update this...
    update_mode_string("");
    break;
  }
}

void
ViewWindow::mouse_rotate_eyep(int action, int x, int y, int, int, int time)
{
  switch(action){
  case MouseStart:
    {
      if(inertia_mode_){
	inertia_mode_=0;
	redraw();
      }
      update_mode_string("rotate lookatp:");
      last_x_ = x;
      last_y_ = y;

      // Find the center of rotation...
      View tmpview(gui_view_.get());
      int xres=renderer_->xres_;
      int yres=renderer_->yres_;

      rotate_valid_p_=0;

      rot_view_=tmpview;
      rotate_valid_p_=1;
      
      double rad = 12;
      HVect center(0,0,0,1.0);
	
      // we also want to keep the old transform information
      // around (so stuff correlates correctly)
      // OGL uses left handed coordinate system!
      Vector z_axis,y_axis,x_axis;

      y_axis = tmpview.up();
      z_axis = tmpview.eyep() - tmpview.lookat();
      eye_dist_ = z_axis.normalize();
      x_axis = Cross(y_axis,z_axis);
      x_axis.normalize();
      y_axis = Cross(z_axis,x_axis);
      y_axis.normalize();
      tmpview.up(y_axis); // having this correct could fix something?

      prev_trans_.load_frame(Point(0.0,0.0,0.0),x_axis,y_axis,z_axis);

      ball_->Init();
      ball_->Place(center,rad);
      HVect mouse(2.0*(xres-x)/xres - 1.0, 2.0*y/yres - 1.0, 0.0, 1.0);
      ball_->Mouse(mouse);
      ball_->BeginDrag();

      prev_time_[0] = time;
      prev_quat_[0] = mouse;
      prev_time_[1] = prev_time_[2] = -100;
      ball_->Update();
      last_time_=time;
      inertia_mode_=0;
      need_redraw_ = 1;
    }
    break;
  case MouseMove:
    {
      int xres=renderer_->xres_;
      int yres=renderer_->yres_;

      if(!rotate_valid_p_)
	break;

      HVect mouse(2.0*(xres-x)/xres - 1.0, 2.0*y/yres - 1.0, 0.0, 1.0);
      prev_time_[2] = prev_time_[1];
      prev_time_[1] = prev_time_[0];
      prev_time_[0] = time;
      ball_->Mouse(mouse);
      ball_->Update();

      prev_quat_[2] = prev_quat_[1];
      prev_quat_[1] = prev_quat_[0];
      prev_quat_[0] = mouse;

      // now we should just sendthe view points through
      // the rotation (after centerd around the ball)
      // eyep lookat and up
      View tmpview(rot_view_);

      Transform tmp_trans;
      HMatrix mNow;
      ball_->Value(mNow);
      tmp_trans.set(&mNow[0][0]);

      Transform prv = prev_trans_;
      prv.post_trans(tmp_trans);

      HMatrix vmat;
      prv.get(&vmat[0][0]);

      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);

      tmpview.up(y_a.vector());
      tmpview.lookat(tmpview.eyep()-(z_a*(eye_dist_)).vector());
      gui_view_.set(tmpview);
      need_redraw_=1;
      update_mode_string("rotate lookatp:");

      last_time_=time;
      inertia_mode_=0;
    }
    break;
  case MouseEnd:
    if(time-last_time_ < 20){
      // now setup the normalized quaternion
      View tmpview(rot_view_);
	    
      Transform tmp_trans;
      HMatrix mNow;
      ball_->Value(mNow);
      tmp_trans.set(&mNow[0][0]);
	    
      Transform prv = prev_trans_;
      prv.post_trans(tmp_trans);
	    
      HMatrix vmat;
      prv.get(&vmat[0][0]);
	    
      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	    
      tmpview.up(y_a.vector());
      tmpview.lookat(tmpview.eyep()-(z_a*(eye_dist_)).vector());
      gui_view_.set(tmpview);
      prev_trans_ = prv;

      // now you need to use the history to 
      // set up the arc you want to use...

      ball_->Init();
      double rad = 12;
      HVect center(0,0,0,1.0);

      ball_->Place(center,rad);

      int index=2;

      if (prev_time_[index] == -100)
	index = 1;

      ball_->vDown = prev_quat_[index];
      ball_->vNow  = prev_quat_[0];
      ball_->dragging = 1;
      ball_->Update();
	    
      ball_->qNorm = ball_->qNow.Conj();
      double mag = ball_->qNow.VecMag();

      // Go into inertia mode...
      inertia_mode_=2;
      need_redraw_=1;

      if (mag < 0.00001) { // arbitrary ad-hoc threshold
	inertia_mode_ = 0;
	need_redraw_ = 1;
      }
      else {
	double c = 1.0/mag;
	// time between last 2 events
	double dt = prev_time_[0] - prev_time_[index];
	ball_->qNorm.x *= c;
	ball_->qNorm.y *= c;
	ball_->qNorm.z *= c;
	angular_v_ = 2*acos(ball_->qNow.w)*1000.0/dt;
      }
    } else {
      inertia_mode_=0;
    }
    ball_->EndDrag();
    rotate_valid_p_ = 0; // so we don't have to draw this...
    need_redraw_ = 1;     // always update this...
    update_mode_string("");
    break;
  }
}


void
ViewWindow::mouse_pick(int action, int x, int y, int state, int btn, int)
{
  BState bs;
  bs.shift=1; // Always for widgets...
  bs.control= ((state&4)!=0);
  bs.alt= ((state&8)!=0);
  bs.btn=btn;
  switch(action){
  case MouseStart:
    {
      if (inertia_mode_) {
	inertia_mode_=0;
	redraw();
      }
      total_x_=0;
      total_y_=0;
      total_z_=0;
      last_x_ = x;
      last_y_ = renderer_->yres_-y;
      pick_x_ = last_x_;
      pick_y_ = last_y_;
      renderer_->get_pick(x, y, pick_obj_, pick_pick_, pick_n_);

      if (pick_obj_.get_rep()){
	update_mode_string(pick_obj_);
	pick_pick_->set_picked_obj(pick_obj_);
	pick_pick_->pick(this,bs);

	need_redraw_=1;
      } else {
	update_mode_string("pick: none");
      }
    }
    break;
  case MouseMove:
    {
      if (!pick_obj_.get_rep() || !pick_pick_.get_rep()) break;
      // project the center of the item grabbed onto the screen -- take the z
      // component and unprojec the last and current x, y locations to get a 
      // vector in object space.
      y=renderer_->yres_-y;
      BBox itemBB;
      pick_obj_->get_bounds(itemBB);
      View tmpview(gui_view_.get());
      Point cen(itemBB.center());
      double depth=tmpview.depth(cen);
      Vector u,v;
      int xres=renderer_->xres_;
      int yres=renderer_->yres_;
      double aspect=double(xres)/double(yres);
      tmpview.get_viewplane(aspect, depth, u, v);
      const double ndx = 2.0 * (x - last_x_) / (xres - 1.0);
      const double ndy = 2.0 * (y - last_y_) / (yres - 1.0);
      Vector motionv(u*ndx+v*ndy);

      const double pdx = (x - pick_x_) / (xres - 1.0);
      const double pdy = (y - pick_y_) / (yres - 1.0);
      Vector pmotionv(u*pdx + v*pdy);

      double maxdot=0;
      int prin_dir=-1;
      for (int i=0; i<pick_pick_->nprincipal(); i++) {
	double pdot=Dot(motionv, pick_pick_->principal(i));
	if(pdot > maxdot){
	  maxdot=pdot;
	  prin_dir=i;
	}
      }
      if(prin_dir != -1){
	//	double dist=motionv.length2()/maxdot;
        double dist=motionv.length();
	Vector mtn(pick_pick_->principal(prin_dir)*dist);
	total_x_+=mtn.x();
	total_y_+=mtn.y();
	total_z_+=mtn.z();
	if (Abs(total_x_) < .0001) total_x_=0;
	if (Abs(total_y_) < .0001) total_y_=0;
	if (Abs(total_z_) < .0001) total_z_=0;
	need_redraw_=1;
	update_mode_string(pick_obj_);
	pick_pick_->moved(prin_dir, dist, mtn, bs, pmotionv);
	need_redraw_=1;
      } else {
	update_mode_string("pick: Bad direction...");
      }
      last_x_ = x;
      last_y_ = y;
    }
    break;
  case MouseEnd:
    if(pick_pick_.get_rep()){
      pick_pick_->release( bs );
      need_redraw_=1;
    }
    pick_pick_=0;
    pick_obj_=0;
    update_mode_string("");
    break;
  }
}

void
ViewWindow::redraw_if_needed()
{
  if( need_redraw_ ){
    need_redraw_ = false;
    redraw();
  }
}


// Used by "redraw" tcl_command.  We check to see if there is aleady a
// redraw message on the mailbox queue before we bother to add a new
// one.
static bool
check_for_redraw_msg(MessageBase *const& a, MessageBase *const& b)
{
  if (a->type == MessageTypes::ViewWindowRedraw &&
      b->type == MessageTypes::ViewWindowRedraw)
  {
    ViewerMessage *av = (ViewerMessage *)a;
    ViewerMessage *bv = (ViewerMessage *)b;
    if (av->rid == bv->rid)
    {
      return false;
    }
  }
  return true;
}


void
ViewWindow::tcl_command(GuiArgs& args, void*)
{
  if (args.count() < 2) {
    args.error("ViewWindow needs a minor command");
    return;
  }
  if (args[1] == "have_mpeg") { // TODO: Should be moved to environment vars
#ifdef HAVE_MPEG
    args.result("1");
#else
    args.result("0");
#endif // HAVE_MPEG
  } else if (args[1] == "dump_viewwindow") {
    if (args.count() != 6) {
      args.error("ViewWindow::dump_viewwindow needs output filename and type");
      return;
    }
    // We need to dispatch this one to the remote thread.  We use an ID string
    // instead of a pointer in case this viewwindow gets killed by the time the
    // redraw message gets dispatched.
    ViewerMessage *msg = scinew ViewerMessage
      (MessageTypes::ViewWindowDumpImage,id_,args[2], args[3],args[4],args[5]);
    viewer_->mailbox.send(msg);
  } else if (args[1] == "startup") {
    // Fill in the visibility database...
    GeomIndexedGroup::IterIntGeomObj iter = viewer_->ports_.getIter();
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
#ifdef __APPLE__
    if (args.count() > 2) renderer_->apple_wait_a_second_=true;
#endif
    // We need to dispatch this one to the  remote thread We use an ID string
    // instead of a pointer in case this viewwindow gets killed by the time the
    // redraw message gets dispatched.
    ViewerMessage *tmp = scinew ViewerMessage(id_);
    if (!viewer_->mailbox.sendIfNotSentLast(tmp, check_for_redraw_msg))
    {
      // Message wasn't needed, delete it.
      delete tmp;
    }
  } else if(args[1] == "anim_redraw"){
    // We need to dispatch this one to the remote thread We use an ID string
    // instead of a pointer in case this viewwindow gets killed by the time the
    // redraw message gets dispatched.
    if(args.count() != 6) {
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
    int num;
    if(!string_to_int(args[4], num)){
      args.error("Can't figure out num");
      return;
    }
    double framerate;
    if(!string_to_double(args[5], framerate)){
      args.error("Can't figure out framerate");
      return;
    }
    ViewerMessage *msg = scinew ViewerMessage(id_, tbeg, tend, num, framerate);
    if(!viewer_->mailbox.trySend(msg))
       cerr << "Redraw event dropped, mailbox full!\n";
  } else if(args[1] == "mtranslate") {
    do_mouse(&ViewWindow::mouse_translate, args);
  } else if(args[1] == "mdolly") {
    do_mouse(&ViewWindow::mouse_dolly, args);
  } else if(args[1] == "mrotate") {
    do_mouse(&ViewWindow::mouse_rotate, args);
  } else if(args[1] == "mrotate_eyep") {
    do_mouse(&ViewWindow::mouse_rotate_eyep, args);
  } else if(args[1] == "mscale") {
    do_mouse(&ViewWindow::mouse_scale, args);
  } else if(args[1] == "municam") {
    do_mouse(&ViewWindow::mouse_unicam, args);
  } else if(args[1] == "mpick") {
    do_mouse(&ViewWindow::mouse_pick, args);
  } else if(args[1] == "sethome") {
    homeview_=gui_view_.get();
  } else if(args[1] == "gohome") {
    inertia_mode_=0;
    gui_view_.set(homeview_);
    viewer_->mailbox.send(scinew ViewerMessage(id_)); // Redraw
  } else if(args[1] == "autoview") {
    BBox bbox;
    inertia_mode_=0;
    get_bounds(bbox);
    autoview(bbox);
  } else if(args[1] == "Views") {
    View df(gui_view_.get());
    // position tells first which axis to look down 
    // (with x1 being the positive x axis and x0 being
    // the negative x axis) and then which axis is up
    // represented the same way
    gui_pos_.reset();
    string position = gui_pos_.get();
    const string prefix("ViewWindow");
    args.result("");
    if (position.find(prefix) == 0) {
      int vw;
      const string str(position, prefix.size(), position.size()-prefix.size());
      if (string_to_int(str,vw)) {
	FutureValue<GeometryData*> reply("Geometry getData reply");
	GeometryComm *msg = scinew GeometryComm
	  (MessageTypes::GeometryGetData, 0, &reply, vw, GEOM_VIEW);
	viewer_->mailbox.send(msg);
	GeometryData *data = reply.receive();
	df = *(data->view);
      }
    } else {
      // position tells first which axis to look down 
      // (with x1 being the positive x axis and x0 being
      // the negative x axis) and then which axis is up
      // represented the same way      
      double distance = (df.eyep()-df.lookat()).length();
      if (position[1] == '0') {
	distance = -distance;
      }    
      if (position[0] == 'x') {
	df.eyep(df.lookat() + Vector(distance, 0.0, 0.0));
      } else if (position[0] == 'y') {
	df.eyep(df.lookat() + Vector(0.0,distance, 0.0));
      } else if (position[0] == 'z') {
	df.eyep(df.lookat() + Vector(0.0,0.0,distance));
      } 
      
      const double up = (position[4] == '1') ? 1.0 : -1.0;
      if (position[3] == 'x') {
	df.up(Vector(up,0.0,0.0));
      } else if (position[3] == 'y') {
	df.up(Vector(0.0,up,0.0));
      } else if (position[3] == 'z') {
	df.up(Vector(0.0,0.0,up));
      } 
    }

    animate_to_view(df, 2.0);
  } else if (args[1] == "edit_light" ){
    if (args.count() != 6) {
      args.error("ViewWindow::switch_light  needs light num, val and vector");
      return;
    }
    // We need to dispatch a message to the remote viewer thread
    // via the viewer_.
    bool on;
    int on_int;
    int lightNo;
    float x = 0,y = 0,z = 0; 
    float r = 0,g = 0,b = 0;

    sscanf(args[2].c_str(), "%d", &lightNo);
    sscanf(args[3].c_str(), "%d", &on_int);  on = (bool)on_int;
    sscanf(args[4].c_str(), "%f%f%f", &x, &y, &z);
    sscanf(args[5].c_str(), "%f%f%f", &r, &g, &b);

    viewer_->
      mailbox.send(scinew ViewerMessage(MessageTypes::ViewWindowEditLight,
					id_, lightNo, on, Vector(x,y,z),
					Color(r,g,b)));
    
  } else if(args[1] == "saveobj") {
    if(args.count() != 6){
      args.error(args[0]+" invalid number of arguments");
      return;
    }
    // We need to dispatch this one to the remote thread We use an ID string
    // instead of a pointer in case this viewwindow gets killed by the time the
    // redraw message gets dispatched.
    ViewerMessage *msg = scinew ViewerMessage
      (MessageTypes::ViewWindowDumpObjects,id_,args[2],args[3],args[4],args[5]);
    viewer_->mailbox.send(msg);
  } else if(args[1] == "listvisuals") {
    args.result(TkOpenGLContext::listvisuals());
  } else if(args[1] == "switchvisual") {
    if(args.count() != 6){
      args.error(args[0]+" needs a window id, visual index, width,and height");
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
    //    renderer_->setvisual(args[2], idx, width, height);
  } else if(args[1] == "setgl") {
    int visualid = 0;
    if (args.count() > 3)
      string_to_int(args[3], visualid);

    int width = 640;
    if (args.count() > 4) 
      string_to_int(args[4], width);

    int height = 480;
    if (args.count() > 5) 
      string_to_int(args[5], height);

    if (renderer_->tk_gl_context_) 
      delete renderer_->tk_gl_context_;

    renderer_->tk_gl_context_ = 
      scinew TkOpenGLContext(args[2], visualid, width, height);
    renderer_->old_tk_gl_context_ = 0;
    renderer_->myname_ = args[2];
    renderer_->start_helper();
  } else if(args[1] == "destroygl") {
    ASSERT(args[2] == renderer_->myname_);
    ASSERT(renderer_->tk_gl_context_);
    delete renderer_->tk_gl_context_;
    renderer_->tk_gl_context_ = 0;
    renderer_->myname_ = "";
  } else if(args[1] == "centerGenAxes") { 
    // have to do this here, as well as in redraw() so the axes can be
    // turned on/off even while spinning with inertia
    viewwindow_objs_draw_[0] = gui_caxes_.get();
  }else
    args.error("Unknown minor command '" + args[1] + "' for ViewWindow");
}

void
ViewWindow::do_mouse(MouseHandler handler, GuiArgs& args)
{
  if(args.count() != 5 && args.count() != 7 &&
     args.count() != 8 && args.count() != 6){
    args.error(args[1]+" needs start/move/end and x y");
    return;
  }
  int action;
  if(args[2] == "start"){
    action=MouseStart;
    mouse_action_ = true;
  } else if(args[2] == "end"){
    action=MouseEnd;
    mouse_action_ = false;
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
  ViewWindowMouseMessage *msg = scinew ViewWindowMouseMessage
    (id_, handler, action, x, y, state, btn, time);
  if (!viewer_->mailbox.trySend(msg))
    cerr << "Mouse event dropped, mailbox full!\n";
}

void
ViewWindow::autoview(const BBox& bbox)
{
  dolly_throttle_=0;
  if (bbox.valid())
  {
    View cv(gui_view_.get());
    // Animate lookat point to center of BBox...
    cv.lookat(bbox.center());
    animate_to_view(cv, 2.0);
    // Move forward/backwards until entire view is in scene.
    // change this a little, make it so that the FOV must be 20 deg.
    double myfov=20.0;

    Vector diag(bbox.diagonal());
    
    double w = diag.length();
    if( w < 0.000001 ){
      BBox bb;
      bb.reset();
      Vector epsilon(0.001, 0.001, 0.001 );
      bb.extend( bbox.min() - epsilon );
      bb.extend( bbox.max() + epsilon );
      w = bb.diagonal().length();
    }
    Vector lookdir(cv.lookat() - cv.eyep()); 
    lookdir.safe_normalize();
    const double scale = 1.0 / (2*Tan(DtoR(myfov/2.0)));
    double length = w*scale;
    cv.fov(myfov);
    cv.eyep(cv.lookat() - lookdir * length);
    animate_to_view(cv, 2.0);
  }
}

void
ViewWindow::redraw()
{
  need_redraw_=0;
  ctx_->reset(); // Get animation variables

  double ct = gui_current_time_.get();
  // Find out whether to draw the axes or not.  Generally, this is handled
  //  in the centerGenAxes case of the tcl_command, but for the first redraw
  //  it's needed here (can't check it in the constructor since the variable
  //  might not exist on the tcl side yet)
  viewwindow_objs_draw_[0] = gui_caxes_.get();

  renderer_->redraw(ct, ct, 1, 0);
}

void
ViewWindow::redraw(double tbeg, double tend, int nframes, double framerate)
{
  need_redraw_=0;
  ctx_->reset();   // Get animation variables
  renderer_->redraw(tbeg, tend, nframes, framerate);
}

void
ViewWindow::update_mode_string(GeomHandle pick_obj)
{
  string ms="pick: ";
  GeomViewerItem* si=dynamic_cast<GeomViewerItem*>(pick_obj.get_rep());
  if(!si){
    ms += "GeomObj";
  } else {
    ms+=si->name_;
  }
  if(pick_n_ != 0x12345678)
    ms+=", index="+to_string(pick_n_);
  update_mode_string(ms);
}

void
ViewWindow::update_mode_string(const string& msg)
{
  gui_->eval(id_ + " updateMode {"+msg+"}");
}

ViewWindowMouseMessage::ViewWindowMouseMessage(const string& rid, 
					       MouseHandler handler,
					       int action, int x, int y, 
					       int state, int btn, int time)
  : MessageBase(MessageTypes::ViewWindowMouse), rid(rid), handler(handler),
    action(action), x(x), y(y), state(state), btn(btn), time(time)
{
}

ViewWindowMouseMessage::~ViewWindowMouseMessage()
{
}

void
ViewWindow::animate_to_view(const View& v, double /*time*/)
{
  gui_view_.set(v);
  viewer_->mailbox.send(scinew ViewerMessage(id_));
}

void
ViewWindow::do_for_visible(OpenGL* r, ViewWindowVisPMF pmf)
{
  // Do internal objects first...
  unsigned int i;
  for (i = 0; i < viewwindow_objs_.size(); i++){
    if (viewwindow_objs_draw_[i] == 1) {
      (r->*pmf)(viewer_, this, viewwindow_objs_[i].get_rep());
    }
  }
  
  for (int pass=0; pass < 4; pass++)
  {
    GeomIndexedGroup::IterIntGeomObj iter = viewer_->ports_.getIter();
    for ( ; iter.first != iter.second; iter.first++) {
      GeomIndexedGroup::IterIntGeomObj serIter = 
	((GeomViewerPort*)((*iter.first).second.get_rep()))->getIter();
      for ( ; serIter.first != serIter.second; serIter.first++) {
	GeomViewerItem *si =
	  (GeomViewerItem*)((*serIter.first).second.get_rep());
	// Look up the name to see if it should be drawn...
	map<string,GuiInt*>::iterator gui_iter = visible_.find(si->name_);
	if (gui_iter != visible_.end()) { // if found
	  if ((*gui_iter).second->get())
	  {
	    const bool transparent =
	      strstr(si->name_.c_str(), "TransParent") ||
	      strstr(si->name_.c_str(), "Transparent");
	    const bool culledtext = strstr(si->name_.c_str(), "Culled Text");
	    const bool sticky = strstr(si->name_.c_str(), "Sticky");
	    if ((pass == 0 && !transparent && !culledtext && !sticky) ||
		(pass == 1 && transparent && !culledtext && !sticky) ||
		(pass == 2 && culledtext && !sticky) ||
		(pass == 3 && sticky))
	    {
	      if(si->crowd_lock_)
		si->crowd_lock_->readLock();
	      (r->*pmf)(viewer_, this, si);
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

void
ViewWindow::set_current_time(double time)
{
  gui_current_time_.set((int)time);
}

void
ViewWindow::dump_objects(const string& filename, const string& format)
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
    viewer_->geomlock_.readLock();
    GeomScene scene(gui_bgcolor_.get(), gui_view_.get(), 
		    &viewer_->lighting_, &viewer_->ports_);
    Pio(*stream, scene);
    if(stream->error()){
      cerr << "Error writing geom file: " << filename << "\n";
    } else {
      cerr << "Done writing geom file: " << filename << "\n";
    }
    delete stream;
    viewer_->geomlock_.readUnlock();
  } else {
    cerr << "WARNING: format " << format << " not supported!\n";
  }
}

void
ViewWindow::getData(int datamask, FutureValue<GeometryData*>* result)
{
  if(renderer_){
    renderer_->getData(datamask, result);
  } else {
    result->send(0);
  }
}

void
ViewWindow::setView(View newView) {
  gui_view_.set(newView);
  viewer_->mailbox.send(scinew ViewerMessage(id_)); // Redraw
}

GeomHandle
ViewWindow::createGenAxes() {
  const Color black(0,0,0), grey(.5,.5,.5);
  MaterialHandle dk_red =   scinew Material(black, Color(.2,0,0), grey, 20);
  MaterialHandle dk_green = scinew Material(black, Color(0,.2,0), grey, 20);
  MaterialHandle dk_blue =  scinew Material(black, Color(0,0,.2), grey, 20);
  MaterialHandle lt_red =   scinew Material(black, Color(.8,0,0), grey, 20);
  MaterialHandle lt_green = scinew Material(black, Color(0,.8,0), grey, 20);
  MaterialHandle lt_blue =  scinew Material(black, Color(0,0,.8), grey, 20);

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

} // End namespace SCIRun
