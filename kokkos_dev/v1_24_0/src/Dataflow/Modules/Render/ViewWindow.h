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
 *  ViewWindow.h: The Geometry Viewer Window
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_ViewWindow_h
#define SCI_project_module_ViewWindow_h

#include <Dataflow/Modules/Render/BallAux.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GuiGeom.h>
#include <Core/Geom/GuiView.h>
#include <Core/Geom/View.h>
#include <map>

// define 'the_time()' function for UniCam
#ifdef WIN32
#  include <windows.h>
#  include <winbase.h>
   inline double the_time() {
      return double(GetTickCount())/1000.0;
   }
#else
#  include <sys/time.h>
   inline double the_time() {
      struct timeval ts; struct timezone tz;
      gettimeofday(&ts, &tz);
      return (double)(ts.tv_sec + (double)ts.tv_usec/1e6);
   }
#endif

namespace SCIRun {

using namespace std;

struct DrawInfoOpenGL;
struct GeometryData;

class GeomObj;
class GeomSphere;
class Light;
class Vector;
class Transform;
class OpenGL;
class Viewer;
class GeomViewerItem;
class BallData;
class OpenGL;
class ViewWindow;

typedef void (ViewWindow::*MouseHandler)(int, int x, int y, 
				  int state, int btn, int time);
typedef void (OpenGL::*ViewWindowVisPMF)(Viewer*, ViewWindow*, GeomHandle);

class ViewWindow : public GuiCallback {
  friend class Viewer;
public:
  ViewWindow(Viewer *s, GuiInterface* gui, GuiContext* ctx);
  ~ViewWindow();

  void			itemAdded(GeomViewerItem*);
  void			itemDeleted(GeomViewerItem*);
  void			itemRenamed(GeomViewerItem*, string newname);
  void			redraw_if_needed();
  // Mouse Callbacks
  void			mouse_dolly(int, int, int, int, int, int);
  void			mouse_translate(int, int, int, int, int, int);
  void			mouse_scale(int, int, int, int, int, int);
  void			mouse_unicam(int, int, int, int, int, int);
  void			mouse_rotate(int, int, int, int, int, int);
  void			mouse_rotate_eyep(int, int, int, int, int, int);
  void			mouse_pick(int, int, int, int, int, int);
  void			tcl_command(GuiArgs&, void*);
  void			get_bounds(BBox&);
  void			get_bounds_all(BBox&);
  void			autoview(const BBox&);
  // sets up the state (OGL) for a tool/viewwindow
  void			setState(DrawInfoOpenGL*, const string&);
  void			setDI(DrawInfoOpenGL*,string); // setup DI for drawinfo
  void			setClip(DrawInfoOpenGL*); // setup OGL clipping planes
  void			setMouse(DrawInfoOpenGL*); // setup mouse state
  void			do_for_visible(OpenGL*, ViewWindowVisPMF);
  void			set_current_time(double time);
  void			dump_objects(const string&, const string& format);
  void			setView(View view);
  void			getData(int mask, FutureValue<GeometryData*>* result);
  GeomHandle		createGenAxes();   

  void                  setMovie( int state );
  void                  setMovieFrame( int movieframe );
  void                  setMessage( string message );

  // UNICAM START
  void			unicam_choose(int X, int Y);
  void			unicam_rot(int X, int Y);
  void			unicam_zoom(int X, int Y);
  void			unicam_pan(int X, int Y);
  void			ShowFocusSphere();
  void			HideFocusSphere();
  void			MyTranslateCamera(Vector offset);
  void			MyRotateCamera(Point center,Vector axis, double angle);
  Vector		CameraToWorld(Vector v);
  void			NormalizeMouseXY( int X, int Y, float *NX, float *NY);
  void			UnNormalizeMouseXY(float NX, float NY, int *X, int *Y);
  float			WindowAspect();
  // for 'film_dir' and 'film_pt', x & y should be in the range [-1, 1].
  Vector		film_dir(double x, double y); 
  Point			film_pt(double x, double y, double z=1.0);
  // UNICAM END

  // Public Variables, (public for OpenGL class access)
  string		id_;

  BallData *		ball_;		// this is the ball for arc ball stuff
  double		angular_v_;	// angular velocity for inertia
  View			rot_view_;	// pre-rotation view
  Transform		prev_trans_;
  double		eye_dist_;
  GuiView		gui_view_;
  GuiInt		gui_sr_;
  GuiInt		gui_do_stereo_;		// Stereo
  GuiInt		gui_ortho_view_;
  GuiInt		gui_track_view_window_0_;
  GuiInt		gui_raxes_;
  GuiDouble		gui_ambient_scale_;	// Scene material scales
  GuiDouble		gui_diffuse_scale_;
  GuiDouble		gui_specular_scale_;
  GuiDouble		gui_emission_scale_;
  GuiDouble		gui_shininess_scale_;
  GuiDouble		gui_polygon_offset_factor_;
  GuiDouble		gui_polygon_offset_units_;
  GuiDouble		gui_point_size_;
  GuiDouble		gui_line_width_;
  GuiDouble		gui_sbase_;
  GuiColor		gui_bgcolor_;		// Background Color
  GuiInt                gui_fogusebg_;
  GuiColor              gui_fogcolor_;
  GuiDouble             gui_fog_start_;
  GuiDouble             gui_fog_end_;
  GuiInt                gui_fog_visibleonly_;
  GuiInt		gui_total_frames_;

  GuiDouble		gui_inertia_mag_;
  GuiDouble		gui_inertia_x_;
  GuiDouble		gui_inertia_y_;
  GuiInt		gui_inertia_recalculate_;
  GuiInt		gui_inertia_mode_;
private:
  ViewWindow(const ViewWindow&); // Should not be called
  void			do_mouse(MouseHandler, GuiArgs&);
  void			update_mode_string(const string&);
  void			update_mode_string(GeomHandle);
  void			animate_to_view(const View& v, double time);
  void			redraw();
  void			redraw(double tbeg, double tend, 
			       int nframes, double framerate);

  // Private Member variables
  Viewer*		viewer_;
  OpenGL*		renderer_;
  GuiInterface*		gui_;
  GuiContext*		ctx_;
  string                tclID_;
  map<string,GuiInt*>	visible_;   // Which of the objects do we draw?
  map<string,int>	obj_tag_;
  bool			need_redraw_;
  int			pick_n_;
  int			maxtag_;
  int			last_x_;
  int			last_y_;
  int			pick_x_;
  int			pick_y_;
  int			last_time_;
  bool			mouse_action_;
  double		total_x_;
  double		total_y_;
  double		total_z_;
  double		total_scale_;
  View			homeview_;
  int			rotate_valid_p_;
  GeomPickHandle	pick_pick_;
  GeomHandle		pick_obj_;
  vector<GeomHandle>	viewwindow_objs_;
  vector<bool>		viewwindow_objs_draw_;   
  // Variables for mouse_dolly method (dollying into/outof a view)
  double		dolly_total_;
  Vector		dolly_vector_;
  double		dolly_throttle_;
  double		dolly_throttle_scale_;
  // history for quaternions and time
  int			prev_time_[3]; 
  HVect			prev_quat_[3];
  // UNICAM variables
  enum {UNICAM_CHOOSE = 0, UNICAM_ROT, UNICAM_PAN, UNICAM_ZOOM};
  int			unicam_state_;
  int			down_x_;
  int			down_y_;
  Point			down_pt_;
  double		dtime_;
  double		uni_dist_;
  float			last_pos_[3];
  float			start_pix_[2];
  float			last_pix_[2];
  GeomSphere *		focus_sphere_;
  // TCL GUI Variables
  GuiInt		gui_current_time_;
  GuiString		gui_currentvisual_;
  GuiInt		gui_caxes_;
  GuiString		gui_pos_;
};

class ViewWindowMouseMessage : public MessageBase {
public:
  string rid;
  MouseHandler handler;
  int action;
  int x, y;
  int state;
  int btn;
  int time;
  
  ViewWindowMouseMessage(const string& rid, MouseHandler handler,
                         int action, int x, int y, int state, int btn,
                         int time);
  virtual ~ViewWindowMouseMessage();
};

} // End namespace SCIRun



#endif
