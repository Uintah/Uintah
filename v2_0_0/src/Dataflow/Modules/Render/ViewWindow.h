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

#include <Core/Containers/Array1.h>
#include <Core/Thread/FutureValue.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GuiGeom.h>
#include <Core/Geom/GuiView.h>
#include <Core/Geom/View.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Render/BallAux.h>

// --  BAWGL -- 
#include <Dataflow/Modules/Render/SCIBaWGL.h>
// --  BAWGL -- 

// CollabVis code begin
#ifdef HAVE_COLLAB_VIS
#include <Packages/CollabVis/Standalone/Server/ViewServer.h>
#undef ASIP_SHORT_NAMES
#endif
// CollabVis code end

#include <map>

// define 'the_time()' function for UniCam
#ifdef WIN32
#include <windows.h>
#include <winbase.h>



inline double the_time() {
    return double(GetTickCount())/1000.0;
}
#else
#include <sys/time.h>

inline double the_time() {
    struct timeval ts; struct timezone tz;
    gettimeofday(&ts, &tz);
    return (double)(ts.tv_sec + (double)ts.tv_usec/1e6);
}
#endif

template <class Type>
inline Type clamp(const Type a,const Type b,const Type c) { return a > b ? (a < 
c ? a : c) : b ; }
inline int  Sign (double a)             { return a > 0 ? 1 : a < 0 ? -1 : 0; }

namespace SCIRun {
  using namespace std;
class GeomObj;
class GeomSphere;
struct DrawInfoOpenGL;
class Light;
class Vector;
class Transform;
struct GeometryData;
class OpenGL;
class DBContext;
class Viewer;
class SCIBaWGL;

class GeomViewerItem;
class BallData;
class OpenGL;
class TexStruct1D;
class TexStruct2D;
class TexStruct3D;
class SegBin;			// bins for sorted line segments...

struct ObjTag {
  GuiInt* visible;
  int tagid;
};

class ViewWindow;
typedef void (ViewWindow::*MouseHandler)(int, int x, int y, 
				  int state, int btn, int time);
typedef void (OpenGL::*ViewWindowVisPMF)(Viewer*, ViewWindow*, GeomHandle);

class ViewWindow : public GuiCallback {
  GuiInterface* gui;
  GuiContext* ctx;
  ViewWindow(const ViewWindow&);
  // --  BAWGL -- 
public:
  Viewer* manager;
  // --  BAWGL -- 
  
public:
  typedef map<string, ObjTag*>	        MapStringObjTag;
  GuiString pos;  
  GuiInt caxes;
  GuiInt raxes;
  GuiInt iaxes;

  // CollabVis code begin
  GuiInt HaveCollabVis_;
  // CollabVis code end
  
protected:
  friend class Viewer;
  
  void do_mouse(MouseHandler, GuiArgs&);
  
  BBox bb;
  
  int last_x, last_y;
  int pick_x, pick_y;
  double total_x, total_y, total_z;
  Point rot_point;
  int rot_point_valid;
  GeomPickHandle pick_pick;
  GeomHandle pick_obj;
  int pick_n;

  void update_mode_string(const string&);
  void update_mode_string(GeomHandle);

  int maxtag;

  // --  BAWGL -- 
  SCIBaWGL* bawgl;
  int bawgl_error;
  // --  BAWGL -- 

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  ViewServer * server;
#endif
  // CollabVis code end
  
  Point orig_eye;
  Vector frame_up;
  Vector frame_right;
  Vector frame_front;

  vector<GeomHandle> viewwindow_objs;
  vector<bool> viewwindow_objs_draw;   

  void animate_to_view(const View& v, double time);
  void redraw();
  void redraw(double tbeg, double tend, int nframes, double framerate);

  Array1< TexStruct1D* >   tmap_1d;
  Array1< unsigned int >   tmap_tex_objs_1d;

  Array1< TexStruct2D* >   tmap_2d;
  Array1< unsigned int >   tmap_tex_objs_2d;

  Array1< TexStruct3D* >   tmap_3d;
  Array1< unsigned int >   tmap_tex_objs_3d;  // no more than 1!!!

  SegBin*                  line_segs;   // for lit streamlines/hedgehogs/etc
  
  int last_time;
  
public:
  int inertia_mode;
  BallData *ball;		// this is the ball for arc ball stuff
  
  double angular_v;		// angular velocity for inertia
  View rot_view;		// pre-rotation view
  Transform prev_trans;
  double eye_dist;
  double total_scale;
  int prev_time[3];		// history for quaternions and time
  HVect prev_quat[3];
  
  bool doingMovie;
  bool makeMPEG;
  int curFrame;
  string curName;
  
  // variables for dollying into / out-of the scene
  double dolly_total;
  Vector dolly_vector;
  double dolly_throttle;
  double dolly_throttle_scale;

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  
  RenderGroupInfo *groupInfo;
  Mutex groupInfoLock;
  bool handlingOneTimeRequest;
  void sendImageToServer( char * image, int xres, int yres );
  inline bool serverNeedsImage() {
    bool returnval;
    groupInfoLock.lock();
    returnval = ( groupInfo != NULL );
    groupInfoLock.unlock();
    return returnval;
  }
  
  Array1<GeomObj*> getGeometry();
  
  void  getViewState( ViewWindowState &state );
  void  setViewState( const ViewWindowState &state );
  CrowdMonitor viewStateLock;

  inline GuiInterface * getGui() { return gui; }
  
  
#endif
  // CollabVis code end
  
  void LoadTexture1D(TexStruct1D*);
  void LoadTexture2D(TexStruct2D*);
  void LoadTexture3D(TexStruct3D*);
  void LoadColorTable(TexStruct1D*);

  int tex_disp_list;

  bool show_rotation_axis;

  OpenGL* current_renderer;
  string id;
  bool need_redraw;

  SCIBaWGL* get_bawgl(void) { return(bawgl); }
    
  ViewWindow(Viewer *s, GuiInterface* gui, GuiContext* ctx);
  ~ViewWindow();

  string set_id(const string& new_id);

  void itemAdded(GeomViewerItem*);
  void itemDeleted(GeomViewerItem*);
  void itemRenamed(GeomViewerItem*, string newname);
  void rotate(double angle, Vector v, Point p);
  void rotate_obj(double angle, const Vector& v, const Point& p);
  void translate(Vector v);
  void scale(Vector v, Point p);
  void addChild(ViewWindow *r);
  void deleteChild(ViewWindow *r);
  void SetParent(ViewWindow *r);
  void SetTop();
  void redraw_if_needed();
  void force_redraw();

  // -- BAWGL --
  void bawgl_pick(int action, GLint iv[3], GLfloat fv[4]);
  // -- BAWGL -- 

  void mouse_dolly(int, int, int, int, int, int);
  void mouse_translate(int, int, int, int, int, int);
  void mouse_scale(int, int, int, int, int, int);
  void mouse_unicam(int, int, int, int, int, int);
  void mouse_rotate(int, int, int, int, int, int);
  void mouse_rotate_eyep(int, int, int, int, int, int);
  void mouse_pick(int, int, int, int, int, int);

#if 1
  // ---- start UniCam interactor methods & member variables

  // XXX - note: dependencies did not work in my (asf's) hierarchy at
  // brown.  When i added variables here, it caused the renderer to
  // crash.  After debugging, i'm 99% sure the reason is files that
  // depend on ViewWindow.h are not updated when it is modified.  Is this a
  // bug in the Makefiles?

  double _dtime;
  double _dist;
  float _last_pos[3], _start_pix[2], _last_pix[2];

  Point _down_pt;
  int   _down_x, _down_y;
  Point _center;  // center of camera rotation

  GeomSphere     *focus_sphere;

  enum {UNICAM_CHOOSE = 0, UNICAM_ROT, UNICAM_PAN, UNICAM_ZOOM};
  int  unicam_state;

  void unicam_choose(int X, int Y);
  void unicam_rot(int X, int Y);
  void unicam_zoom(int X, int Y);
  void unicam_pan(int X, int Y);

  void   ShowFocusSphere();
  void   HideFocusSphere();
  Point  GetPointUnderCursor( int x, int y );

  void   MyTranslateCamera(Vector offset);
  void   MyRotateCamera   (Point  center,
                           Vector axis,
                           double angle);
  Vector CameraToWorld(Vector v);
  void   NormalizeMouseXY( int X, int Y, float *NX, float *NY);
  float  WindowAspect();

  // for 'film_dir' and 'film_pt', x & y should be in the range [-1, 1].
  Vector film_dir   (double x, double y);
  Point  film_pt    (double x, double y, double z=1.0);

  // ---- end UniCam interactor methods & member variables
#endif

  void tcl_command(GuiArgs&, void*);
  void get_bounds(BBox&);

  void autoview(const BBox&);

				// sets up the state (OGL) for a tool/viewwindow
  void setState(DrawInfoOpenGL*, const string&);
				// sets up DI for this drawinfo
  void setDI(DrawInfoOpenGL*,string);
				// sets up OGL clipping planes...
  void setClip(DrawInfoOpenGL*); 

				// Which of the objects do we draw?
  MapStringObjTag visible;

				// Which of the lights are on?
  //map<string, int> light_on;
    
				// The Camera
  GuiView view;
  View homeview;

  GuiString lightColors;        // Colors for lights
  GuiString lightVectors;       // Viewing hemisphere, light direction vectors
				// Background Color
  GuiColor bgcolor;

				// Shading parameters, etc.
  GuiString shading;

				// Stereo
  GuiInt do_stereo;

  GuiDouble ambient_scale;      // Scene material scales
  GuiDouble diffuse_scale;
  GuiDouble specular_scale;
  GuiDouble emission_scale;
  GuiDouble shininess_scale;

  GuiDouble polygon_offset_factor;
  GuiDouble polygon_offset_units;
  GuiDouble point_size;
  GuiDouble line_width;

  GuiDouble sbase;
  GuiInt sr;

  // --  BAWGL -- 
  GuiInt do_bawgl;
  // --  BAWGL -- 

  GuiInt drawimg;

  GuiString saveprefix;

  GuiInt file_resx;
  GuiInt file_resy;
  GuiInt file_aspect;
  GuiDouble file_aspect_ratio;
				// Object processing utility routines
  void do_for_visible(OpenGL*, ViewWindowVisPMF);
  
  void set_current_time(double time);
  
  void dump_objects(const string&, const string& format);
  
  void getData(int datamask, FutureValue<GeometryData*>* result);
  void setView(View view);
  GeomHandle createGenAxes();   
  void emit_vars(std::ostream& out, const std::string& midx);

  bool ortho_view() { return gui_ortho_view_.get(); }

private:
  // Cache off these so that they are saved in the nets.
  GuiInt gui_global_light_;
  GuiInt gui_global_fog_;
  GuiInt gui_global_debug_;
  GuiInt gui_global_clip_;
  GuiInt gui_global_cull_;
  GuiInt gui_global_dl_;
  GuiString gui_global_type_;
  GuiInt gui_ortho_view_;
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
