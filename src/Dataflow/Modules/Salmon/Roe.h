/*
 *  Roe.h: The Geometry Viewer Window
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Roe_h
#define SCI_project_module_Roe_h

#include <Core/Containers/Array1.h>
#include <Core/Thread/FutureValue.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TCLGeom.h>
#include <Core/Geom/TCLView.h>
#include <Core/Geom/View.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/TclInterface/TCL.h>
#include <Core/TclInterface/TCLvar.h>
#include <Dataflow/Modules/Salmon/BallAux.h>

// --  BAWGL -- 
#include <Dataflow/Modules/Salmon/SCIBaWGL.h>
// --  BAWGL -- 

#include <map.h>

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
    return (double)(ts.tv_sec + ts.tv_usec/1e6);
}
#endif

template <class Type>
inline Type clamp(const Type a,const Type b,const Type c) { return a > b ? (a < 
c ? a : c) : b ; }
inline int  Sign (double a)             { return a > 0 ? 1 : a < 0 ? -1 : 0; }

namespace SCIRun {
class GeomObj;
class GeomPick;
class GeomSphere;
struct DrawInfoOpenGL;
class Light;
class Vector;
class Transform;
struct GeometryData;

class DBContext;
class Renderer;
class Salmon;
class SCIBaWGL;

class GeomSalmonItem;
class BallData;

class TexStruct1D;
class TexStruct2D;
class TexStruct3D;
class SegBin;			// bins for sorted line segments...

struct ObjTag {
  TCLvarint* visible;
  int tagid;
};

class Roe;
typedef void (Roe::*MouseHandler)(int, int x, int y, 
				  int state, int btn, int time);
typedef void (Renderer::*RoeVisPMF)(Salmon*, Roe*, GeomObj*);

class Roe : public TCL {
  
  // --  BAWGL -- 
public:
  Salmon* manager;
  // --  BAWGL -- 
  
public:
  typedef map<clString, Renderer*>	MapClStringRenderer;
  typedef map<clString, ObjTag*>	MapClStringObjTag;
  TCLstring pos;  
  TCLint caxes;
  TCLint iaxes;  
protected:
  friend class Salmon;
  
  MapClStringRenderer renderers;
  
  void do_mouse(MouseHandler, TCLArgs&);
  
  BBox bb;
  
  int last_x, last_y;
  double total_x, total_y, total_z;
  Point rot_point;
  int rot_point_valid;
  GeomPick* pick_pick;
  GeomObj* pick_obj;
  int pick_n;

  void update_mode_string(const clString&);
  void update_mode_string(GeomObj*);

  int maxtag;

  // --  BAWGL -- 
  SCIBaWGL* bawgl;
  int bawgl_error;
  // --  BAWGL -- 

  Point orig_eye;
  Vector frame_up;
  Vector frame_right;
  Vector frame_front;

  Point mousep;
  GeomSphere* mouse_obj;
  Array1<GeomObj*> roe_objs;
  Array1<char> roe_objs_draw;   

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
  clString curName;
  
  void LoadTexture1D(TexStruct1D*);
  void LoadTexture2D(TexStruct2D*);
  void LoadTexture3D(TexStruct3D*);
  void LoadColorTable(TexStruct1D*);

  int tex_disp_list;


  Renderer* current_renderer;
  Renderer* get_renderer(const clString&);
  clString id;
  int need_redraw;

  SCIBaWGL* get_bawgl(void) { return(bawgl); }
    
  Roe(Salmon *s, const clString& id);
  Roe(const Roe&);
  ~Roe();

  clString set_id(const clString& new_id);

  void itemAdded(GeomSalmonItem*);
  void itemDeleted(GeomSalmonItem*);
  void rotate(double angle, Vector v, Point p);
  void rotate_obj(double angle, const Vector& v, const Point& p);
  void translate(Vector v);
  void scale(Vector v, Point p);
  void addChild(Roe *r);
  void deleteChild(Roe *r);
  void SetParent(Roe *r);
  void SetTop();
  void redraw_if_needed();
  void force_redraw();

  // -- BAWGL --
  void bawgl_pick(int action, GLint iv[3], GLfloat fv[4]);
  // -- BAWGL -- 

  void mouse_translate(int, int, int, int, int, int);
  void mouse_scale(int, int, int, int, int, int);
  void mouse_rotate(int, int, int, int, int, int);
  void mouse_rotate_eyep(int, int, int, int, int, int);
  void mouse_pick(int, int, int, int, int, int);

#if 1
  // ---- start UniCam interactor methods & member variables

  // XXX - note: dependencies did not work in my (asf's) hierarchy at
  // brown.  When i added variables here, it caused the renderer to
  // crash.  After debugging, i'm 99% sure the reason is files that
  // depend on Roe.h are not updated when it is modified.  Is this a
  // bug in the Makefiles?

  double _dtime;
  double _dist;
  float _last_pos[3], _start_pix[2], _last_pix[2];

  Point _down_pt;
  int   _down_x, _down_y;
  Point _center;  // center of camera rotation

  GeomSphere     *focus_sphere;
  int             is_dot;  // is the focus_sphere being displayed?

  enum {UNICAM_CHOOSE = 0, UNICAM_ROT, UNICAM_PAN, UNICAM_ZOOM};
  int  unicam_state;

  void choose(int X, int Y);
  void rot   (int X, int Y);
  void zoom  (int X, int Y);
  void pan   (int X, int Y);

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

  void tcl_command(TCLArgs&, void*);
  void get_bounds(BBox&);

  void autoview(const BBox&);

				// sets up the state (OGL) for a tool/roe
  void setState(DrawInfoOpenGL*,clString);
				// sets up DI for this drawinfo
  void setDI(DrawInfoOpenGL*,clString);
				// sets up OGL clipping planes...
  void setClip(DrawInfoOpenGL*); 

				// Which of the objects do we draw?
  MapClStringObjTag visible;

				// Which of the lights are on?
  //map<clString, int> light_on;
    
				// The Camera
  TCLView view;
  View homeview;

				// Background Color
  TCLColor bgcolor;

				// Shading parameters, etc.
  TCLstring shading;

				// Stereo
  TCLint do_stereo;

  // --  BAWGL -- 
  TCLint do_bawgl;
  // --  BAWGL -- 

  TCLint drawimg;

  TCLstring saveprefix;
  
				// Object processing utility routines
  void do_for_visible(Renderer*, RoeVisPMF);
  
  void set_current_time(double time);
  
  void dump_objects(const clString&, const clString& format);
  
  void getData(int datamask, FutureValue<GeometryData*>* result);
  void setView(View view);
  GeomGroup* createGenAxes();   
};

class RoeMouseMessage : public MessageBase {
public:
  clString rid;
  MouseHandler handler;
  int action;
  int x, y;
  int state;
  int btn;
  int time;
  
  
  RoeMouseMessage(const clString& rid, MouseHandler handler,
		  int action, int x, int y, int state, int btn,
		  int time);
  virtual ~RoeMouseMessage();
};

} // End namespace SCIRun



#endif
