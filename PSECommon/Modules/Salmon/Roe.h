
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

#include <Containers/Array1.h>
#include <Containers/HashTable.h>
#include <Comm/MessageBase.h>
#include <Geom/Color.h>
#include <Geom/TCLGeom.h>
#include <Geom/TCLView.h>
#include <Geom/View.h>
#include <Geometry/BBox.h>
#include <Geometry/Transform.h>
#include <TclInterface/TCL.h>
#include <TclInterface/TCLvar.h>
#include <Modules/Salmon/BallAux.h>

namespace SCICore {
  namespace GeomSpace {
    class GeomObj;
    class GeomPick;
    class GeomSphere;
    struct DrawInfoOpenGL;
    class Light;
  }
  namespace Geometry {
    class Vector;
    class Transform;
  }
  namespace Multitask {
    template<class T> class AsyncReply;
  }
}

namespace PSECommon {
  namespace CommonDatatypes {
    struct GeometryData;
  }
}

namespace PSECommon {
namespace Modules {

using PSECommon::Comm::MessageBase;
using PSECommon::CommonDatatypes::GeometryData;

using SCICore::GeomSpace::TCLColor;
using SCICore::GeomSpace::TCLView;
using SCICore::GeomSpace::View;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::GeomPick;
using SCICore::GeomSpace::GeomSphere;
using SCICore::GeomSpace::DrawInfoOpenGL;
using SCICore::Multitask::AsyncReply;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::BBox;
using SCICore::Geometry::Transform;
using SCICore::Containers::Array1;
using SCICore::Containers::HashTable;

using namespace SCICore::TclInterface;

class DBContext;
class Renderer;
class Salmon;
//class SceneItem;

class GeomSalmonItem;
class BallData;

class TexStruct1D;
class TexStruct2D;
class TexStruct3D;
class SegBin;       // bins for sorted line segments...

struct ObjTag {
    TCLvarint* visible;
    int tagid;
};

class Roe;
typedef void (Roe::*MouseHandler)(int, int x, int y, int state, int btn, int time);
typedef void (Renderer::*RoeVisPMF)(Salmon*, Roe*, GeomObj*);

class Roe : public TCL {
protected:
    friend class Salmon;
    Salmon* manager;
    HashTable<clString, Renderer*> renderers;

    void do_mouse(MouseHandler, TCLArgs&);

    BBox bb;

    int last_x, last_y;
    double total_x, total_y, total_z;
    Point rot_point;
    int rot_point_valid;
    GeomPick* pick_pick;
    GeomObj* pick_obj;
    int pick_n;

    void update_mode_string(const char*);
    char* modebuf;
    char* modecommand;

    int maxtag;

    Point orig_eye;
    Vector frame_up;
    Vector frame_right;
    Vector frame_front;

    Point mousep;
    GeomSphere* mouse_obj;
    Array1<GeomObj*> roe_objs;

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
    BallData *ball;  // this is the ball for arc ball stuff

    double angular_v; // angular velocity for inertia
    View rot_view;    // pre-rotation view
    Transform prev_trans;
    double eye_dist;
    double total_scale;
    int prev_time[3]; // history for quaternions and time
    HVect prev_quat[3];

    int doingMovie;
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

    void mouse_translate(int, int, int, int, int, int);
    void mouse_scale(int, int, int, int, int, int);
    void mouse_rotate(int, int, int, int, int, int);
    void mouse_pick(int, int, int, int, int, int);

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
    HashTable<clString, ObjTag*> visible;

    // Which of the lights are on?
    HashTable<clString, int> light_on;

    // The Camera
    TCLView view;
    View homeview;

    // Background Color
    TCLColor bgcolor;

    // Shading parameters, etc.
    TCLstring shading;

    // Stereo
    TCLint do_stereo;

    TCLint drawimg;

    TCLstring saveprefix;

    // Object processing utility routines
    void do_for_visible(Renderer*, RoeVisPMF);

    void set_current_time(double time);

    void dump_objects(const clString&, const clString& format);

    void getData(int datamask, AsyncReply<GeometryData*>* result);
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:52  mcq
// Initial commit
//
// Revision 1.4  1999/05/13 18:25:12  dav
// Removed TCLView from Roe.h
//
// Revision 1.3  1999/05/06 20:17:10  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//

#endif
