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

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/HashTable.h>
#include <PSECore/Comm/MessageBase.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/TCLGeom.h>
#include <SCICore/Geom/TCLView.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <PSECommon/Modules/Salmon/BallAux.h>

// >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
#include <PSECommon/Modules/Salmon/SCIBaWGL.h>
// <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<

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
  namespace Thread {
      template<class T> class FutureValue;
  }
}

namespace PSECore {
  namespace Datatypes {
    struct GeometryData;
  }
}

namespace PSECommon {
namespace Modules {

using PSECore::Comm::MessageBase;
using PSECore::Datatypes::GeometryData;

using SCICore::GeomSpace::TCLColor;
using SCICore::GeomSpace::TCLView;
using SCICore::GeomSpace::View;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::GeomPick;
using SCICore::GeomSpace::GeomSphere;
using SCICore::GeomSpace::DrawInfoOpenGL;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::BBox;
using SCICore::Geometry::Transform;
using SCICore::Containers::Array1;
using SCICore::Containers::HashTable;
using SCICore::Thread::FutureValue;

using namespace SCICore::TclInterface;

class DBContext;
class Renderer;
class Salmon;
class SCIBaWGL;

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

// >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
public:
    Salmon* manager;
// <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<

protected:
    friend class Salmon;
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

    void update_mode_string(const clString&);
    void update_mode_string(GeomObj*);

    int maxtag;

// >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
    SCIBaWGL* bawgl;
    int bawgl_error;
// <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<

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

    //>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    void bawgl_pick(int action, GLint iv[3], GLfloat fv[4]);
    //<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

// >>>>>>>>>>>>>>>>>>>> BAWGL >>>>>>>>>>>>>>>>>>>>
    TCLint do_bawgl;
// <<<<<<<<<<<<<<<<<<<< BAWGL <<<<<<<<<<<<<<<<<<<<

    TCLint drawimg;

    TCLstring saveprefix;

    // Object processing utility routines
    void do_for_visible(Renderer*, RoeVisPMF);

    void set_current_time(double time);

    void dump_objects(const clString&, const clString& format);

    void getData(int datamask, FutureValue<GeometryData*>* result);
    void setView(View view);
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
// Revision 1.11  1999/12/03 00:28:59  dmw
// added setView message for Salmon/Roe
//
// Revision 1.10  1999/11/19 19:46:24  kuzimmer
// Took out #ifdef __sgi in Roe.h.
//
// Revision 1.9  1999/11/16 00:47:26  yarden
// put "#ifdef __sgi" around code for BAWGL
//
// Revision 1.8  1999/10/21 22:39:06  ikits
// Put bench.config into PSE/src (where the executable gets invoked from). Fixed bug in the bawgl code and added preliminary navigation and picking.
//
// Revision 1.7  1999/10/16 20:51:00  jmk
// forgive me if I break something -- this fixes picking and sets up sci
// bench - go to /home/sci/u2/VR/PSE for the latest sci bench technology
// gota getup to get down.
//
// Revision 1.6  1999/10/07 02:06:57  sparker
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
// Revision 1.3  1999/08/25 03:47:57  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:37:39  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
