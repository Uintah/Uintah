
/*
 *  OpenGL.cc: Render geometry using opengl
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */
//milan was here

#ifndef SCI_project_module_OpenGL_h
#define SCI_project_module_OpenGL_h

#include <tcl.h>
#include <tk.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>

#include <map.h>

#include "image.h"
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Util/Timer.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geom/Light.h>
#include <SCICore/Geom/Lighting.h>
#include <SCICore/Geom/RenderMode.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/Datatypes/Image.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECommon/Modules/Salmon/Ball.h>
#include <PSECommon/Modules/Salmon/MpegEncoder.h>
#include <PSECommon/Modules/Salmon/Renderer.h>
#include <PSECommon/Modules/Salmon/Roe.h>
#include <PSECommon/Modules/Salmon/Salmon.h>
#include <SCICore/Thread/FutureValue.h>
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Thread.h>

#include <SCICore/Geom/GeomCone.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomText.h>

#ifdef __sgi
#include <X11/extensions/SGIStereo.h>
#include "imagelib.h"
#endif

namespace PSECommon {
namespace Modules {

using std::cerr;
using std::endl;
using std::ostringstream;
using std::ofstream;

using PSECore::Datatypes::GeometryData;
using SCICore::Datatypes::DepthImage;

using SCICore::Datatypes::ColorImage;
using SCICore::GeomSpace::Light;
using SCICore::GeomSpace::Lighting;
using SCICore::GeomSpace::View;
using SCICore::Containers::to_string;
using SCICore::TclInterface::TCLTask;
using SCICore::Thread::Runnable;
using SCICore::Thread::Thread;

using SCICore::GeomSpace::GeomCone;
using SCICore::GeomSpace::GeomCylinder;
using SCICore::GeomSpace::GeomCappedCylinder;
using SCICore::GeomSpace::GeomSphere;
using SCICore::GeomSpace::GeomTri;
using SCICore::GeomSpace::GeomText;

class OpenGLHelper;
int query_OpenGL();

#define DO_REDRAW 0
#define DO_PICK 1
#define DO_GETDATA 2
#define REDRAW_DONE 4
#define PICK_DONE 5

struct GetReq {
    int datamask;
    FutureValue<GeometryData*>* result;
    GetReq(int, FutureValue<GeometryData*>* result);
    GetReq();
};

class OpenGL : public Renderer {
  
protected:
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    int maxlights;
    SCICore::GeomSpace::DrawInfoOpenGL* drawinfo;
    WallClockTimer fpstimer;
    double current_time;

    int old_stereo;
    GLuint imglist;
    void make_image();

    void redraw_obj(Salmon* salmon, Roe* roe, GeomObj* obj);
    void pick_draw_obj(Salmon* salmon, Roe* roe, GeomObj* obj);
    OpenGLHelper* helper;
    clString my_openglname;
    SCICore::Containers::Array1<XVisualInfo*> visuals;

   /* Call this each time an mpeg frame is generated. */
   void addMpegFrame();
   MpegEncoder mpEncoder;
   bool encoding_mpeg;

public:
    OpenGL();
    virtual ~OpenGL();
    virtual clString create_window(Roe* roe, const clString& name,
      const clString& width, const clString& height);
    virtual void redraw(Salmon*, Roe*, double tbeg, double tend,
      int ntimesteps, double frametime);
    void real_get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&, int&);
    virtual void get_pick(Salmon*, Roe*, int, int,
      GeomObj*&, GeomPick*&, int& );
    virtual void hide();
    virtual void dump_image(const clString&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat=1);

    clString myname;
    virtual void redraw_loop();
    SCICore::Thread::Mailbox<int> send_mb;
    SCICore::Thread::Mailbox<int> recv_mb;
    SCICore::Thread::Mailbox<GetReq> get_mb;

    Salmon* salmon;
    Roe* roe;
    double tbeg;
    double tend;
    int nframes;
    double framerate;
    void redraw_frame();

    int send_pick_x;
    int send_pick_y;

    GeomObj* ret_pick_obj;
    GeomPick* ret_pick_pick;
    int ret_pick_index;

    virtual void listvisuals(TCLArgs&);
    virtual void setvisual(const clString&, int i, int width, int height);

    View lastview;
    double znear, zfar;

    GeomCappedCylinder* stylusCylinder[2];
    GeomTri* stylusTriangle[4];
    GeomSphere* pinchSphere;
    Material* stylusMaterial[16], *pinchMaterial;
    
    GeomText* pinchText[2];
    GeomCappedCylinder* pinchCylinder[4];

    // these functions were added to clean things up a bit...

//protected:
    virtual void getData(int datamask,
      FutureValue<GeometryData*>* result);
  
protected:
    
    virtual void real_getData(int datamask,
      FutureValue<GeometryData*>* result);
  
    void initState(void);

};

} // namespace PSECommon
} // namespace Modules

#endif // SCI_project_module_OpenGL_h
