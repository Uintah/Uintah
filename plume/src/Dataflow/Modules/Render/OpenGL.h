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

#ifndef SCIRun_src_Dataflow_Modules_Render_OpenGL_h
#define SCIRun_src_Dataflow_Modules_Render_OpenGL_h

#include <sci_defs/image_defs.h>

#include <tcl.h>
#include <tk.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Modules/Render/Viewer.h>
#include <Dataflow/Modules/Render/Ball.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Datatypes/Image.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/Light.h>
#include <Core/Geom/Lighting.h>
#include <Core/Geom/GeomRenderMode.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/TkOpenGLContext.h>
#include <Core/Math/Trig.h>
#include <Core/Util/Timer.h>

#ifdef HAVE_MPEG
#include <mpege.h>
#endif // HAVE_MPEG

#ifdef __sgi
#include <X11/extensions/SGIStereo.h>
#endif // __sgi

namespace SCIRun {

using std::ostringstream;
using std::ofstream;
using std::vector;

class OpenGLHelper;
class GuiArgs;
class Pbuffer;

struct GetReq {
  int datamask;
  FutureValue<GeometryData*>* result;
  GetReq(int, FutureValue<GeometryData*>* result);
  GetReq();
};

struct ImgReq {
  ImgReq(const string& n, const string& t, int rx, int ry);
  ImgReq();
  string name;
  string type;
  int resx;
  int resy;
};

struct Frustum {
  double znear;
  double zfar;
  double left;
  double right;
  double bottom;
  double top;
  double width;
  double height;
};

struct HiRes {
  double nrows;
  double ncols;
  int row;
  int col;
  int resx;
  int resy;
};

class OpenGL {
public:
  OpenGL(GuiInterface* gui, Viewer *, ViewWindow *);
  ~OpenGL();
  void			kill_helper();
  void			start_helper();
  static bool		query(GuiInterface* gui);
  void			redraw_loop();
  void			get_pick(int, int, GeomHandle&, GeomPickHandle&, int&);
  void			redraw(double tbeg, double tend,
			       int ntimesteps, double frametime);
  void			getData(int datamask,
				FutureValue<GeometryData*>* result);
  bool			compute_depth(const View& view,
				      double& near, double& far);
  bool			compute_fog_depth(const View& view,
					  double& near, double& far,
					  bool visible_only);
  void			saveImage(const string& fname,
				  const string& type = "ppm",
				  int x=640, int y=512);

  // Adds a DO_SYNC_FRAME to the send_mailbox_.
  void                  scheduleSyncFrame();
  
  // Compute world space point under cursor (x,y).  If successful,
  // set 'p' to that value & return true.  Otherwise, return false.
  bool			pick_scene(int x, int y, Point *p);

  // Public Member Variables
  int			xres_;
  int			yres_;
  bool			doing_image_p_;
  bool			doing_movie_p_;
  bool			make_MPEG_p_;
  int			current_movie_frame_;
  string		movie_name_;
  // True if we want only to dump a movie frame when a DO_SYNC_FRAME
  // message is received.
  bool                  doing_sync_frame_;
  // If doing_sync_frame_ is set, only write a frame if
  // dump_sync_frame_ is true.  This will prevent dumping frames on
  // regular redraws.
  bool                  dump_sync_frame_;

  TkOpenGLContext *	tk_gl_context_;
  TkOpenGLContext *	old_tk_gl_context_;
  string		myname_;
#ifdef __APPLE__
  bool                  apple_wait_a_second_;
#endif

private:
  void			redraw_frame();
  void			setFrustumToWindowPortion();
  void			deriveFrustum();
  void			redraw_obj(Viewer*, ViewWindow*, GeomHandle obj);
  void			pick_draw_obj(Viewer* , ViewWindow*, GeomHandle obj);
  void			dump_image(const string&, const string& type = "raw");
  void			put_scanline(int, int, Color* scanline, int repeat=1);
  void			StartMpeg(const string& fname);
  void			AddMpegFrame();
  void			EndMpeg();
  void			real_get_pick(int, int, GeomHandle&, 
				      GeomPickHandle&, int&);
  void			render_and_save_image(int x, int y,
					      const string& fname,
					      const string& type = "ppm");
  void			real_getData(int datamask, 
				     FutureValue<GeometryData*>* result);
  void			render_rotation_axis(const View &view,
					     bool do_stereo, int i, 
					     const Vector &eyesep);

  // Private Member Variables
  GuiInterface*		gui_;
  Runnable *		helper_;
  Thread *		helper_thread_;


  Viewer*		viewer_;
  ViewWindow*		view_window_;
  DrawInfoOpenGL*	drawinfo_;
  WallClockTimer	fps_timer_;
  Frustum		frustum_;
  HiRes			hi_res_;
  bool			dead_;
  bool			do_hi_res_;
  bool			encoding_mpeg_;
  int			max_gl_lights_;
  int			animate_num_frames_;
  double		animate_time_begin_;
  double		animate_time_end_;
  double		animate_framerate_;
  double		znear_;
  double		zfar_;
  double		current_time_;
  unsigned int		frame_count_;
  // HACK -- support data for get_pixel_depth, assume no screen is > 1280x1024
  float			depth_buffer_[1310720]; // 1310720 = 1280x1024 
  GLdouble		modelview_matrix_[16];
  GLdouble		projection_matrix_[16];
  GLint			viewport_matrix_[4];
  View			cached_view_;

  // Mouse Picking variables
  int			send_pick_x_;
  int			send_pick_y_;
  int			ret_pick_index_;
  GeomHandle		ret_pick_obj_;
  GeomPickHandle	ret_pick_pick_;

  // Thread Communication Mailboxes
  Mailbox<int>		send_mailbox_;
  Mailbox<int>		recv_mailbox_;
  Mailbox<GetReq>	get_mailbox_;
  Mailbox<ImgReq>	img_mailbox_;

#ifdef HAVE_MPEG
  FILE *		mpeg_file_;
  MPEGe_options		mpeg_options_;
#endif // HAVE_MPEG

  Pbuffer		*pbuffer_;
};

} // End namespace SCIRun

#endif // of #ifndef SCIRun_src_Dataflow_Modules_Render_OpenGL_h
