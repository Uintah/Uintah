//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : OpenGLViewer.h
//    Author : Martin Cole
//    Date   : Sat May 27 08:51:31 2006
//    Much of this code was taken from Dataflow/Modules/Render/* 
//    which was mostly written by Steve Parker.


#if !defined(OpenGLViewer_h)
#define OpenGLViewer_h

#include <sci_defs/image_defs.h>
#include <sci_glx.h>
#include <tcl.h>
#include <tk.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include <Core/Malloc/Allocator.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/Tools/ToolManager.h>
#include <Core/Events/Tools/Ball.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Geom/Light.h>
#include <Core/Geom/Lighting.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/IndexedGroup.h>
#include <Core/Geom/GeomViewerItem.h>
#include <Core/Geom/View.h>
#include <Core/Geom/OpenGLContext.h>
#include <Core/Util/Timer.h>

#ifdef _WIN32
#  include <sci_gl.h>
#endif

#ifdef HAVE_MPEG
#  include <mpege.h>
#endif // HAVE_MPEG

#ifdef __sgi
#  include <X11/extensions/SGIStereo.h>
#endif // __sgi

namespace SCIRun {

using std::ostringstream;
using std::ofstream;
using std::vector;

class Pbuffer;

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

class OpenGLViewer : public Runnable 
{
public:
  enum draw_type_e {
    DEFAULT_E,
    WIRE_E,
    FLAT_E,
    GOURAUD_E
  };

  OpenGLViewer(OpenGLContext*);
  virtual ~OpenGLViewer();
  // main rendering loop is the run method.
  virtual void          run();

  void                  get_pick(int, int, GeomHandle&, GeomPickHandle&, int&);
  void                  redraw(double tbeg, double tend,
                               int ntimesteps, double frametime);
  //   void                  getData(int datamask,
  //                                 FutureValue<GeometryData*>* result);
  bool                  compute_depth(const View& view,
                                      double& near, double& far);
  bool                  compute_fog_depth(const View& view,
                                          double& near, double& far,
                                          bool visible_only);
  void                  save_image(const string& fname,
				   const string& type = "ppm",
				   int x=640, int y=512);

  
  // Compute world space point under cursor (x,y).  If successful,
  // set 'p' to that value & return true.  Otherwise, return false.
  bool                  pick_scene(int x, int y, Point *p);

  bool                  do_stereo_p() { return do_stereo_; }
  bool                  do_hi_res_p() { return do_hi_res_; }

  // FIX_ME hook these up to tools for setting state...
  bool                do_backface_cull_p() { return false; }
  bool                do_display_list_p()  { return false; }
  bool                do_fog_p()           { return false; }
  bool                do_lighting_p()      { return true; }
  bool                do_ortho_view_p()    { return ortho_view_; }
  bool                fog_visibleonly_p()  { return fog_visibleonly_; }
  bool                do_rotation_axis_p() { return true; }
  bool                do_picking_p()       { return false; }
  bool                do_bbox_p()          { return false; }

  const Color&          bgcolor() { return bgcolor_; }

private:
  void                  redraw_frame();
  GeomHandle            create_viewer_axes() ;
  void                  draw_visible_scene_graph();
  bool                  item_visible_p(GeomViewerItem* si);
  void                  get_bounds(BBox &bbox, bool check_visible = true);
  void                  get_bounds_all(BBox &bbox) { get_bounds(bbox, false); }
  void                  set_state(DrawInfoOpenGL* drawinfo);
  void                  setFrustumToWindowPortion();
  void                  deriveFrustum();
  void                  redraw_obj(MaterialHandle def, GeomHandle obj);
  void                  pick_draw_obj(MaterialHandle def, GeomHandle obj);
  void                  dump_image(const string&, const string& type = "raw");
  void                  put_scanline(int, int, Color* scanline, int repeat=1);
  void                  StartMpeg(const string& fname);
  void                  AddMpegFrame();
  void                  EndMpeg();
  void                  real_get_pick(int, int, GeomHandle&, 
                                      GeomPickHandle&, int&);

  void                  render_and_save_image();

  void                  render_rotation_axis(const View &view,
                                             bool do_stereo, int i, 
                                             const Vector &eyesep);


  // Private Member Variables
  int                   xres_;
  int                   yres_;
  bool                  doing_image_p_;
  bool                  doing_movie_p_;
  bool                  make_MPEG_p_;
  string                movie_frame_extension_;  // Currently "png" or "ppm".
  int                   current_movie_frame_;
  string                movie_name_;
  OpenGLContext        *gl_context_;
  string                myname_;
  DrawInfoOpenGL*       drawinfo_;
  WallClockTimer        fps_timer_;
  Frustum               frustum_;
  HiRes                 hi_res_;
  bool                  dead_;
  bool                  do_hi_res_;
  bool                  encoding_mpeg_;
  int                   max_gl_lights_;
  int                   animate_num_frames_;
  double                animate_time_begin_;
  double                animate_time_end_;
  double                animate_framerate_;
  double                znear_;
  double                zfar_;
  double                current_time_;
  unsigned int          frame_count_;
  vector<float>         depth_buffer_;
  GLdouble              modelview_matrix_[16];
  GLdouble              projection_matrix_[16];
  GLint                 viewport_matrix_[4];
  View                  cached_view_;

  // Mouse Picking variables
  int                   send_pick_x_;
  int                   send_pick_y_;
  int                   ret_pick_index_;
  GeomHandle            ret_pick_obj_;
  GeomPickHandle        ret_pick_pick_;

  ToolManager                      tm_;
  EventManager::event_mailbox_t   *events_;

#ifdef HAVE_MPEG
  FILE *                mpeg_file_;
  MPEGe_options         mpeg_options_;
#endif // HAVE_MPEG

  Lighting               lighting_;
  Pbuffer               *pbuffer_;
  Color                  bgcolor_;
  View                   homeview_;
  View                   view_;
  bool                   do_stereo_;
  float                  ambient_scale_;	     
  float                  diffuse_scale_;	     
  float                  specular_scale_;	     
  float                  shininess_scale_;	     
  float                  emission_scale_;	     
  float                  line_width_;	     
  float                  point_size_;	     
  float                  polygon_offset_factor_;
  float                  polygon_offset_units_; 
  double                 eye_sep_base_; //! used in stereo rendering.
  bool                   ortho_view_;
  bool                   fogusebg_;
  Color                  fogcolor_;
  double                 fog_start_;
  double                 fog_end_;
  bool                   fog_visibleonly_;
  vector<GeomHandle>	 internal_objs_;
  vector<bool>		 internal_objs_visible_p_;   
  GeomSphere            *focus_sphere_;
  GeomIndexedGroup      *scene_graph_;
  map<string, bool>	 visible_;
  map<string, int>	 obj_tag_;
  MaterialHandle         default_material_;
  draw_type_e            draw_type_;
  bool                   capture_z_data_;
};

} // End namespace SCIRun

#endif // OpenGLViewer_h
