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
 *  ViewImage.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   September, 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <tcl.h>
#include <tk.h>
#include <stdlib.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Core/GuiInterface/UIvar.h>
#include <Core/Geom/Material.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>

#include <Core/Geom/GeomSwitch.h>
#include <Core/Algorithms/Visualization/RenderField.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Geom/OpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>

#include <typeinfo>
#include <iostream>

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;


namespace SCIRun {

// SWAP ----------------------------------------------------------------
template <class T>
inline void SWAP(T& a, T& b) {
  T temp;
  temp = a;
  a = b;
  b = temp;
}


class ViewImage : public Module
{
  enum {
    SHIFT_E	= 1,
    CAPS_LOCK_E = 2,
    CONTROL_E	= 4,
    ALT_E	= 8,
    M1_E	= 16,
    M2_E	= 32,
    M3_E	= 64,
    M4_E	= 128,
    BUTTON_1_E	= 256,
    BUTTON_2_E  = 512,
    BUTTON_3_E  = 1024
  } Tk_Button_State_e;

  struct NrrdVolume { 
    NrrdVolume		(GuiContext*ctx);
    void		reset();
    NrrdDataHandle	nrrd_;
    UIdouble		opacity_;
    UIint		invert_;
    UIint		flip_x_;
    UIint		flip_y_;
    UIint		flip_z_;
    UIint		transpose_yz_;
    UIint		transpose_xz_;
    UIint		transpose_xy_;
  };

  struct NrrdSlice {
    NrrdSlice(int axis=0, int slice=0, NrrdVolume *volume=0);
    int			axis_;  // which axis
    int			slice_num_;   // which slice # along axis
    NrrdDataHandle      nrrd_;
    
    bool		dirty_;
    
    unsigned int	tex_wid_;
    unsigned int	tex_hei_;

    unsigned int	wid_;
    unsigned int	hei_;

    float		opacity_;
    float		tex_coords_[8];  // s,t * 4 corners
    float		pos_coords_[12]; // x,y,z * 4 corners
    GLuint		tex_name_;

    NrrdVolume *	volume_;
  };
  typedef vector<NrrdSlice *>		NrrdSlices;
  typedef vector<NrrdSlices>		NrrdVolumeSlices;



  struct SliceWindow { 
    SliceWindow() { ASSERT(0); };
    SliceWindow(GuiContext *ctx);
    string		name_;
    void		reset();

    OpenGLViewport *	viewport_;
    NrrdSlices		slices_;
     
    UIint		slice_[3];
    UIint		axis_;
    UIdouble		zoom_;
      
    UIdouble		x_;
    UIdouble		y_;
    UIdouble		z_;
      
    UIdouble		scale_;
    UIdouble		bias_;

    UIint		clut_ww_;
    UIint		clut_wl_;
      
    UIint		auto_levels_;
    UIint		mode_;
    UIint		crosshairs_;
    UIint		snoop_;
    UIint		invert_;
    UIint		reverse_;
      
    UIint		mouse_x_;
    UIint		mouse_y_;
    
    GLfloat *		rgba_[4];
  };
  typedef vector<SliceWindow *>	SliceWindows;

  struct WindowLayout {
    WindowLayout	(GuiContext *ctx);
    //    void		reset();
    OpenGLContext *	opengl_;
    int			mouse_x_;
    int			mouse_y_;
    UIint		mode_;
    
    SliceWindows	windows_;
  };



  typedef vector<NrrdVolume *>		NrrdVolumes;
  typedef map<string, WindowLayout *>	WindowLayouts;

  WindowLayouts		layouts_;
  NrrdVolumes		volumes_;
  //  NrrdVolumeSlices	slices_;

  ColorMapHandle	colormap_;
  int			colormap_generation_;
  int			colormap_size_;



  Point			cursor_;

  int			max_slice_[3];
  double		scale_[3];
  double		center_[3];

  //! output port
  GeometryOPort *	ogeom_;

  void			redraw_all();
  void			redraw_window_layout(WindowLayout &);
  void			redraw_window(SliceWindow &);
  void			draw_slice(SliceWindow &, NrrdSlice &);
  
  bool			extract_colormap(SliceWindow &);
  bool			extract_clut(SliceWindow &);

  void			draw_guidelines(SliceWindow &, float, float, float);
  void			setup_gl_view(SliceWindow &);

  void			set_slice_coords(NrrdSlice &slice);


  void			extract_window_slices(SliceWindow &);
  void			extract_slice(NrrdVolume &, 
				      NrrdSlice &,
				      int axis, int slice_num);

  void			set_axis(SliceWindow &, unsigned int axis);
  void			next_slice(SliceWindow &);
  void			prev_slice(SliceWindow &);
  void			zoom_in(SliceWindow &);
  void			zoom_out(SliceWindow &);

  Point			screen_to_world(SliceWindow &,
					unsigned int x, unsigned int y);

  unsigned int		pow2(const unsigned int) const;
  unsigned int		log2(const unsigned int) const;
  bool			mouse_in_window(SliceWindow &);

  void			handle_gui_motion(GuiArgs &args);
  void			handle_gui_button(GuiArgs &args);
  void			handle_gui_keypress(GuiArgs &args);

  void			debug_print_state(int state);
  
public:
  ViewImage(GuiContext* ctx);
  virtual ~ViewImage();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
};

ViewImage::NrrdSlice::NrrdSlice(int axis, int slice, NrrdVolume *volume) :
  axis_(axis),
  slice_num_(slice),
  nrrd_(0),
  dirty_(true),
  tex_wid_(0),
  tex_hei_(0),
  wid_(0),
  hei_(0),
  opacity_(1.0),
  tex_name_(0),
  volume_(volume)
{
  int i;
  for (i = 0; i < 8; i++) tex_coords_[i] = 0;
  for (i = 0; i < 12; i++) pos_coords_[i] = 0;
}

ViewImage::SliceWindow::SliceWindow(GuiContext *ctx) :  
  viewport_(0),
  slices_(),
  axis_(ctx->subVar("axis"), 2),
  zoom_(ctx->subVar("zoom"), 100.0),
  x_(ctx->subVar("posx"),0.0),
  y_(ctx->subVar("posy"),0.0),
  z_(ctx->subVar("posz"),0.0),
  scale_(ctx->subVar("scale"), 1.0),
  bias_(ctx->subVar("bias"),0.0),
  clut_ww_(ctx->subVar("clut_ww"), -1),
  clut_wl_(ctx->subVar("clut_wl"), -1),
  auto_levels_(ctx->subVar("auto_levels"),0),
  mode_(ctx->subVar("mode"),0),
  crosshairs_(ctx->subVar("crosshairs"),0),
  snoop_(ctx->subVar("snoop"),0),
  invert_(ctx->subVar("invert"),0),
  reverse_(ctx->subVar("reverse"),0),
  mouse_x_(ctx->subVar("mouse_x"),0),
  mouse_y_(ctx->subVar("mouse_y"),0)
{
  for (int i = 0; i < 3; ++i) {
    //    slice_[i] = ctx->subVar("slice_"+to_string(i),0);
    slice_[i] = ctx->subVar("slice");
    slice_[i] = 0;
  }
  for (int i = 0; i < 4; ++i) rgba_[i] = 0; 
}


ViewImage::WindowLayout::WindowLayout(GuiContext *ctx) :  
  opengl_(0),
  mouse_x_(0),
  mouse_y_(0),
  windows_()
{
}


ViewImage::NrrdVolume::NrrdVolume(GuiContext *ctx) :
  nrrd_(0),
  opacity_(ctx->subVar("opacity"), 1.0),
  invert_(ctx->subVar("invert"), 0),
  flip_x_(ctx->subVar("flip_x"),0),
  flip_y_(ctx->subVar("flip_y"),0),
  flip_z_(ctx->subVar("flip_z"),0),
  transpose_yz_(ctx->subVar("transpose_yz"),0),
  transpose_xz_(ctx->subVar("transpose_xz"),0),
  transpose_xy_(ctx->subVar("transpose_xy"),0)
{
}


DECLARE_MAKER(ViewImage)

ViewImage::ViewImage(GuiContext* ctx) :
  Module("ViewImage", ctx, Filter, "Render", "SCIRun"),
  layouts_(),
  volumes_(),
  //  slices_(),
  colormap_(0),
  colormap_generation_(-1),
  colormap_size_(32),
  ogeom_(0)
{

}

void
ViewImage::SliceWindow::reset() {
  zoom_ = 0.0;

  x_ = 0.0;
  y_ = 0.0;
  z_ = 0.0;

  scale_ = 1.0;
  bias_ = 0.0;
  clut_ww_ = -1;
  clut_wl_ = -1;
  
  auto_levels_ = 0;
  mode_ = 0;
  crosshairs_ = 1;
  snoop_ = 0;
  invert_ = 0;
  reverse_ = 0;
  //  for (unsigned int v=0; v < windows_.size(); ++v)
  //  windows_[v]->reset();

}

void
ViewImage::NrrdVolume::reset() {
  opacity_ = 0.5;
  invert_ = 0;
  flip_x_ = 0;
  flip_y_ = 0;
  flip_z_ = 0;
  transpose_yz_ = 0;
  transpose_xz_ = 0;
  transpose_xy_ = 0;
}
  
  

    

ViewImage::~ViewImage()
{
}


// TODO: Query opengl max texture size
unsigned int
ViewImage::pow2(const unsigned int dim) const {
  unsigned int val = 1;
  while (val < dim) { val = val << 1; };
  return val;
}

unsigned int
ViewImage::log2(const unsigned int dim) const {
  unsigned int log = 0;
  unsigned int val = 1;
  while (val < dim) { val = val << 1; log++; };
  return log;
}


void
ViewImage::redraw_all()
{
  WindowLayouts::iterator pos = layouts_.begin();
  while (pos != layouts_.end()) {
    redraw_window_layout(*(*pos).second);
    pos++;
  }
}


void
ViewImage::redraw_window_layout(ViewImage::WindowLayout &layout)
{
  if (!layout.opengl_) return;

  // TODO: better make_current so we dont release every viewport
  //  if (!layout.opengl_->make_current()) {
  //  error("Unable to make GL window current");
  //  return;
  //}
 
  SliceWindows::iterator viter, vend = layout.windows_.end();
  for (viter = layout.windows_.begin(); viter != vend; ++viter) {
    redraw_window(**viter); 
  }

  //  layout.opengl_->release();
}



void
ViewImage::redraw_window(SliceWindow &window) {
  window.viewport_->make_current();
  window.viewport_->clear();//drand48(), drand48(), drand48(), 1.0);
  //  ASSERT(slices_[window.axis_][window.slice_num_]);
  for (unsigned int s = 0; s < window.slices_.size(); ++s)
    draw_slice(window, *window.slices_[s]);
  draw_guidelines(window, cursor_.x(), cursor_.y(), cursor_.z());
  window.viewport_->swap();
  window.viewport_->release();
}

  




bool
ViewImage::extract_colormap(SliceWindow &window)
{
  if (!colormap_.get_rep()) return false;
  bool recompute = false;
  int ww = window.clut_ww_;
  int wl = window.clut_wl_;
  if (colormap_->generation != colormap_generation_) recompute = true;
  if (ww != *window.clut_ww_) recompute = true;
  if (wl != *window.clut_wl_) recompute = true;
  if (!recompute) return false;

  colormap_generation_ = colormap_->generation;
  int i, c;
  colormap_size_ = pow2(colormap_->resolution());
  GLint max_size;    
  window.viewport_->make_current();
  glGetIntegerv(GL_MAX_PIXEL_MAP_TABLE, &max_size);
  window.viewport_->release();
  //  if (colormap_size_ > max_size) 
  colormap_size_ = max_size;
  
  const double scale = 
    double(colormap_->resolution()) / window.clut_ww_;
  
  for (i = 0; i < 4; i++) {
    if (window.rgba_[i]) delete [] window.rgba_[i];
    window.rgba_[i] = scinew GLfloat[colormap_size_];
  }

  const int min = window.clut_wl_ - window.clut_ww_/2;
  
  for (c = 0; c < colormap_size_; c++) 
    for (i = 0; i < 4; i++) {
      const int cmap_pos = Round(scale*(c-min));
      if (cmap_pos < 0) 
	window.rgba_[i][c] = 0.0;
      else if (cmap_pos >= colormap_->resolution())
	window.rgba_[i][c] = 1.0;
      else
	window.rgba_[i][c] =colormap_->get_rgba()[cmap_pos*4+i];
    }
  
  return true;
}

  


// draw_guidlines
// renders vertical and horizontal bars that represent
// selected slices in other dimensions
// if x, y, or z < 0, then that dimension wont be rendered
void
ViewImage::draw_guidelines(SliceWindow &window, float x, float y, float z) {
  window.viewport_->make_current();
  setup_gl_view(window);
  const bool inside = mouse_in_window(window);
  Vector tmp = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  tmp[window.axis_] = 0;
  float one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.8 };
  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };
  //  GLdouble blue[4] = { 0.3, 0.2, 0.7, 0.8 };
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 0.8 };
  

  if (inside) 
    glColor4dv(green);
  else 
    glColor4dv(red);
    

  glBegin(GL_QUADS);
  const int axis = window.axis_;
  int p = (axis+1)%3;
  int s = (axis+2)%3;
  double c[3];
  c[0] = x;
  c[1] = y;
  c[2] = z;
  for (int i = 0; i < 2; ++i) {
     if (c[p] > 0 && c[p] < max_slice_[p]) {
       glVertex3f(p==0?x:0.0, p==1?y:0.0, p==2?z:0.0);
       glVertex3f(p==0?x+one:0.0, p==1?y+one:0.0, p==2?z+one:0.0);
       glVertex3f(p==0?x+one:(axis==0?0.0:max_slice_[s]*scale_[s]),
		  p==1?y+one:(axis==1?0.0:max_slice_[s]*scale_[s]),
		  p==2?z+one:(axis==2?0.0:max_slice_[s]*scale_[s]));
       
       glVertex3f(p==0?x:(axis==0?0.0:max_slice_[s]*scale_[s]),
		  p==1?y:(axis==1?0.0:max_slice_[s]*scale_[s]),
		  p==2?z:(axis==2?0.0:max_slice_[s]*scale_[s]));
     }
     SWAP(p,s);
  }

  glEnd();


  if (!inside) {
    window.viewport_->release();
    return;
  }

    
  Point cvll = screen_to_world(window, 0, 0);
  Point cvur = screen_to_world(window, 
			       window.viewport_->max_width(), 
			       window.viewport_->max_height());
  cvll(window.axis_) = 0;
  cvur(window.axis_) = 0;
  glColor4dv(yellow);
  glBegin(GL_QUADS);
  for (int i = 0; i < 2; ++i) {
    if (c[p] < 0 || c[p] > max_slice_[p]) {
      glVertex3f(p==0?x:cvll(0), 
		 p==1?y:cvll(1), 
		 p==2?z:cvll(2));
      glVertex3f(p==0?x+one:cvll(0), 
		 p==1?y+one:cvll(1), 
		 p==2?z+one:cvll(2));
      glVertex3f(p==0?x+one:(axis==0?cvll(0):cvur(0)),
		 p==1?y+one:(axis==1?cvll(1):cvur(1)),
		 p==2?z+one:(axis==2?cvll(2):cvur(2)));
      
      glVertex3f(p==0?x:(axis==0?cvll(0):cvur(0)),
		 p==1?y:(axis==1?cvll(1):cvur(1)),
		 p==2?z:(axis==2?cvll(2):cvur(2)));
    } else {
      glVertex3f(p==0?x:cvll(0), p==1?y:cvll(1), p==2?z:cvll(2));
      glVertex3f(p==0?x+one:cvll(0), p==1?y+one:cvll(1), p==2?z+one:cvll(2));
      glVertex3f(p==0?x+one:(axis==0?cvll(0):0.0),
		 p==1?y+one:(axis==1?cvll(1):0.0),
		 p==2?z+one:(axis==2?cvll(2):0.0));
      
      glVertex3f(p==0?x:(axis==0?cvll(0):0.0),
		 p==1?y:(axis==1?cvll(1):0.0),
		 p==2?z:(axis==2?cvll(2):0.0));


      glVertex3f(p==0?x:(axis==0?0.0:max_slice_[s]*scale_[s]), 
		 p==1?y:(axis==1?0.0:max_slice_[s]*scale_[s]),
		 p==2?z:(axis==2?0.0:max_slice_[s]*scale_[s]));

      glVertex3f(p==0?x+one:(axis==0?0.0:max_slice_[s]*scale_[s]), 
		 p==1?y+one:(axis==1?0.0:max_slice_[s]*scale_[s]), 
		 p==2?z+one:(axis==2?0.0:max_slice_[s]*scale_[s]));

      glVertex3f(p==0?x+one:(axis==0?(axis==0?0.0:max_slice_[s]*scale_[s]):cvur(0)),
		 p==1?y+one:(axis==1?(axis==1?0.0:max_slice_[s]*scale_[s]):cvur(1)),
		 p==2?z+one:(axis==2?(axis==2?0.0:max_slice_[s]*scale_[s]):cvur(2)));
      
      glVertex3f(p==0?x:(axis==0?(axis==0?0.0:max_slice_[s]*scale_[s]):cvur(0)),
		 p==1?y:(axis==1?(axis==1?0.0:max_slice_[s]*scale_[s]):cvur(1)),
		 p==2?z:(axis==2?(axis==2?0.0:max_slice_[s]*scale_[s]):cvur(2)));
    }
    SWAP(p,s);
  }

      
  glEnd();


  window.viewport_->release();
}  




void
ViewImage::setup_gl_view(SliceWindow &window)
{
  static int here = 0;
  if (here) return;
  here = 1;
  glMatrixMode(GL_MODELVIEW);
  //  glPushMatrix();
  glLoadIdentity();

  int axis = window.axis_;

  if (window.axis_ == 0) {
    glRotated(-90,0.,1.,0.);
    glRotated(-90,1.,0.,0.);
  } else if (window.axis_ == 1) {
    glRotated(90,1.,0.,0.);
    glRotated(90,0.,1.,0.);
  }
  glTranslated((axis==0)?-double(window.slice_[axis]):0.0,
  	       (axis==1)?-double(window.slice_[axis]):0.0,
  	       (axis==2)?-double(window.slice_[axis]):0.0);

  glMatrixMode(GL_PROJECTION);
  //  glPushMatrix();
  glLoadIdentity();

  Vector x_dir = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  Vector y_dir = screen_to_world(window, 0, 1) - screen_to_world(window, 0, 0);
  int pri_axis = 0;
  int sec_axis = 0;
  for (int c = 0; c < 3; ++c) {
    if (fabs(x_dir[c]) > fabs(x_dir[pri_axis])) pri_axis = c;
    if (fabs(y_dir[c]) > fabs(y_dir[sec_axis])) sec_axis = c;
  }
  here = 0;

  double hwid = (window.viewport_->width()/(*window.zoom_/100.0))/2;
  double hhei = (window.viewport_->height()/(*window.zoom_/100.0))/2;
  
  double cx = double(*window.x_) + center_[pri_axis];
  double cy = double(*window.y_) + center_[sec_axis];
  
  glOrtho(cx - hwid, cx + hwid, cy - hhei, cy + hhei,
	  -max_slice_[axis], max_slice_[axis]);
  
  
  // // Then Scale to window coordinates: [0,0]..[Window Width, Window Height]
  //   GLdouble scales[3] = { 1.0, 1.0, 1.0 };
  //   if (window.axis_ == 0) {
  //     scales[1] /= window.opengl_->width();
  //     scales[2] /= window.opengl_->height();
  //   } else if (window.axis_ == 1) {
  //     scales[0] /= window.opengl_->height();
  //     scales[2] /= window.opengl_->width();
  //   } else /*if (window.axis_ == 2)*/ {
  //     scales[0] /= window.opengl_->width();
  //     scales[1] /= window.opengl_->height();
  //   }
  //  glScaled(scales[0], scales[1], scales[2]);
  //  glScaled(slice.flip_x_?-1.0:1.0, slice.flip_y_?-1.0:1.0, 1.0);
  //  glScaled(window.zoom_, window.zoom_, window.zoom_);
  //glTranslated(window.x_, window.y_, window.z_);
  //  glMatrixMode(GL_PROJECTION);
  //  glPushMatrix();
  //glLoadIdentity();
}


void
ViewImage::draw_slice(SliceWindow &window, NrrdSlice &slice)
{
  if (!slice.nrrd_.get_rep())
    return;

  if (slice.axis_ != window.axis_) return;

  // Indexes of the primary and secondary axes
  //  const unsigned int pri = axis_.get()==0?1:0;
  //const unsigned int sec = axis_.get()==2?1:2;

  // x and y texture coordinates of data to be drawn [0, 1]
  unsigned int i = 0;
  window.viewport_->make_current();

  setup_gl_view(window);

  glRasterPos2d(0.0, 0.0);

  GLfloat ones[4] = {1.0, 1.0, 1.0, 1.0};
  glColor4fv(ones);
  //glColor4f(slice.opacity_, slice.opacity_, slice.opacity_, slice.opacity_);
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);

  //  if (slice.invert_)
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //else 
  //glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);
    
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, ones);


  if (extract_colormap(window) ||  slice.dirty_) {
    if (glIsTexture(slice.tex_name_)) {
      glDeleteTextures(1,&slice.tex_name_);
      slice.tex_name_ = 0;
    }
    slice.dirty_ = false;
  }


  const bool bound = glIsTexture(slice.tex_name_);
  if (!bound) {
    glGenTextures(1, &slice.tex_name_);
  }
  glBindTexture(GL_TEXTURE_2D, slice.tex_name_);

	   
  if (!bound) {
    glPixelTransferf(GL_RED_SCALE, *window.scale_);
    glPixelTransferf(GL_GREEN_SCALE, *window.scale_);
    glPixelTransferf(GL_BLUE_SCALE, *window.scale_);
    glPixelTransferf(GL_ALPHA_SCALE, *window.scale_);
    glPixelTransferf(GL_RED_BIAS, *window.bias_);
    glPixelTransferf(GL_GREEN_BIAS, *window.bias_);
    glPixelTransferf(GL_BLUE_BIAS, *window.bias_);
    glPixelTransferf(GL_ALPHA_BIAS, *window.bias_);
    glPixelTransferi(GL_MAP_COLOR, 1);
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


    glPixelMapfv(GL_PIXEL_MAP_I_TO_R, colormap_size_, window.rgba_[0]);
    glPixelMapfv(GL_PIXEL_MAP_I_TO_G, colormap_size_, window.rgba_[1]);
    glPixelMapfv(GL_PIXEL_MAP_I_TO_B, colormap_size_, window.rgba_[2]);
    glPixelMapfv(GL_PIXEL_MAP_I_TO_A, colormap_size_, window.rgba_[3]);

    //int shift = log2(colormap_size_) - 8;
    //    glPixelTransferi(GL_INDEX_SHIFT, shift);
    //    glPixelTransferi(GL_INDEX_OFFSET, window.clut_wl_ - window.clut_ww_/2);

    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.tex_wid_, slice.tex_hei_, 
		 0, GL_COLOR_INDEX, GL_UNSIGNED_SHORT, slice.nrrd_->nrrd->data);    
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.tex_wid_, slice.tex_hei_, 
    //	   0, GL_ALPHA, GL_UNSIGNED_BYTE, slice.nrrd_->nrrd->data);    
  }


  set_slice_coords(slice);
  glBegin( GL_QUADS );
  for (i = 0; i < 4; i++) {
    glTexCoord2fv(&slice.tex_coords_[i*2]);
    glVertex3fv(&slice.pos_coords_[i*3]);
  }
  glEnd();
  glFlush();

  glMatrixMode(GL_PROJECTION);
  //  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  //  glPopMatrix();

  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) 
    cerr << "GL error : " << int(err) << std::endl;

  window.viewport_->release();

  
}



void
ViewImage::set_slice_coords(NrrdSlice &slice) {
  int axis = slice.axis_;
  int slice_num = slice.slice_num_;
  double x_pos=0,y_pos=0,z_pos=0;
  double x_wid=0,y_wid=0,z_wid=0;
  double x_hei=0,y_hei=0,z_hei=0;
  
  if (axis == 0) {
    x_pos = slice_num;
    y_wid = slice.volume_->nrrd_->nrrd->axis[1].size;
    z_hei = slice.volume_->nrrd_->nrrd->axis[2].size; 
  } else if (axis == 1) {
    y_pos = slice_num;
    x_wid = slice.volume_->nrrd_->nrrd->axis[0].size;
    z_hei = slice.volume_->nrrd_->nrrd->axis[2].size;
  } else /*if (axis == 2)*/ {
    z_pos = slice_num;
    x_wid = slice.volume_->nrrd_->nrrd->axis[0].size;
    y_hei = slice.volume_->nrrd_->nrrd->axis[1].size;
  }

  if (*slice.volume_->flip_x_) {
    x_pos += x_wid+x_hei;
    x_wid *= -1.0;
    x_hei *= -1.0;
  }

  if (*slice.volume_->flip_y_) {
    y_pos += y_wid+y_hei;
    y_wid *= -1.0;
    y_hei *= -1.0;
  }

  if (*slice.volume_->flip_z_) {
    z_pos += z_wid+z_hei;
    z_wid *= -1.0;
    z_hei *= -1.0;
  }

  if (*slice.volume_->transpose_yz_) {
    SWAP(y_wid, z_wid);
    SWAP(y_hei, z_hei);
  } else if (*slice.volume_->transpose_xz_) {
    SWAP(x_wid, z_wid);
    SWAP(x_hei, z_hei);
  } else if (*slice.volume_->transpose_xy_) {
    SWAP(x_wid, y_wid);
    SWAP(x_hei, y_hei);
  } 

  int i = 0;
  slice.pos_coords_[i++] = x_pos;
  slice.pos_coords_[i++] = y_pos;
  slice.pos_coords_[i++] = z_pos;

  slice.pos_coords_[i++] = x_pos+x_wid;
  slice.pos_coords_[i++] = y_pos+y_wid;
  slice.pos_coords_[i++] = z_pos+z_wid;

  slice.pos_coords_[i++ % 12] = x_pos+x_wid+x_hei;
  slice.pos_coords_[i++ % 12] = y_pos+y_wid+y_hei;
  slice.pos_coords_[i++ % 12] = z_pos+z_wid+z_hei;

  slice.pos_coords_[i++ % 12] = x_pos+x_hei;
  slice.pos_coords_[i++ % 12] = y_pos+y_hei;
  slice.pos_coords_[i++ % 12] = z_pos+z_hei;

  for (i = 0; i < 12; ++i) 
    slice.pos_coords_[i] *= scale_[i%3];
}


void
ViewImage::extract_window_slices(SliceWindow &window) {
  unsigned int v, s, a;
  for (s = 0; s < window.slices_.size(); ++s)
    delete window.slices_[s];
  window.slices_.clear();

  for (v = 0; v < volumes_.size(); ++v) {
    for (a = 0; a < 3; a++) {
      window.slices_.push_back
	(scinew NrrdSlice(a, window.slice_[a], volumes_[v]));
      if (int(window.axis_) == a)
	extract_slice(*volumes_[v], *window.slices_.back(), a, *window.slice_[a]);
    }
  }
}



void
ViewImage::extract_slice(NrrdVolume &volume,
			 NrrdSlice &slice,
			 int axis, int slice_num)
{
  if (!volume.nrrd_.get_rep()) return;

  slice.nrrd_ = scinew NrrdData;
  slice.nrrd_->nrrd = nrrdNew();
  
  if (nrrdSlice(slice.nrrd_->nrrd, volume.nrrd_->nrrd, axis, slice_num)) {
    char *err = biffGetDone(NRRD);
    error(string("Error Slicing nrrd: ") + err);
    free(err);
  }

  slice.axis_ = axis;
  slice.slice_num_ = slice_num;
  slice.dirty_ = true;
  slice.wid_     = slice.nrrd_->nrrd->axis[0].size;
  slice.hei_     = slice.nrrd_->nrrd->axis[1].size;
  slice.tex_wid_ = pow2(slice.wid_);
  slice.tex_hei_ = pow2(slice.hei_);
  slice.opacity_ = volume.opacity_;
  slice.tex_name_ = 0;
  slice.volume_ = &volume;


  NrrdDataHandle temp1 = scinew NrrdData;
  temp1->nrrd = nrrdNew();  
  NrrdRange *range =nrrdRangeNewSet(slice.nrrd_->nrrd,nrrdBlind8BitRangeState);

//   if (nrrdQuantize(temp1->nrrd, slice.nrrd_->nrrd, range, 8)) {
//     char *err = biffGetDone(NRRD);
//     error(string("Trouble quantizing: ") + err);
//     free(err);
//     return;
//  }

  int minp[2] = { 0, 0 };
  int maxp[2] = { slice.tex_wid_-1, slice.tex_hei_-1 };

  if (nrrdPad(temp1->nrrd, slice.nrrd_->nrrd, minp,maxp,nrrdBoundaryPad, 0.0)) 
  {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    free(err);
  }
  slice.nrrd_ = temp1;

  // x and y texture coordinates of data to be drawn [0, 1]
  const double tex_x = double(slice.wid_)/double(slice.tex_wid_);
  const double tex_y = double(slice.hei_)/double(slice.tex_hei_);
  unsigned int i = 0;
  slice.tex_coords_[i++] = 0.0; 
  slice.tex_coords_[i++] = 0.0;
  slice.tex_coords_[i++] = tex_x;
  slice.tex_coords_[i++] = 0.0;
  slice.tex_coords_[i++] = tex_x;
  slice.tex_coords_[i++] = tex_y;
  slice.tex_coords_[i++] = 0.0;
  slice.tex_coords_[i++] = tex_y;
}

void
ViewImage::set_axis(SliceWindow &window, unsigned int axis) {
  window.axis_ = axis;
  extract_window_slices(window);
  redraw_window(window);
}

void
ViewImage::prev_slice(SliceWindow &window)
{
  if (window.slice_[window.axis_] == 0) 
    return;
  if (window.slice_[window.axis_] < 1)
    window.slice_[window.axis_] = 0;
  else 
    window.slice_[window.axis_]--;
  
  extract_window_slices(window);
  redraw_window(window);
}

void
ViewImage::next_slice(SliceWindow &window)
{
  if (*window.slice_[window.axis_] == max_slice_[window.axis_]) 
    return;
  if (window.slice_[window.axis_] > max_slice_[window.axis_])
    window.slice_[window.axis_] = max_slice_[window.axis_];
  else
    window.slice_[window.axis_]++;
  extract_window_slices(window);
  redraw_window(window);
}

void
ViewImage::zoom_in(SliceWindow &window)
{
  window.zoom_ *= 1.1;
  redraw_window(window);
}

void
ViewImage::zoom_out(SliceWindow &window)
{
  window.zoom_ /= 1.1;
  redraw_window(window);
}

  
Point
ViewImage::screen_to_world(SliceWindow &window, 
			   unsigned int x, unsigned int y) {
  GLdouble model[16];
  GLdouble proj[16];
  GLint	   view[4];
  GLdouble xyz[3];
  window.viewport_->make_current();
  setup_gl_view(window);
  glGetDoublev(GL_MODELVIEW_MATRIX,model);
  glGetDoublev(GL_PROJECTION_MATRIX,proj);
  glGetIntegerv(GL_VIEWPORT, view);
  gluUnProject(double(x), double(y), double(window.slice_[window.axis_]),
	       model, proj, view, xyz+0, xyz+1, xyz+2);
  window.viewport_->release();
  xyz[window.axis_] = window.slice_[window.axis_];
  return Point(xyz[0], xyz[1], xyz[2]);
}
  

void
ViewImage::execute()
{
  update_state(Module::JustStarted);
  NrrdIPort *nrrd1_port = (NrrdIPort*)get_iport("Nrrd1");
  NrrdIPort *nrrd2_port = (NrrdIPort*)get_iport("Nrrd2");
  ColorMapIPort *color_iport = (ColorMapIPort *)get_iport("ColorMap");

  ogeom_ = (GeometryOPort *)get_oport("Scene Graph");

  if (!nrrd1_port) 
  {
    error("Unable to initialize iport Nrrd1.");
    return;
  }

  if (!nrrd2_port) 
  {
    error("Unable to initialize iport Nrrd.");
    return;
  }

  if (!color_iport) 
  {
    error("Unable to initialize iport ColorMap.");
    return;
  }

  if (!ogeom_) {
    error("Unable to initialize oport Scene Graph.");
    return;
  }

  update_state(Module::NeedData);
  NrrdDataHandle nrrd1, nrrd2;
  nrrd1_port->get(nrrd1);
  nrrd2_port->get(nrrd2);
  
  if (!nrrd1.get_rep() && !nrrd2.get_rep())
  {
    error ("Unable to get a nrrd from Nrrd1 or Nrrd2.");
    return;
  }
  int i;

					 
  for (i = 0; i < 3; i++)
    max_slice_[i] = -1;

  if (nrrd1.get_rep() && nrrd2.get_rep()) 
    for (i = 0; i < 3; i++)
      if (nrrd1->nrrd->axis[i].size != nrrd2->nrrd->axis[i].size) {
	error("Both input nrrds must have same dimensions.");
	error("  Only rendering first inpput.");
	nrrd2 = 0;
      } else
	max_slice_[i] = nrrd1->nrrd->axis[i].size-1;
  else if (nrrd1.get_rep()) 
    for (i = 0; i < 3; i++)
      max_slice_[i] = nrrd1->nrrd->axis[i].size-1;
  else if (nrrd2.get_rep()) 
    for (i = 0; i < 3; i++)
      max_slice_[i] = nrrd2->nrrd->axis[i].size-1;


  // next line not efficent, but easy to code
  volumes_.clear();
  if (nrrd1.get_rep()) {
    volumes_.push_back(scinew NrrdVolume(ctx->subVar("nrrd1",0)));
    volumes_.back()->nrrd_ = nrrd1;
  }

  if (nrrd2.get_rep()) {
    volumes_.push_back(scinew NrrdVolume(ctx->subVar("nrrd2",0)));
    volumes_.back()->nrrd_ = nrrd2;
  }

  for (unsigned int v = 0; v < volumes_.size(); ++v) {
    NrrdVolume *volume = volumes_[v];
    for (i = 0; i < 3; ++i) {
      scale_[i] = (airExists_d(volume->nrrd_->nrrd->axis[i].spacing) ?
		   volume->nrrd_->nrrd->axis[i].spacing : 1.0);
      center_[i] = (volume->nrrd_->nrrd->axis[i].size*scale_[i])/2;
    }
  }


  
  color_iport->get(colormap_);
  update_state(Module::Executing);  
  redraw_all();
  update_state(Module::Completed);

  //  ogeom_->addObj(geom, fname + name);
  //  ogeom_->flushViews();
  
}

bool
ViewImage::mouse_in_window(SliceWindow &window) {
  return (window.mouse_x_ >= window.viewport_->x() && 
	  window.mouse_x_ < window.viewport_->x()+window.viewport_->width() &&
	  window.mouse_y_ >= window.viewport_->y() &&
	  window.mouse_y_ < window.viewport_->y()+window.viewport_->height());
}

  


void
ViewImage::handle_gui_motion(GuiArgs &args) {
  int state;
  if (!string_to_int(args[4], state)) {
    args.error ("Cannot convert motion state");
    return;
  }
  //  cerr << "Motion: X: " << args[2] << " Y: " << args[3];
  // debug_print_state(state);
  //  cerr << "  Time: " << args[5] << std::endl;
  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle motion on "+args[2]);
    return;
  }

  WindowLayout &layout = *layouts_[args[2]];
  int x, y;
  string_to_int(args[3], x);
  string_to_int(args[4], y);
  y = layout.opengl_->yres() - 1 - y;
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    window.mouse_x_ = x; // - window.viewport_->x();
    window.mouse_y_ = y; //- window.viewport_->y();
    if (mouse_in_window(window))
      cursor_ = screen_to_world(window, x, y);
  }
  redraw_all();


}


void
ViewImage::debug_print_state(int state) {
  cerr << "State: " << state;
  vector<string> modstrings;
  modstrings.push_back("Shift"); // 1
  modstrings.push_back("Caps Lock"); // 2
  modstrings.push_back("Control"); // 8
  modstrings.push_back("Alt"); // 4
  modstrings.push_back("M1"); // 16
  modstrings.push_back("M2"); // 32
  modstrings.push_back("M3"); // 64
  modstrings.push_back("M4"); // 128
  modstrings.push_back("Button1"); // 256
  modstrings.push_back("Button2"); // 512
  modstrings.push_back("Button3"); // 1024
  int i = 1;
  unsigned int power = 0;
  while (i < state) {
    if (state & i)
      if (power < modstrings.size())
	cerr << modstrings[power] << "-";
      else
	cerr << i << "-";  
    i <<= 1;
    ++power;
  }
  cerr << std::endl;
}



void
ViewImage::handle_gui_button(GuiArgs &args) {
  int button;
  int state;

  if (args.count() != 5) {
    args.error(args[0]+" "+args[1]+" expects a window #, button #, and state");
    return;
  }

  if (!string_to_int(args[3], button)) {
    args.error ("Cannot convert window #");
    return;
  }

  if (!string_to_int(args[4], state)) {
    args.error ("Cannot convert button state");
    return;
  }

  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle motion on "+args[2]);
    return;
  }

  WindowLayout &layout = *layouts_[args[2]];
  
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    if (!mouse_in_window(window)) continue;
    switch (button) {
    case 4:
      if (state & CONTROL_E == CONTROL_E) 
	zoom_in(window);
      else
	next_slice(window);
      break;
      
    case 5:
      if (state & SHIFT_E == SHIFT_E) 
	zoom_out(window);
      else
	prev_slice(window);
      break;
      
    default: 
      break;
    }
  }
}

void
ViewImage::handle_gui_keypress(GuiArgs &args) {
  int keycode;

  if (false)
    for (int i = 0; i < args.count(); ++i)
      cerr << args[i] << (i == args.count()-1)?"\n":" ";
  
  if (args.count() != 6) {
    args.error(args[0]+" "+args[1]+" expects a win #, keycode, keysym,& time");
    return;
  }
  

  if (!string_to_int(args[3], keycode)) {
    args.error ("Cannot convert keycode");
    return;
  }

  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle motion on "+args[2]);
    return;
  }

  WindowLayout &layout = *layouts_[args[2]];

  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    if (!mouse_in_window(window)) continue;
    if (args[4] == "r") {
      //    reset();
      //extract_slice();
      //draw();
    } else if (args[4] == "equal" || args[4] == "plus") {
      zoom_in(window);
    } else if (args[4] == "minus" || args[4] == "underscore") {
      zoom_out(window);
    } else if (args[4] == "0") {
      set_axis(window, 0);
    } else if (args[4] == "1") {
      set_axis(window, 1);
    } else if (args[4] == "2") {
      set_axis(window, 2);
    } else if (args[4] == "3") {
      window.viewport_->resize(0.0, 0.0, 0.5, 0.5);
      redraw_window(window);
    } else if (args[4] == "4") {
      window.viewport_->resize(0.0, 0.5, 0.5, 0.5);
      redraw_window(window);
    } else if (args[4] == "5") {
      window.viewport_->resize(0.5, 0.0, 0.5, 0.5);
      redraw_window(window);
    } else if (args[4] == "6") {
      window.viewport_->resize(0.5, 0.5, 0.5, 0.5);
      redraw_window(window);
    } else if (args[4] == "7") {
      window.viewport_->resize(0., 0., 1., 1.);
      redraw_window(window);
    } else if (args[4] == "i") {
      window.invert_ = window.invert_?0:1;
      redraw_window(window);
    } else if (args[4] == "x") {
      for (unsigned int v = 0; v < volumes_.size(); ++v) 
	volumes_[v]->flip_x_ = volumes_[v]->flip_x_?0:1;
      redraw_all();
    } else if (args[4] == "y") {
      for (unsigned int v = 0; v < volumes_.size(); ++v) 
	volumes_[v]->flip_y_ = volumes_[v]->flip_y_?1:0;
      redraw_all();
    } else if (args[4] == "z") {
      for (unsigned int v = 0; v < volumes_.size(); ++v) 
	volumes_[v]->flip_z_ = volumes_[v]->flip_z_?1:0;
      redraw_all();
    } else if (args[4] == "a") {
      window.scale_ /= 1.1;
      redraw_window(window);
    } else if (args[4] == "s") {
      window.scale_ *= 1.1;
      redraw_window(window);
    } else if (args[4] == "q") {
      window.bias_ += 0.1;
      redraw_window(window);
    } else if (args[4] == "w") {
      window.scale_ += 0.1;
      redraw_window(window);
    } else if (args[4] == "Left") {
      window.x_ -= 1.0;
      redraw_window(window);
    } else if (args[4] == "Right") {
      window.x_ += 1.0;
      redraw_window(window);
    } else if (args[4] == "Down") {
      window.y_ -= 1.0;
      redraw_window(window);
    } else if (args[4] == "Up") {
      window.y_ += 1.0;
      redraw_window(window);
    } else if (args[4] == "less" || args[4] == "comma") {
      prev_slice(window);
    } else if (args[4] == "greater" || args[4] == "period") {
      next_slice(window);
    } 
  }
}


void
ViewImage::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("ViewImage needs a minor command");
    return;
  } else if (args[1] == "motion") handle_gui_motion(args);
  else if (args[1] == "button") handle_gui_button(args);
  else if (args[1] == "keypress") handle_gui_keypress(args);
  else if(args[1] == "setgl") {
    ASSERT(layouts_.find(args[2]) == layouts_.end());
    layouts_[args[2]] = scinew WindowLayout(ctx->subVar(args[3],0));
    layouts_[args[2]]->opengl_ = scinew OpenGLContext(gui, args[2]);

  } else if(args[1] == "add_viewport") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    WindowLayout *layout = layouts_[args[2]];
    SliceWindow *viewport = scinew SliceWindow(ctx->subVar(args[3],0));
    viewport->name_ = args[3];
    layout->windows_.push_back(viewport);
    viewport->viewport_ = scinew OpenGLViewport(layout->opengl_);
    if (args[2].find("bottomr") != string::npos) {
      viewport->viewport_->resize(0.0, 0.0, 0.5, 0.5);

      viewport = scinew SliceWindow(ctx->subVar(args[3],0));
      viewport->name_ = args[3]+"-upperleft";
      layout->windows_.push_back(viewport);
      viewport->viewport_ = scinew OpenGLViewport(layout->opengl_, 0.0, 0.5, 0.5, 0.5);
      viewport = scinew SliceWindow(ctx->subVar(args[3],0));
      viewport->name_ = args[3]+"-lowerright";
      layout->windows_.push_back(viewport);
      viewport->viewport_ = scinew OpenGLViewport(layout->opengl_, 0.5, 0.0, 0.5, 0.5);

      viewport = scinew SliceWindow(ctx->subVar(args[3],0));
      viewport->name_ = args[3]+"-upperright";
      layout->windows_.push_back(viewport);
      viewport->viewport_ = scinew OpenGLViewport(layout->opengl_, 0.5, 0.5, 0.5, 0.5);
    }
      


  } else if(args[1] == "redraw") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    redraw_window_layout(*layouts_[args[2]]);
  } else if(args[1] == "redrawall") {
    redraw_all();
  } else if(args[1] == "rebind") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    WindowLayout &layout = *layouts_[args[2]];
    for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
      SliceWindow &window = *layout.windows_[w];
      extract_window_slices(window);
    }
    redraw_window_layout(layout);
  } else Module::tcl_command(args, userdata);
}




} // End namespace SCIRun


