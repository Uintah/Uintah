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
//    File   : ICUMonitor.cc
//    Author : Martin Cole, McKay Davis
//    Date   : Thu Nov 11 15:54:44 2004

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <tcl.h>
#include <tk.h>
#include <stdlib.h>


#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/Timer.h>
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

#include <Core/Geom/FreeType.h>

#include <Core/Util/Environment.h>

#include <typeinfo>
#include <iostream>

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;


namespace VS {
using namespace SCIRun;
using std::cerr;
using std::endl;

class RTDraw;

class ICUMonitor : public Module
{
public:
  ICUMonitor(GuiContext* ctx);
  virtual ~ICUMonitor();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  void			redraw_all();
  void                  inc_time(double d);
  
private:
  //! Storage for each digit texture.
  struct LabelTex
  {
    LabelTex() :
      tex_width_(0),
      tex_height_(0),
      u_(0.0),
      v_(0.0),
      tex_id_(0),
      text_(string("bogus"))
    {}

    LabelTex(string t) :
      tex_width_(0),
      tex_height_(0),
      u_(0.0),
      v_(0.0),
      tex_id_(0),
      text_(t)
    {}
    void draw (float px, float py, float sx, float sy);
    void bind (FreeTypeFace *font);
    
    unsigned      tex_width_;
    unsigned      tex_height_;
    float         u_;
    float         v_;
    GLuint        tex_id_;
    string        text_;
  };


  struct Plot
  {
    Plot() :
      nw_label_(0),
      sw_label_(0),
      min_ref_label_(0),
      max_ref_label_(0),
      label_(0),
      index_(-1),
      snd_(0),
      min_(0.0),
      max_(1.0),
      r_(1.0),
      g_(1.0),
      b_(1.0)
    {}
    
    LabelTex *nw_label_;
    LabelTex *sw_label_;
    LabelTex *min_ref_label_;
    LabelTex *max_ref_label_;
    LabelTex *label_;
    int       index_;
    int       snd_;
    float     min_;
    float     max_;
    float     r_;
    float     g_;
    float     b_;
  };

  GuiDouble                            gui_time_;	 
  GuiDouble                            gui_sample_rate_; 
  GuiDouble                            gui_sweep_speed_; 
  GuiDouble                            gui_dots_per_inch_;
  GuiDouble                            gui_plot_height_;
  GuiInt                               gui_play_mode_;

  GuiInt                               gui_plot_count_;
  vector<GuiString*>                   gui_nw_label_;
  vector<GuiString*>                   gui_sw_label_;
  vector<GuiString*>                   gui_label_;
  vector<GuiString*>                   gui_min_ref_label_;
  vector<GuiString*>                   gui_max_ref_label_;
  vector<GuiDouble*>                   gui_min_;
  vector<GuiDouble*>                   gui_max_;
  vector<GuiInt*>                      gui_idx_;
  vector<GuiInt*>                      gui_snd_;
  vector<GuiDouble*>                   gui_red_;
  vector<GuiDouble*>                   gui_green_;
  vector<GuiDouble*>                   gui_blue_;

  GLXContext                            ctx_;
  Display*                              dpy_;
  Window                                win_;
  int                                   width_;
  int                                   height_;
  FreeTypeLibrary *	                freetype_lib_;
  map<string, FreeTypeFace *>		fonts_;
  RTDraw *		                runner_;
  Thread *		                runner_thread_;
  //! 0-9 digit textures.
  vector<LabelTex>                      digits_;
  bool                                  dig_init_;
  vector<Plot>                          plots_;
  NrrdDataHandle                        data_;
  NrrdDataHandle                        data2_;
  int                                   cur_idx_;
  bool                                  plots_dirty_;

  bool                  make_current();
  void                  synch_plot_vars(int s);
  void                  init_plots();
  void                  draw_plots();
  void                  draw_counter(float x, float y);
  void                  get_places(vector<int> &places, int num) const;
  void			setup_gl_view();
  static unsigned int	pow2(const unsigned int);
  
};


class RTDraw : public Runnable {
public:
  RTDraw(ICUMonitor* module) : 
    module_(module), 
    throttle_(), 
    dead_(0) 
  {};
  virtual ~RTDraw();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  ICUMonitor *	module_;
  TimeThrottle	throttle_;
  bool		dead_;
};

RTDraw::~RTDraw()
{
}

void
RTDraw::run()
{
  throttle_.start();
  const double inc = 1./75.;
  double t = throttle_.time();
  double tlast = t;
  while (!dead_) {
    t = throttle_.time();
    throttle_.wait_for_time(t + inc);
    double elapsed = t - tlast;
    module_->inc_time(elapsed);
    module_->redraw_all();
    tlast = t;
  }
}

void 
ICUMonitor::LabelTex::draw(float px, float py, float sx, float sy)
{
  glBindTexture(GL_TEXTURE_2D, tex_id_);
  
  float qwidth  = tex_width_  * u_ * sx;
  float qheight = tex_height_ * v_ * sy;
  
  glBegin(GL_QUADS);
  
  glTexCoord2f(0.0, 0.0);
  glVertex2f(px * sx, py * sy);
  glTexCoord2f(u_, 0.0);
  glVertex2f(px * sx + qwidth, py * sy);
  glTexCoord2f(u_, v_);
  glVertex2f(px * sx + qwidth, py * sy + qheight);      
  glTexCoord2f(0.0, v_);
  glVertex2f(px * sx, py * sy + qheight);

  glEnd();
  glBindTexture(GL_TEXTURE_2D, 0);
}


void 
ICUMonitor::LabelTex::bind(FreeTypeFace *font) 
{
  BBox bbox;
  FreeTypeText fttext(text_, font);
  fttext.get_bounds(bbox);
  if (bbox.min().y() < 0) {
    fttext.set_position(Point(0, -bbox.min().y(), 0));
    bbox.reset();
    fttext.get_bounds(bbox);
  }
  float w = bbox.max().x() + 1.0;
  float h = bbox.max().y() + 1.0;

  tex_width_  = pow2(Round(w));
  tex_height_ = pow2(Round(h));
  u_ = w / (float)tex_width_;
  v_ = h / (float)tex_height_;

  GLubyte *buf = scinew GLubyte[tex_width_ * tex_height_ * 4];
  memset(buf, 0, tex_width_ * tex_height_ * 4);
  fttext.render(tex_width_, tex_height_, buf);     

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &tex_id_);
  glBindTexture(GL_TEXTURE_2D, tex_id_);
      
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glPixelTransferi(GL_MAP_COLOR, 0);
      
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width_, tex_height_, 
	       0, GL_RGBA, GL_UNSIGNED_BYTE, buf);
      
  delete[] buf;
}


DECLARE_MAKER(ICUMonitor)

ICUMonitor::ICUMonitor(GuiContext* ctx) :
  Module("ICUMonitor", ctx, Filter, "Render", "VS"),
  gui_time_(ctx->subVar("time")),
  gui_sample_rate_(ctx->subVar("sample_rate")),
  gui_sweep_speed_(ctx->subVar("sweep_speed")),
  gui_dots_per_inch_(ctx->subVar("dots_per_inch")),
  gui_plot_height_(ctx->subVar("plot_height")),
  gui_play_mode_(ctx->subVar("play_mode")),
  gui_plot_count_(ctx->subVar("plot_count")),
  ctx_(0),
  dpy_(0),
  win_(0),
  freetype_lib_(0),
  fonts_(),
  runner_(0),
  runner_thread_(0),
  digits_(10),
  dig_init_(false),
  plots_(0),
  data_(0),
  data2_(0),
  cur_idx_(0),
  plots_dirty_(true)
{
  try {
    freetype_lib_ = scinew FreeTypeLibrary();
  } catch (...) {
    error("Cannot Initialize FreeType Library.");
    error("Did you configure with --with-freetype= ?");
    error("Module will not render text.");
  }

  if (freetype_lib_) {
    try {
      freetype_lib_ = scinew FreeTypeLibrary();
      string sdir;
      const char *dir = sci_getenv("SCIRUN_FONT_PATH");
      if (dir) 
	sdir = dir;
      else
	sdir = string(sci_getenv("SCIRUN_SRCDIR"))+"/pixmaps";

      fonts_["default"] = freetype_lib_->load_face(sdir+"/scirun.ttf");
      fonts_["anatomical"] = fonts_["default"];
      fonts_["patientname"] = fonts_["default"];
      fonts_["fps"] = fonts_["default"];
      fonts_["view"] = fonts_["default"];
      fonts_["position"] = fonts_["default"];
      
    } catch (...) {
      fonts_.clear();
      error("Error loading fonts.\n"
	    "Please set SCIRUN_FONT_PATH to a directory with scirun.ttf\n");
    }
  }

}

ICUMonitor::~ICUMonitor()
{
  if (runner_thread_) {
    runner_->set_dead(true);
    runner_thread_->join();
    runner_thread_ = 0;
  }
}

void 
ICUMonitor::inc_time(double elapsed)
{
  gui_sample_rate_.reset();
  gui_play_mode_.reset();

  float samp_rate = gui_sample_rate_.get();  // samples per second.
  if (! data_.get_rep() || ! gui_play_mode_.get()) return;
  int samples = (int)round(samp_rate * elapsed);
  cur_idx_ += samples;
  if (cur_idx_ > data_->nrrd->axis[1].size) {
    cur_idx_ = 0;
  }

  gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
  gui_time_.reset();
  gui->execute("update idletasks");
}

bool
ICUMonitor::make_current()
{
  //----------------------------------------------------------------
  // obtain rendering ctx 
  if(!ctx_) {
    const string myname(".ui" + id + ".f.gl.gl");
    Tk_Window tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
                                      Tk_MainWindow(the_interp));
    if(!tkwin) {
      warning("Unable to locate window!");
      gui->unlock();
      return false;
    }
    dpy_ = Tk_Display(tkwin);
    win_ = Tk_WindowId(tkwin);
    ctx_ = OpenGLGetContext(the_interp, ccast_unsafe(myname));
    width_ = Tk_Width(tkwin);
    height_ = Tk_Height(tkwin);
    // check if it was created
    if(!ctx_) {
      error("Unable to obtain OpenGL context!");
      gui->unlock();
      return false;
    }
    glXMakeCurrent(dpy_, win_, ctx_);
  } else {
    glXMakeCurrent(dpy_, win_, ctx_);
  } 
  return true;
}

// TODO: Query opengl max texture size
unsigned int
ICUMonitor::pow2(const unsigned int dim) {
  unsigned int val = 1;
  while (val < dim) { val = val << 1; };
  return val;
}

void 
ICUMonitor::draw_counter(float x, float y) 
{
  if (! dig_init_) {
    FreeTypeFace *font = fonts_["anatomical"];
    font->set_points(66.0);
    digits_.clear();
    digits_.resize(10);
    for (int i = 0; i < 10; i++)
    {
      ostringstream str;
      str << i;
      LabelTex dt(str.str());
      dt.bind(font);
      digits_[i] = dt;
    }

    dig_init_ = true;
  }

  const float sx = 1.0 / width_;
  const float sy = 1.0 / height_;

  // have the texture of digits
  glEnable(GL_TEXTURE_2D);

  vector<int> places(3);
  for (int i = 0; i < 100; i++) 
  {
    float px = x;
    float py = y;
    //cerr << "drawing : " << i << endl;
    get_places(places, i);
    for (int place = places.size() - 1; place >= 0; place--) 
    {
      //cerr << "place=" << place << endl;
      //cerr << "places[place]=" << places[place] << endl;
      LabelTex dt = digits_[places[place]];

      dt.draw(px, py, sx, sy);
      px += dt.u_ * dt.tex_width_ ;
     }

  }
}



template <class T> struct del_list : public unary_function<T, void>
{
  void operator() (T x) { delete x; }
};

template <class T>
void clear_vector(T &vec, int s)
{
  for_each(vec.begin(), vec.end(), 
	   del_list<typename T::value_type>());
  vec.clear();
  vec.resize(s);
}

void 
ICUMonitor::synch_plot_vars(int s)
{
  plots_.clear();
  plots_.resize(s);
  clear_vector(gui_nw_label_, s);
  clear_vector(gui_sw_label_, s);
  clear_vector(gui_label_, s);
  clear_vector(gui_min_ref_label_, s);
  clear_vector(gui_max_ref_label_, s);
  clear_vector(gui_min_, s);
  clear_vector(gui_max_, s);
  clear_vector(gui_idx_, s);
  clear_vector(gui_snd_, s);
  clear_vector(gui_red_, s);
  clear_vector(gui_green_, s);
  clear_vector(gui_blue_, s);
 
}



void 
ICUMonitor::init_plots()
{
  if (! plots_dirty_) return;
  plots_dirty_ = false;
  reset_vars();
  synch_plot_vars(gui_plot_count_.get());
  if (! gui_plot_count_.get()) return;

  FreeTypeFace *font = fonts_["anatomical"];
  if (! font) return;

  int i = 0;
  vector<Plot>::iterator iter = plots_.begin();
  while (iter != plots_.end()) {
    const string num = to_string(i);
    Plot &g = *iter++;
    font->set_points(12.0);
    if (! gui_nw_label_[i]) {
      gui_nw_label_[i] = scinew GuiString(ctx->subVar("nw_label-" + num));
    }
    if (gui_nw_label_[i]->get() != string("")) {
      if (g.nw_label_) delete g.nw_label_;
      g.nw_label_ = scinew LabelTex(gui_nw_label_[i]->get());
      g.nw_label_->bind(font);
    }

    if (! gui_sw_label_[i]) {
      gui_sw_label_[i] = scinew GuiString(ctx->subVar("sw_label-" + num));
    }
    if (gui_sw_label_[i]->get() != string("")) {
      if (g.sw_label_) delete g.sw_label_;
      g.sw_label_ = scinew LabelTex(gui_sw_label_[i]->get());
      g.sw_label_->bind(font);
    }

    font->set_points(18.0);
    if (! gui_label_[i]) {
      gui_label_[i] = scinew GuiString(ctx->subVar("label-" + num));
    }
    if (gui_label_[i]->get() != string("")) {
      if (g.label_) delete g.label_;
      g.label_ = scinew LabelTex(gui_label_[i]->get());
      g.label_->bind(font);
    }

    font->set_points(14.0);
    if (! gui_min_ref_label_[i]) {
      gui_min_ref_label_[i] = scinew GuiString(
					ctx->subVar("min_ref_label-" + num));
    }
    if (gui_min_ref_label_[i]->get() != string("")) {
      if (g.min_ref_label_) delete g.min_ref_label_;
      g.min_ref_label_ = scinew LabelTex(gui_min_ref_label_[i]->get());
      g.min_ref_label_->bind(font);
    }
    if (! gui_max_ref_label_[i]) {
      gui_max_ref_label_[i] = scinew GuiString(
					 ctx->subVar("max_ref_label-" + num));
    }
    if (gui_max_ref_label_[i]->get() != string("")) {
      if (g.max_ref_label_) delete g.max_ref_label_;
      g.max_ref_label_ = scinew LabelTex(gui_max_ref_label_[i]->get());
      g.max_ref_label_->bind(font);
    }
      
    if (! gui_min_[i]) {
      gui_min_[i] = scinew GuiDouble(ctx->subVar("min-" + num));
    }
    g.min_ = gui_min_[i]->get();
    if (! gui_max_[i]) {
      gui_max_[i] = scinew GuiDouble(ctx->subVar("max-" + num));
    }
    g.max_ = gui_max_[i]->get();
    if (! gui_idx_[i]) {
      gui_idx_[i] = scinew GuiInt(ctx->subVar("idx-" + num));
    }
    g.index_ = gui_idx_[i]->get();
    if (! gui_snd_[i]) {
      gui_snd_[i] = scinew GuiInt(ctx->subVar("snd-" + num));
    }
    g.snd_ = gui_snd_[i]->get();

    if (! gui_red_[i]) {
      gui_red_[i] = scinew GuiDouble(ctx->subVar("plot_color-" + num + "-r"));
    }
    g.r_ = gui_red_[i]->get();

    if (! gui_green_[i]) {
      gui_green_[i]= scinew GuiDouble(ctx->subVar("plot_color-" + num + "-g"));
    }
    g.g_ = gui_green_[i]->get();

    if (! gui_blue_[i]) {
      gui_blue_[i] = scinew GuiDouble(ctx->subVar("plot_color-" + num + "-b"));
    }
    g.b_ = gui_blue_[i]->get();
    ++i;
  }

}

void 
ICUMonitor::draw_plots()
{
  reset_vars();
  const int gr_ht = (int)gui_plot_height_.get();

  CHECK_OPENGL_ERROR("start draw_plots")

  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  const float w = width_;
  const float h = height_;
  const float sx = 1.0 / w;
  const float sy = 1.0 / h;

  float cur_x = 20;
  float cur_y = h - 30;
  

  glLineWidth(1.0);
  glLineStipple(3, 0x1001);
  vector<Plot>::iterator iter = plots_.begin();
  while (iter != plots_.end())
  {
    Plot &g = *iter++;
    glColor4f(g.r_, g.g_, g.b_, 1.0);

    if (g.nw_label_) { 
      float xoff = g.nw_label_->tex_width_ * g.nw_label_->u_;
      g.nw_label_->draw(cur_x - xoff, cur_y, sx, sy);
    }    
    if (g.sw_label_) { 
      float xoff = g.sw_label_->tex_width_ * g.sw_label_->u_;
      g.sw_label_->draw(cur_x - xoff, cur_y - gr_ht * 0.5, sx, sy);
    }
    if (g.min_ref_label_) { 
      float xoff = g.min_ref_label_->tex_width_ * g.min_ref_label_->u_ + 3;
      float yoff = g.min_ref_label_->tex_height_ * g.min_ref_label_->v_ * 0.5;
      g.min_ref_label_->draw(cur_x, cur_y - gr_ht - yoff, sx, sy);
      glDisable(GL_TEXTURE_2D);
      glBegin(GL_LINES);
      glVertex2f((cur_x + xoff) * sx, (cur_y - gr_ht) * sy);
      glVertex2f((cur_x + (w * .70)) * sx, (cur_y - gr_ht) * sy);
      glEnd();
      glEnable(GL_TEXTURE_2D);
    }    
    if (g.max_ref_label_) { 
      float xoff = g.max_ref_label_->tex_width_ * g.max_ref_label_->u_ + 3;
      float yoff = g.max_ref_label_->tex_height_ * g.max_ref_label_->v_ * 0.5;
      g.max_ref_label_->draw(cur_x, cur_y - yoff, sx, sy);
      glDisable(GL_TEXTURE_2D);
      glBegin(GL_LINES);
      glVertex2f((cur_x + xoff) * sx, cur_y * sy);
      glVertex2f((cur_x + (w * .70)) * sx, cur_y * sy);
      glEnd();
      glEnable(GL_TEXTURE_2D);
    }
    if (g.label_) { 
      g.label_->draw(cur_x + w * 0.70, cur_y, sx, sy);
    }
    if (data_.get_rep()) {
      const float norm = (float)gr_ht / (g.max_ - g.min_);
      // draw the plot
      //1 millimeters = 0.0393700787 inches
      const float dpi = gui_dots_per_inch_.get();
      const float pixels_per_mm = dpi * 0.0393700787;
      const float sweep_speed = gui_sweep_speed_.get() * pixels_per_mm; 
      const float samp_rate = gui_sample_rate_.get();  // samples per second.
      const float pix_per_sample = sweep_speed / samp_rate;
      const float gwidth = w * .75;  // total width of the drawable area.
      const float samples = gwidth / pix_per_sample;
      float start_y = cur_y - gr_ht;
      //glColor4f(0.0, 0.9, 0.1, 1.0);
      glDisable(GL_TEXTURE_2D);

      glBegin(GL_LINE_STRIP);
      for (int i = 0; i < (int)samples; i++) {
	int idx = i + cur_idx_;
	if (idx > data_->nrrd->axis[1].size) {
	  idx -= data_->nrrd->axis[1].size;
	}
	float *dat = (float*)data_->nrrd->data;
	int dat_index = idx * data_->nrrd->axis[0].size + g.index_;
	float val = (dat[dat_index] - g.min_) * norm;
	glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, (start_y + val) * sy);

	if (idx % (int)samp_rate == 0){
	  float tick = gr_ht * .15;// * norm;
	  //glColor4f(0.0, 0.1, 0.9, 1.0);
	  glColor4f(1.0, 1.0, 1.0, 1.0);
	  glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, 
		     (start_y + val + tick) * sy);
	  glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, 
		     (start_y + val - tick) * sy);
	  glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, 
		     (start_y + val) * sy);
	  glColor4f(g.r_, g.g_, g.b_, 1.0);
	}
      }
      glEnd();

      if (g.snd_ == 1 && data2_.get_rep()) {
         glBegin(GL_LINE_STRIP);
         for (int i = 0; i < (int)samples; i++) {
            int idx = i + cur_idx_;
            if (idx > data2_->nrrd->axis[1].size) {
               idx -= data2_->nrrd->axis[1].size;
            }
            float *dat = (float*)data2_->nrrd->data;
            int dat_index = idx * data2_->nrrd->axis[0].size + g.index_;
            float val = (dat[dat_index] - g.min_) * norm;
            glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, (start_y + val) * sy);

            glColor4f(0.5, 0.5, 0.5, 0.7);
         }
         glEnd();
      }

      glEnable(GL_TEXTURE_2D);     
    }

    cur_y -= gr_ht + 20;
  }

  CHECK_OPENGL_ERROR("end draw_plots")
}


void 
ICUMonitor::get_places(vector<int> &places, int num) const
{
  places.clear();
  if (num == 0) {
    places.push_back(0);
    return;
  }
  while (num > 0) 
  {
    places.push_back(num % 10);
    num = num / 10;
  }
}

void
ICUMonitor::redraw_all()
{
  gui->lock();
  if (! make_current()) return;

  init_plots();

  glDrawBuffer(GL_BACK);

  setup_gl_view();

  FreeTypeFace *font = fonts_["anatomical"];
  if (! font) return;

  draw_plots();

  glXSwapBuffers(dpy_, win_);
  glXMakeCurrent(dpy_, 0, 0);
  gui->unlock();
}

void
ICUMonitor::setup_gl_view()
{
  glViewport(0, 0, width_, height_);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);

  glClearColor(0.0, .25, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);

}


void
ICUMonitor::execute()
{
  NrrdIPort *nrrd1_port = (NrrdIPort*)get_iport("Nrrd1");

  if (!nrrd1_port) 
  {
    error("Unable to initialize iport Nrrd1.");
    return;
  }

  nrrd1_port->get(data_);

  if (!data_.get_rep())
  {
    error ("Unable to get input data.");
    return;
  } 
  
  NrrdIPort *nrrd2_port = (NrrdIPort*)get_iport("Nrrd2");

  if (!nrrd2_port) 
  {
    error("Unable to initialize iport Nrrd2.");
    return;
  }

  nrrd2_port->get(data2_);

  //if (!data2_.get_rep())
  //{
   // error ("Unable to get input data.");
    //return;
  //} 
  
  if (!runner_) {
    runner_ = scinew RTDraw(this);
    runner_thread_ = scinew Thread(runner_, string(id+" RTDraw OpenGL").c_str());
  }
}

void
ICUMonitor::tcl_command(GuiArgs& args, void* userdata) 
{
  if(args.count() < 2) {
    args.error("ICUMonitor needs a minor command");
    return;
  } else if(args[1] == "time") {
    gui_time_.reset();
    if (data_.get_rep()) {
      cur_idx_ = (int)round(gui_time_.get() * data_->nrrd->axis[1].size);
    }
  } else if(args[1] == "expose") {
    redraw_all();
  } else if(args[1] == "redraw") {
    redraw_all();
  } else if(args[1] == "init") {
    plots_dirty_ = true;
    //    init_plots();
  } else {
    Module::tcl_command(args, userdata);
  }
}



} // End namespace VS


