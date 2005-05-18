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
//    Author : Martin Cole, McKay Davis, Alex Ade
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
#include <Dataflow/Ports/TimePort.h>

#include <Core/Geom/FreeType.h>

#include <Core/Util/Environment.h>

#include <typeinfo>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

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
    void set (string s);
    
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
      aux_data_(0),
      aux_data_label_(0),
      previous_(0),
      index_(-1),
      snd_(0),
      clamp_(0),
      lines_(0),
      draw_aux_data_(0),
      use_plot_color_(0),
      auxindex_(-1),
      min_(0.0),
      max_(1.0),
      r_(1.0),
      g_(1.0),
      b_(1.0),
      init_(false)
    {}
    
    LabelTex *nw_label_;
    LabelTex *sw_label_;
    LabelTex *min_ref_label_;
    LabelTex *max_ref_label_;
    LabelTex *label_;
    LabelTex *aux_data_;
    LabelTex *aux_data_label_;
    int       previous_;
    int       index_;
    int       snd_;
    int       clamp_;
    int       lines_;
    int       draw_aux_data_;
    int       use_plot_color_;
    int       auxindex_;
    float     min_;
    float     max_;
    float     r_;
    float     g_;
    float     b_;
    bool      init_;
  };

  GuiDouble                            gui_time_;	 
  GuiDouble                            gui_sample_rate_; 
  GuiDouble                            gui_sweep_speed_; 
  GuiDouble                            gui_dots_per_inch_;
  GuiDouble                            gui_plot_height_;
  GuiInt                               gui_play_mode_;
  GuiInt                               gui_dump_frames_;
  GuiInt                               gui_time_markers_mode_;
  GuiInt                               gui_selected_marker_;
  GuiInt                               gui_injury_offset_;
  GuiDouble			       gui_top_margin_;
  GuiDouble			       gui_left_margin_;
  GuiDouble			       gui_plot_spacing_;
  GuiDouble			       gui_font_scale_;
  GuiInt			       gui_show_name_;
  GuiInt			       gui_show_date_;
  GuiInt			       gui_show_time_;
  GuiInt                               gui_plot_count_;
  GuiInt                               gui_geom_;
  GuiDouble										gui_2ndred_;
  GuiDouble										gui_2ndgreen_;
  GuiDouble										gui_2ndblue_;

  vector<GuiString*>                   gui_nw_label_;
  vector<GuiString*>                   gui_sw_label_;
  vector<GuiString*>                   gui_label_;
  vector<GuiString*>                   gui_aux_data_label_;
  vector<GuiString*>                   gui_min_ref_label_;
  vector<GuiString*>                   gui_max_ref_label_;
  vector<GuiDouble*>                   gui_min_;
  vector<GuiDouble*>                   gui_max_;
  vector<GuiInt*>                      gui_idx_;
  vector<GuiInt*>                      gui_snd_;
  vector<GuiInt*>                      gui_clamp_;
  vector<GuiInt*>                      gui_lines_;
  vector<GuiInt*>                      gui_draw_aux_data_;
  vector<GuiInt*>                      gui_use_plot_color_;
  vector<GuiInt*>                      gui_auxidx_;
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
  vector<int>                           markers_;
  int                                   cur_idx_;
  bool                                  plots_dirty_;
  unsigned int                          frame_count_;
  double                                last_global_time_;
  double                                elapsed_since_global_change_;
  double                                time_sf_;
  LabelTex 			       *name_label;
  string 				name_text;
  //int 				injury_offset_;
  LabelTex			       *time_label;
  string 				time_text;
  LabelTex 			       *date_label;
  string 				date_text;
  TimeViewerHandle                      time_viewer_h_;


  bool                  make_current();
  void                  synch_plot_vars(int s);
  void                  init_plots();
  void                  draw_plots();
  void                  draw_counter(float x, float y);
  void                  get_places(vector<int> &places, int num) const;
  void			setup_gl_view();
  static unsigned int	pow2(const unsigned int);
  void 			setTimeLabel();
  void 			addMarkersToMenu();
  void			setConfigFromData();
  void 			setNameAndDateAndTime();
  void                  save_image(int x, int y,const string& fname,
				   const string &ftype);
};


class RTDraw : public Runnable {
public:
  RTDraw(ICUMonitor* module, TimeViewerHandle tvh) : 
    module_(module), 
    throttle_(), 
    tvh_(tvh),
    dead_(0),
    lock_("RTDraw mutex")
  {};
  virtual ~RTDraw();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
  void lock() { lock_.lock(); }
  void unlock() { lock_.unlock(); }
private:
  ICUMonitor            *module_;
  TimeThrottle	         throttle_;
  TimeViewerHandle       tvh_;
  bool		         dead_;
  Mutex                  lock_;
};

RTDraw::~RTDraw()
{
}

void
RTDraw::run()
{
  throttle_.start();
  const double inc = 1./20.; // the rate at which we refresh the monitor.
  double t = throttle_.time();
  while (!dead_) {
     t = throttle_.time();
    throttle_.wait_for_time(t + inc);
    lock(); 
    module_->inc_time(tvh_->view_elapsed_since_start());
    module_->redraw_all();
    unlock();
  }
}

void 
ICUMonitor::LabelTex::set(string s)
{
  text_.replace(0, text_.length(), s);
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

  GLubyte *buf = scinew GLubyte[tex_width_ * tex_height_];
  memset(buf, 0, tex_width_ * tex_height_);
  fttext.render(tex_width_, tex_height_, buf);     

  if (glIsTexture(tex_id_))
    glDeleteTextures(1, &tex_id_);

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
      
  glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, tex_width_, tex_height_, 
	       0, GL_ALPHA, GL_UNSIGNED_BYTE, buf);
      
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
  gui_dump_frames_(ctx->subVar("dump_frames")),
  gui_time_markers_mode_(ctx->subVar("time_markers_mode")),
  gui_selected_marker_(ctx->subVar("selected_marker")),
  gui_injury_offset_(ctx->subVar("injury_offset")),
  gui_top_margin_(ctx->subVar("top_margin")),
  gui_left_margin_(ctx->subVar("left_margin")),
  gui_plot_spacing_(ctx->subVar("plot_spacing")),
  gui_font_scale_(ctx->subVar("font_scale")),
  gui_show_name_(ctx->subVar("show_name")),
  gui_show_date_(ctx->subVar("show_date")),
  gui_show_time_(ctx->subVar("show_time")),
  gui_plot_count_(ctx->subVar("plot_count")),
  gui_geom_(ctx->subVar("geom")),
  gui_2ndred_(ctx->subVar("2ndplot_color-r")),
  gui_2ndgreen_(ctx->subVar("2ndplot_color-g")),
  gui_2ndblue_(ctx->subVar("2ndplot_color-b")),
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
  markers_(0),
  cur_idx_(0),
  plots_dirty_(true),
  frame_count_(0),
  name_label(0),
  name_text(" "),
  //injury_offset_(0),
  time_label(0),
  time_text("Time: 00:00:00"),
  date_label(0),
  date_text(" ")
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

// absolute elapsed time.
void 
ICUMonitor::inc_time(double elapsed)
{
  gui_sample_rate_.reset();
  gui_play_mode_.reset();
  gui_time_markers_mode_.reset();

  float samp_rate = gui_sample_rate_.get();  // samples per second.
  if (! data_.get_rep() || ! gui_play_mode_.get()) return;
  int samples = (int)round(samp_rate * elapsed);
  cur_idx_ = samples;
  if (cur_idx_ > data_->nrrd->axis[1].size - 1) {
    cur_idx_ = data_->nrrd->axis[1].size - 1;
  }

  gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
  gui_time_.reset();
  gui->execute("update idletasks");

  setTimeLabel();
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
  clear_vector(gui_aux_data_label_, s);
  clear_vector(gui_min_ref_label_, s);
  clear_vector(gui_max_ref_label_, s);
  clear_vector(gui_min_, s);
  clear_vector(gui_max_, s);
  clear_vector(gui_idx_, s);
  clear_vector(gui_snd_, s);
  clear_vector(gui_clamp_, s);
  clear_vector(gui_lines_, s);
  clear_vector(gui_draw_aux_data_, s);
  clear_vector(gui_use_plot_color_, s);
  clear_vector(gui_auxidx_, s);
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

  font->set_points(14.0 * gui_font_scale_.get());
  if (name_label) delete name_label;
  name_label = scinew LabelTex(name_text);
  name_label->bind(font);

  if (date_label) delete date_label;
  date_label = scinew LabelTex(date_text);
  date_label->bind(font);

  if (time_label) delete time_label;
  time_label = scinew LabelTex(time_text);
  time_label->bind(font);

  int i = 0;
  vector<Plot>::iterator iter = plots_.begin();
  while (iter != plots_.end()) {
    const string num = to_string(i);
    Plot &g = *iter++;

    font->set_points(36.0);

    if (g.aux_data_) delete g.aux_data_;
      g.aux_data_ = scinew LabelTex(" ");
      g.aux_data_->bind(font);

    font->set_points(12.0 * gui_font_scale_.get());

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

    font->set_points(14.0 * gui_font_scale_.get());

    if (! gui_label_[i]) {
      gui_label_[i] = scinew GuiString(ctx->subVar("label-" + num));
    }
    if (gui_label_[i]->get() != string("")) {
      if (g.label_) delete g.label_;
      g.label_ = scinew LabelTex(gui_label_[i]->get());
      g.label_->bind(font);
    }
    if (! gui_aux_data_label_[i]) {
      gui_aux_data_label_[i] = scinew GuiString(ctx->subVar("aux_data_label-" + num));
    }
    if (gui_aux_data_label_[i]->get() != string("")) {
      if (g.aux_data_label_) delete g.aux_data_label_;
      g.aux_data_label_ = scinew LabelTex(gui_aux_data_label_[i]->get());
      g.aux_data_label_->bind(font);
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
    if (! gui_clamp_[i]) {
      gui_clamp_[i] = scinew GuiInt(ctx->subVar("clamp-" + num));
    }
    g.clamp_ = gui_clamp_[i]->get();
    if (! gui_lines_[i]) {
      gui_lines_[i] = scinew GuiInt(ctx->subVar("lines-" + num));
    }
    g.lines_ = gui_lines_[i]->get();
    if (! gui_auxidx_[i]) {
      gui_auxidx_[i] = scinew GuiInt(ctx->subVar("auxidx-" + num));
    }
    g.auxindex_ = gui_auxidx_[i]->get();
    if (! gui_draw_aux_data_[i]) {
      gui_draw_aux_data_[i] = scinew GuiInt(ctx->subVar("draw_aux_data-" + num));
    }
    g.draw_aux_data_ = gui_draw_aux_data_[i]->get();
    if (! gui_use_plot_color_[i]) {
      gui_use_plot_color_[i] = scinew GuiInt(ctx->subVar("use_plot_color-" + num));
    }
    g.use_plot_color_ = gui_use_plot_color_[i]->get();

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

} // end ICUMonitor::init_plots()


void 
ICUMonitor::draw_plots()
{
  reset_vars();
  const int gr_ht = (int)gui_plot_height_.get();

  CHECK_OPENGL_ERROR();
  
  glDrawBuffer(GL_BACK);
  //glClearColor(1.0, 1.0, 1.0, 1.0);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  const float w = width_;
  const float h = height_;
  const float sx = 1.0 / w;
  const float sy = 1.0 / h;
  const int gp = 3;
  const float cw = 0.70;
  const int cg = 25;
  float cur_x = gui_left_margin_.get();
  float cur_y = h - gui_top_margin_.get();
  float yoff(0.0);

  glEnable(GL_LINE_SMOOTH);
  glLineWidth(1.0);
  // type of stippling (dashed lines) - use  0xFFFF for solid line
  glLineStipple(1, 0x0F0F);
  if (name_label && gui_show_name_.get()) { 
    //float yoff = name_label->tex_height_ * name_label->v_ * 1.5;
    yoff += name_label->tex_height_ * name_label->v_ * 1.5;
    glColor4f(1.0, 1.0, 1.0, 1.0);
    name_label->draw(cur_x, h - yoff, sx, sy);
  }
  if (date_label && gui_show_date_.get()) { 
    //float yoff = date_label->tex_height_ * date_label->v_ * 1.5;
    yoff += date_label->tex_height_ * date_label->v_ * 1.5;
    //float xoff = 0;
    //if (name_label && gui_show_name_.get()) { 
    // xoff = name_label->tex_width_ * name_label->u_ + cg;
    //}
    //date_label->draw(cur_x + xoff, h - yoff, sx, sy);
    //float xoff = date_label->tex_width_ * date_label->u_;
    //g.label_->draw(cur_x + w * cw, cur_y, sx, sy);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    //date_label->draw((cur_x + (w*cw)) - xoff, h - yoff, sx, sy);
    date_label->draw(cur_x, h - yoff, sx, sy);
  }
  if (time_label && gui_show_time_.get()) { 
    FreeTypeFace *font = fonts_["anatomical"];
    font->set_points(14.0 * gui_font_scale_.get());
    time_label->set(time_text);
    time_label->bind(font);

    float yoff = time_label->tex_height_ * time_label->v_ * 1.5;
    float xoff = time_label->tex_width_ * time_label->u_;

    glColor4f(1.0, 1.0, 1.0, 1.0);
    time_label->draw((cur_x + (w*cw)) - xoff, h - yoff, sx, sy);
  }

  vector<Plot>::iterator iter = plots_.begin();
  while (iter != plots_.end())
  {
    Plot &g = *iter++;

    glColor4f(g.r_, g.g_, g.b_, 1.0);

    if (g.nw_label_) { 
      float xoff = g.nw_label_->tex_width_ * g.nw_label_->u_ + gp;
      g.nw_label_->draw(cur_x - xoff, cur_y, sx, sy);
    }    
    if (g.sw_label_) { 
      float xoff = g.sw_label_->tex_width_ * g.sw_label_->u_ + gp;
      g.sw_label_->draw(cur_x - xoff, cur_y - gr_ht * 0.5, sx, sy);
    }
    if (g.min_ref_label_) { 
      float xoff = g.min_ref_label_->tex_width_ * g.min_ref_label_->u_ + gp;
      float yoff = g.min_ref_label_->tex_height_ * g.min_ref_label_->v_ * 0.5;
      //g.min_ref_label_->draw(cur_x, cur_y - gr_ht - yoff, sx, sy);
      g.min_ref_label_->draw(cur_x - xoff, cur_y - gr_ht - yoff, sx, sy);
      if (g.lines_ == 1) {
        glDisable(GL_TEXTURE_2D);
	// added line for stippling
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        //glVertex2f((cur_x + xoff) * sx, (cur_y - gr_ht) * sy);
        glVertex2f(cur_x * sx, (cur_y - gr_ht) * sy);
        glVertex2f((cur_x + (w * cw)) * sx, (cur_y - gr_ht) * sy);
        glEnd();
	// added line for stippling
        glDisable(GL_LINE_STIPPLE);
	//
        glEnable(GL_TEXTURE_2D);
      }
    }    
    if (g.max_ref_label_) { 
      float xoff = g.max_ref_label_->tex_width_ * g.max_ref_label_->u_ + gp;
      float yoff = g.max_ref_label_->tex_height_ * g.max_ref_label_->v_ * 0.5;
      //g.max_ref_label_->draw(cur_x, cur_y - yoff, sx, sy);
      g.max_ref_label_->draw(cur_x - xoff, cur_y - yoff, sx, sy);
      if (g.lines_ == 1) {
        xoff = 0;
        if (g.label_) { 
          xoff = g.label_->tex_width_ * g.label_->u_ + gp;
        }
        glDisable(GL_TEXTURE_2D);
	// added line for stippling
        glEnable(GL_LINE_STIPPLE);
	//
        glBegin(GL_LINES);
        //glVertex2f((cur_x + xoff) * sx, cur_y * sy);
        glVertex2f(cur_x * sx, cur_y * sy);
        glVertex2f((cur_x + (w * cw) - xoff) * sx, cur_y * sy);
        glEnd();
	// added line for stippling
        glDisable(GL_LINE_STIPPLE);
	//
        glEnable(GL_TEXTURE_2D);
      }
    }
    if (g.label_) { 
      float xoff = g.label_->tex_width_ * g.label_->u_;
      //g.label_->draw(cur_x + w * cw, cur_y, sx, sy);
      g.label_->draw((cur_x + (w*cw)) - xoff, cur_y, sx, sy);
    }

    if (g.use_plot_color_ == 1)
      glColor4f(g.b_, g.r_, g.g_, 1.0);
    else
      glColor4f(1.0, 1.0, 1.0, 1.0);

    if (g.draw_aux_data_ == 1) { 
      if (g.aux_data_label_)
        g.aux_data_label_->draw(cur_x + w * cw + cg, cur_y, sx, sy);

      if (data_.get_rep()) {
	int idx = cur_idx_;
	if (idx > data_->nrrd->axis[1].size) {
	  idx -= data_->nrrd->axis[1].size;
	}
	float *dat = (float *)data_->nrrd->data;
	int dat_index = idx * data_->nrrd->axis[0].size + g.auxindex_;
         
	int val = (int)dat[dat_index];

	ostringstream auxstr;
	auxstr << val;
	ostringstream prevstr;
	prevstr << g.previous_;

	FreeTypeFace *font = fonts_["anatomical"];
	font->set_points(50.0);
	if (val == -1)
	  g.aux_data_->set(prevstr.str());
	else {
	  g.aux_data_->set(auxstr.str());
	  g.previous_ = val;
	}

	g.aux_data_->bind(font);

	float yoff = g.aux_data_->tex_height_ * g.aux_data_->v_;
	g.aux_data_->draw(cur_x + w * cw + cg + 15, cur_y - yoff*1.25, sx, sy);
      }
    }
    glColor4f(g.r_, g.g_, g.b_, 1.0);
    if (data_.get_rep()) {
      // draw the plot
      const float norm = (float)gr_ht / (g.max_ - g.min_);
      //1 millimeters = 0.0393700787 inches
      const float dpi = gui_dots_per_inch_.get();
      const float pixels_per_mm = dpi * 0.0393700787;
      const float sweep_speed = gui_sweep_speed_.get() * pixels_per_mm; 
      const float samp_rate = gui_sample_rate_.get();  // samples per second.
      const float pix_per_sample = sweep_speed / samp_rate;
      const float gwidth = w * cw;  // total width of the drawable area.
      const float samples = gwidth / pix_per_sample;
      float start_y = cur_y - gr_ht;
      //glColor4f(0.0, 0.9, 0.1, 1.0);
      glDisable(GL_TEXTURE_2D);

      glLineWidth(1.5);
      glBegin(GL_LINE_STRIP);
      for (int i = 0; i < (int)samples; i++) {
	bool last_tick = false;
	static bool wrapping = false;
	int idx = i + cur_idx_;
	if (idx > data_->nrrd->axis[1].size - 1) {
	  idx -= data_->nrrd->axis[1].size - 1;
	   last_tick = !wrapping;
	   wrapping = true;
	} else {
	  wrapping = false;
	}
	
	float *dat = (float*)data_->nrrd->data;
	int dat_index = idx * data_->nrrd->axis[0].size + g.index_;
        float tmpdat = dat[dat_index];
        if (g.clamp_) {
          if (tmpdat > g.max_) tmpdat = g.max_;
          if (tmpdat < g.min_) tmpdat = g.min_;
        }
	//float val = (dat[dat_index] - g.min_) * norm;
        float val = (tmpdat - g.min_) * norm;

	glVertex2f((cur_x + (i * pix_per_sample)) * sx, (start_y + val) * sy);

	if (idx % (int)samp_rate == 0 || last_tick){
	  float tick = gr_ht * .15;// * norm;
	  if (gui_time_markers_mode_.get()) {
	    glColor4f(1.0, 1.0, 0.0, 1.0);
	    glVertex2f((cur_x + (i * pix_per_sample)) * sx, 
		       (start_y + val + tick) * sy);
	    glVertex2f((cur_x + (i * pix_per_sample)) * sx, 
		       (start_y + val - tick) * sy);
	    glVertex2f((cur_x + (i * pix_per_sample)) * sx, 
		       (start_y + val) * sy);
	  }
	  glColor4f(g.r_, g.g_, g.b_, 1.0);
	}
      }
      glEnd();

      if (data2_.get_rep() && g.snd_ == 1) {
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < (int)samples; i++) {
          int idx = i + cur_idx_;
          if (idx > data2_->nrrd->axis[1].size - 1) {
	    idx -= data2_->nrrd->axis[1].size - 1; 
          }
          float *dat = (float*)data2_->nrrd->data;
          int dat_index = idx * data2_->nrrd->axis[0].size + g.index_;
          float tmpdata2 = dat[dat_index];
          if (g.clamp_) {
            if (tmpdata2 > g.max_) tmpdata2 = g.max_;
            if (tmpdata2 < g.min_) tmpdata2 = g.min_;
          }
          //float val = (dat[dat_index] - g.min_) * norm;
          float val = (tmpdata2 - g.min_) * norm;
          glVertex2f((cur_x + (i * pix_per_sample)) * sx, 
		     (start_y + val) * sy);
          //glColor4f(.8, .8, .8, 0.8);
	  glColor4f(gui_2ndred_.get(), gui_2ndgreen_.get(), gui_2ndblue_.get(), 0.8);
        }
        glEnd();
      }
      glLineWidth(1.0);
      glEnable(GL_TEXTURE_2D);     
    }

    cur_y -= gr_ht + gui_plot_spacing_.get();
  }
  gui_dump_frames_.reset();
  if (gui_dump_frames_.get()) {
    ostringstream fname;
    fname << "icu_frame."; 
    fname << setfill('0');
    fname.width(5); 
    fname << frame_count_++ << ".ppm";
    save_image(width_, height_, fname.str(), "ppm");
  }
  CHECK_OPENGL_ERROR();
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
  gui->lock();
  if (! make_current()) return;
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

  gui->unlock();
}

void 
ICUMonitor::setTimeLabel()
{
    int hrs, min, sec;
    bool neg;

    int val = (int)(cur_idx_ / gui_sample_rate_.get());
    //val -= injury_offset_;
    val -= gui_injury_offset_.get();

    hrs = val/(60*60);
    min = (val - hrs*60*60)/60;
    sec = val - hrs*60*60 - min*60;

    neg = (hrs < 0 || min < 0 || sec < 0);

    hrs = Abs(hrs);
    min = Abs(min);
    sec = Abs(sec);

    ostringstream timestr;
    timestr << setfill('0');
    timestr << "Time: ";
    if (neg)
      timestr << "-";
    timestr << setw(2) << hrs << ":";
    timestr << setw(2) << min << ":";
    timestr << setw(2) << sec;
    gui->execute(id + " setTimeLabel {" + timestr.str() + "}");

    time_text.replace(0, time_text.length(), timestr.str());
}

void 
ICUMonitor::addMarkersToMenu()
{
  int value;
  string val;
  hash_map<int, string> tmpmkrs;
  set<int> keys;

  for (unsigned int c = 0; c < data_->nproperties(); c++) {
     string name = data_->get_property_name(c);
     // only look at MKR_<name> values here
     if(string(name, 0, 4) == "MKR_")
     {
       //data_->get_property(name, value);
       data_->get_property(name, val);
 
       stringstream ss(val);
       ss >> value;

       keys.insert(value);
       // strip off "MKR_" from name
       tmpmkrs[value] = string(name, 4, name.size() - 4);
     }
  }

  markers_.clear();
  gui->execute(id + " clearMarkers");

  set<int>::iterator iter;
  for (iter = keys.begin(); iter != keys.end(); iter++) {
      markers_.push_back(*iter);
      gui->execute(id + " setMarkers {" + tmpmkrs[*iter] + "}");
  }
}

void 
ICUMonitor::setConfigFromData()
{
  string value;
  int intValue, idxDigits;
  double floatValue;
  string name;

  // find how many plots we are drawing
  if(! data_->get_property(string("DSPY_plot_count"), value))
  { // no properties -- nothing to do
    return;
  }
  // set gui int plot count
  stringstream plot_ss(value);
  plot_ss >> intValue;
  gui_plot_count_.set(intValue);

  reset_vars();
  synch_plot_vars(gui_plot_count_.get());

  if(intValue < 10) idxDigits = 1;
  else if(intValue < 100) idxDigits = 2;

  // now that the GUI variables have been allocated, populate them
  for (unsigned int c = 0; c < data_->nproperties(); c++) {
     name = data_->get_property_name(c);
     // only look at DSPY_<name> values here
     if(string(name, 0, 5) == "DSPY_")
     {
       data_->get_property(name, value);
       // convert value string to various types
       intValue = atoi(value.c_str());
       floatValue = atof(value.c_str());
       // strip off "DSPY_"
       string ICUvarName = string(name, 5, name.size()-5);

       if(string(ICUvarName, 0, 9) ==  "nw_label-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 9, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_nw_label_[i])
         {
           gui_nw_label_[i] = scinew GuiString(ctx->subVar("nw_label-" + num));
         }
         // set label string value
         gui_nw_label_[i]->set(value);
       }
       if(string(ICUvarName, 0, 9) ==  "sw_label-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 9, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_sw_label_[i])
         {
           gui_sw_label_[i] = scinew GuiString(ctx->subVar("sw_label-" + num));
         }
         // set label string value
         gui_sw_label_[i]->set(value);
       }
       if(string(ICUvarName, 0, 14) ==  "min_ref_label-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 15, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_min_ref_label_[i])
         {
           gui_min_ref_label_[i] = scinew GuiString(
                                         ctx->subVar("min_ref_label-" + num));
         }
         // set label string value
         gui_min_ref_label_[i]->set(value);
       }
       if(string(ICUvarName, 0, 14) ==  "max_ref_label-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 15, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_max_ref_label_[i])
         {
           gui_max_ref_label_[i] = scinew GuiString(
                                         ctx->subVar("max_ref_label-" + num));
         }
         // set label string value
         gui_max_ref_label_[i]->set(value);
       }
       if(string(ICUvarName, 0, 6) ==  "label-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 6, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_label_[i])
         {
           gui_label_[i] = scinew GuiString(ctx->subVar("label-" + num));
         }
         // set label string value
         gui_label_[i]->set(value);
       }
       if(string(ICUvarName, 0, 15) ==  "aux_data_label-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 15, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_aux_data_label_[i])
         {
           gui_aux_data_label_[i] =
             scinew GuiString(ctx->subVar("aux_data_label-" + num));
         }
         // set label string value
         gui_aux_data_label_[i]->set(value);
       }
       if(string(ICUvarName, 0, 4) ==  "min-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 4, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_min_[i])
         {
           gui_min_[i] = scinew GuiDouble(ctx->subVar("min-" + num));
         }
         // set plot min float value
         gui_min_[i]->set(floatValue);
       }
       if(string(ICUvarName, 0, 4) ==  "max-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 4, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_max_[i])
         {
           gui_max_[i] = scinew GuiDouble(ctx->subVar("max-" + num));
         }

         // set plot max float value
         gui_max_[i]->set(floatValue);
       }
       if(string(ICUvarName, 0, 4) ==  "idx-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 4, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_idx_[i])
         {
           gui_idx_[i] = scinew GuiInt(ctx->subVar("idx-" + num));
         }

         // set plot idx int value
         gui_idx_[i]->set(intValue);
       }
       if(string(ICUvarName, 0, 4) ==  "snd-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 4, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_snd_[i])
         {
           gui_snd_[i] = scinew GuiInt(ctx->subVar("snd-" + num));
         }
         // set plot snd int value
         gui_snd_[i]->set(intValue);
       }
       if(string(ICUvarName, 0, 6) ==  "clamp-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 6, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_clamp_[i])
         {
           gui_clamp_[i] = scinew GuiInt(ctx->subVar("clamp-" + num));
         }
         // set plot clamp int value
         gui_clamp_[i]->set(intValue);
       }
       if(string(ICUvarName, 0, 6) ==  "lines-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 6, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_lines_[i])
         {
           gui_lines_[i] = scinew GuiInt(ctx->subVar("lines-" + num));
         }
         // set plot lines int value
         gui_lines_[i]->set(intValue);
       }
       if(string(ICUvarName, 0, 7) ==  "auxidx-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 7, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_auxidx_[i])
         {
           gui_auxidx_[i] = scinew GuiInt(ctx->subVar("auxidx-" + num));
         }
         // set plot auxidx int value
         gui_auxidx_[i]->set(intValue);
       }
       if(string(ICUvarName, 0, 14) ==  "draw_aux_data-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 14, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_draw_aux_data_[i])
         {
           gui_draw_aux_data_[i] =
               scinew GuiInt(ctx->subVar("draw_aux_data-" + num));
         }
         // set plot draw_aux_data int value
         gui_draw_aux_data_[i]->set(intValue);
       }
       // plot_color is of the form plot_color-0-[r|g|b]
       if(string(ICUvarName, 0, 11) ==  "plot_color-")
       {
         // get gui var index
         string labelIndexStr = string(ICUvarName, 11, idxDigits);
         int i;
         i = atoi(labelIndexStr.c_str());
         const string num = to_string(i);
         if (! gui_red_[i])
         {
           gui_red_[i] =
             scinew GuiDouble(ctx->subVar("plot_color-" + num + "-r"));
         }
         if (! gui_green_[i])
         {
           gui_green_[i] =
             scinew GuiDouble(ctx->subVar("plot_color-" + num + "-g"));
         }
         if (! gui_blue_[i])
         {
           gui_blue_[i] =
             scinew GuiDouble(ctx->subVar("plot_color-" + num + "-b"));
         }

         string primaryStr = string(ICUvarName, 12+idxDigits, 1);
         char colorChar;
         colorChar = primaryStr.c_str()[0];
         // set plot plot color float value
         switch(colorChar)
         {
           case 'r':
             gui_red_[i]->set(floatValue);
             break;
           case 'g':
             gui_green_[i]->set(floatValue);
             break;
           case 'b':
             gui_blue_[i]->set(floatValue);
             break;
         } // end switch(colorChar)
       } // end if(string(ICUvarName, 0, 11) ==  "plot_color-")
     } // end if(string(name, 0, 5) == "DSPY_")
  } // end for ( c from  0 to data_->nproperties()-1 )

  // tell the ICUMonitor to redraw
  plots_dirty_ = true;

} // end setConfigFromData()

void 
ICUMonitor::setNameAndDateAndTime()
{
  char *name = nrrdKeyValueGet(data_->nrrd, "name");

  if (name != NULL) {
    string title(name);
    gui->execute(id + " setWindowTitle {Physiology Monitor: " + name + "}");

    ostringstream titlestr;
    titlestr << "Name: " << title;

    name_text.replace(0, name_text.length(), titlestr.str());

    plots_dirty_ = true;
  }

  char *date = nrrdKeyValueGet(data_->nrrd, "date");

  if (date != NULL) {
    string created(date);
    //ostringstream datestr;
    //datestr << "Date: " << created;

    //date_text.replace(0, date_text.length(), datestr.str());
    date_text.replace(0, date_text.length(), created);

    plots_dirty_ = true;
  }

  //char *inj = nrrdKeyValueGet(data_->nrrd, "injury");
  //if (inj != NULL) {
   //  string injoff(inj);
                                                                                
    // stringstream ss(injoff);
     //ss >> injury_offset_;
                                                                                
     //injury_offset_ /= (int)gui_sample_rate_.get();
                                                                                
     //setTimeLabel();
  //}
}

void
ICUMonitor::execute()
{
  TimeIPort *time_port = (TimeIPort*)get_iport("Time");

  if (!time_port) 
  {
    error("Unable to initialize iport Time.");
    return;
  }

  time_port->get(time_viewer_h_);
  if (time_viewer_h_.get_rep() == 0) {
    error("No data in the Time port. It is required.");
    return;
  }

  NrrdIPort *nrrd1_port = (NrrdIPort*)get_iport("Nrrd1");

  if (!nrrd1_port) 
  {
    error("Unable to initialize iport Nrrd1.");
    return;
  }
  if (runner_) runner_->lock();
  nrrd1_port->get(data_);

  if (!data_.get_rep())
  {
    error ("Unable to get input data.");
    return;
  } 

  double rt = data_->nrrd->axis[0].spacing;
  if (rt == rt)
    gui_sample_rate_.set(1/data_->nrrd->axis[0].spacing);

  addMarkersToMenu();

  setNameAndDateAndTime();

  setConfigFromData();

  NrrdIPort *nrrd2_port = (NrrdIPort*)get_iport("Nrrd2");

  if (!nrrd2_port) 
  {
    error("Unable to initialize iport Nrrd2.");
    return;
  }

  nrrd2_port->get(data2_);

  if (runner_) runner_->unlock();

  if (data2_.get_rep() && data2_->nrrd->axis[1].size != 
      data_->nrrd->axis[1].size)
  {
    remark ("Axis 1 size for both NRRD files are not identical.");
  } 
  
  if (!runner_) {
    runner_ = scinew RTDraw(this, time_viewer_h_);
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

    setTimeLabel();

  } else if(args[1] == "expose") {
    redraw_all();
  } else if(args[1] == "redraw") {
    redraw_all();
  } else if(args[1] == "init") {
    plots_dirty_ = true;
    //    init_plots();
  } else if(args[1] == "marker") {
    if (!data_.get_rep()) return;

    int mkr = gui_selected_marker_.get();
    int val = (mkr == -1)?-1:markers_[mkr];

    if (val == -1 || val > data_->nrrd->axis[1].size) return;

    cur_idx_ = val;

    gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
    gui_time_.reset();
    gui->execute("update idletasks");

    setTimeLabel();

  } else if(args[1] == "increment") {
    if (!data_.get_rep()) return;

    cur_idx_ += (int)gui_sample_rate_.get();
    if (cur_idx_ > data_->nrrd->axis[1].size) {
        cur_idx_ = data_->nrrd->axis[1].size;
    }
    gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
    gui_time_.reset();
    gui->execute("update idletasks");

    setTimeLabel();

  } else if(args[1] == "decrement") {
    if (!data_.get_rep()) return;

    cur_idx_ -= (int)gui_sample_rate_.get();
    if (cur_idx_ < 0) {
       cur_idx_ = 0;
    }
    gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
    gui_time_.reset();
    gui->execute("update idletasks");

    setTimeLabel();

  } else if(args[1] == "configure") {
    const string myname(".ui" + id + ".f.gl.gl");
    Tk_Window tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
                                      Tk_MainWindow(the_interp));
    if(!tkwin) {
      warning("Unable to locate window!");
      //gui->unlock();
    } else {
      width_ = Tk_Width(tkwin);
      height_ = Tk_Height(tkwin);
      redraw_all();
    }

  } else {
    Module::tcl_command(args, userdata);
  }
}

// Copied from main viewer OpenGL.cc
void
ICUMonitor::save_image(int x, int y,
		       const string& fname, const string &ftype)
{
#ifndef HAVE_MAGICK
  if (ftype != "ppm" && ftype != "raw")
  {
    cerr << "Error - ImageMagick is not enabled, " 
	 << "can only save .ppm or .raw files.\n";
    return;
  }
#endif

  cerr << "Saving ICU Image: " << fname << " with width=" << x
       << " and height=" << y <<"...\n";


  //gui_->lock();
  // Make sure our GL context is current
  make_current();
 
  // Get Viewport dimensions
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);
  //gui_->unlock();

  ofstream *image_file = NULL;
#ifdef HAVE_MAGICK
  C_Magick::Image *image = NULL;
  C_Magick::ImageInfo *image_info = NULL;
#endif
  int channel_bytes, num_channels;
  bool do_magick;

  if (ftype == "ppm" || ftype == "raw")
  {
    image_file = scinew std::ofstream(fname.c_str());
    channel_bytes = 1;
    num_channels = 3;
    do_magick = false;
    if (ftype == "ppm")
    {
      (*image_file) << "P6" << std::endl;
      (*image_file) << x << " " << y << std::endl;
      (*image_file) << 255 << std::endl;
    }
  }
  else
  {
#ifdef HAVE_MAGICK
    C_Magick::InitializeMagick(0);
    num_channels = 4;
    channel_bytes = 2;
    do_magick = true;
    image_info = C_Magick::CloneImageInfo((C_Magick::ImageInfo *)0);
    strcpy(image_info->filename,fname.c_str());
    image_info->colorspace = C_Magick::RGBColorspace;
    image_info->quality = 90;
    image = C_Magick::AllocateImage(image_info);
    image->columns = x;
    image->rows = y;
#endif
  }

  const int pix_size = channel_bytes*num_channels;

  // Write out a screen height X image width chunk of pixels at a time
  unsigned char* pixels = scinew unsigned char[x * y * pix_size];

  // Start writing image_file
  static unsigned char* tmp_row = 0;
  if (!tmp_row )
    tmp_row = scinew unsigned char[x * pix_size];

  if (do_magick)
  {
#ifdef HAVE_MAGICK
    cerr << "HAVE_MAGICK true" << endl;
    pixels = (unsigned char *)C_Magick::SetImagePixels(image, 0, 0, x, y);
#endif
  }

  if (!pixels)
  {
    cerr << "No ImageMagick Memory! Aborting...\n";
    return;
  }

    
  // render the col and row in the hi_res struct
  //gui_->lock();
    
  // Tell OpenGL where to put the data in our pixel buffer
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  // Set OpenGL back to nice PixelStore values for somebody else
  glPixelStorei(GL_PACK_SKIP_PIXELS,0);
  glPixelStorei(GL_PACK_ROW_LENGTH,0);
    
    
  // Read the data from OpenGL into our memory
  glReadBuffer(GL_FRONT);
  glReadPixels(0, 0, x, y,
	       (num_channels == 3) ? GL_RGB : GL_BGRA,
	       (channel_bytes == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
	       pixels);
  //gui_->unlock();
    
  // OpenGL renders upside-down to image_file writing
  unsigned char *top_row, *bot_row;	
  int top, bot;
  for(top = y - 1, bot = 0; bot < y / 2; top--, bot++)
  {
    top_row = pixels + x * top * pix_size;
    bot_row = pixels + x * bot * pix_size;
    memcpy(tmp_row, top_row, x * pix_size);
    memcpy(top_row, bot_row, x * pix_size);
    memcpy(bot_row, tmp_row, x * pix_size);
  }
  if (do_magick)
  {
#ifdef HAVE_MAGICK
    C_Magick::SyncImagePixels(image);
#endif
  } else
    image_file->write((char *)pixels, x * y * pix_size);

  //gui_->lock();



  if (do_magick)
  {
#ifdef HAVE_MAGICK
    if (!C_Magick::WriteImage(image_info, image))
    {
      cerr << "\nCannont Write " << fname << " because: "
	   << image->exception.reason << std::endl;
    }
	
    C_Magick::DestroyImageInfo(image_info);
    C_Magick::DestroyImage(image);
    C_Magick::DestroyMagick();
#endif
  }
  else
  {
    image_file->close();
    delete[] pixels;
  }

  //gui_->unlock();

  if (tmp_row)
  {
    delete[] tmp_row;
    tmp_row = 0;
  }

}

} // End namespace VS
