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
//    File   : ExecutiveState.cc
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
#include <Core/Math/Trig.h>

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
#include <sstream>
#include <iomanip>

#define AXIS_LENGTH 10

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;


namespace VS {
using namespace SCIRun;
using std::cerr;
using std::endl;

class RTDraw2;

class ExecutiveState : public Module
{
public:
  ExecutiveState(GuiContext* ctx);
  virtual ~ExecutiveState();
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
      lines_(0),
      draw_aux_data(0),
      auxindex_(-1),
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
    LabelTex *aux_data_;
    LabelTex *aux_data_label_;
    int       previous_;
    int       index_;
    int       snd_;
    int       lines_;
    int       draw_aux_data;
    int       auxindex_;
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
  GuiInt                               gui_time_markers_mode_;
  GuiInt                               gui_selected_marker_;

  GuiInt                               gui_plot_count_;
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
  vector<GuiInt*>                      gui_lines_;
  vector<GuiInt*>                      gui_draw_aux_data;
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
  RTDraw2 *		                runner_;
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
  GLuint				CONTROL_SPACE_LIST;
  int 					mouse_last_x_;
  int 					mouse_last_y_;
  double 				pan_x_;
  double 				pan_y_;
  double 				scale_;
  LabelTex 				*status_label1a;
  LabelTex 				*status_label1b;
  LabelTex 				*status_label1c;
  LabelTex 				*status_label2;
  LabelTex 				*status_label3;
  LabelTex				*name_label;
  string				name_text;
  int 					injury_offset_;
  int 					lvp_index_;
  int 					rvp_index_;
  int 					ttd_index_;
  int 					pos_index_;
  int 					lvs_index_;
  int 					rvs_index_;
  int 					alarm_index_;
  int 					vector_index_;
  bool					alarm_now;
  LabelTex				*alarm_label;

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
  void 			setNameAndTime();
  void 			getIndices();
  void 			createDecisionSpace();
  void 			translate_start(int x, int y);
  void 			translate_motion(int x, int y);
  void 			translate_end(int x, int y);
  void 			scale_start(int x, int y);
  void 			scale_motion(int x, int y);
  void 			scale_end(int x, int y);
  void			screen_val(int &x, int &y);
};


class RTDraw2 : public Runnable {
public:
  RTDraw2(ExecutiveState* module) : 
    module_(module), 
    throttle_(), 
    dead_(0) 
  {};
  virtual ~RTDraw2();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  ExecutiveState *	module_;
  TimeThrottle	throttle_;
  bool		dead_;
};

RTDraw2::~RTDraw2()
{
}

void
RTDraw2::run()
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
ExecutiveState::LabelTex::set(string s)
{
  text_.replace(0, text_.length(), s);
}

void 
ExecutiveState::LabelTex::draw(float px, float py, float sx, float sy)
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
ExecutiveState::LabelTex::bind(FreeTypeFace *font) 
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

  GLboolean istex = glIsTexture(tex_id_);
  if (istex) {
    glDeleteTextures(1, &tex_id_);
  }

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


DECLARE_MAKER(ExecutiveState)

ExecutiveState::ExecutiveState(GuiContext* ctx) :
  Module("ExecutiveState", ctx, Filter, "Render", "VS"),
  gui_time_(ctx->subVar("time")),
  gui_sample_rate_(ctx->subVar("sample_rate")),
  gui_sweep_speed_(ctx->subVar("sweep_speed")),
  gui_dots_per_inch_(ctx->subVar("dots_per_inch")),
  gui_plot_height_(ctx->subVar("plot_height")),
  gui_play_mode_(ctx->subVar("play_mode")),
  gui_time_markers_mode_(ctx->subVar("time_markers_mode")),
  gui_selected_marker_(ctx->subVar("selected_marker")),
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
  markers_(0),
  cur_idx_(0),
  plots_dirty_(true),
  mouse_last_x_(0),
  mouse_last_y_(0),
  pan_x_(-AXIS_LENGTH/2),
  pan_y_(-AXIS_LENGTH/2),
  scale_(-8.0),
  name_label(0),
  name_text(" "),
  injury_offset_(0),
  lvp_index_(0),
  rvp_index_(0),
  ttd_index_(0),
  pos_index_(0),
  lvs_index_(0),
  rvs_index_(0),
  alarm_index_(0),
  vector_index_(0),
  alarm_now(0),
  alarm_label(0)
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

ExecutiveState::~ExecutiveState()
{
  if (runner_thread_) {
    runner_->set_dead(true);
    runner_thread_->join();
    runner_thread_ = 0;
  }
}

void 
ExecutiveState::inc_time(double elapsed)
{
  gui_sample_rate_.reset();
  gui_play_mode_.reset();
  gui_time_markers_mode_.reset();
  gui_sweep_speed_.reset();

  float samp_rate = gui_sample_rate_.get();  // samples per second.
  const float sweep_speed = gui_sweep_speed_.get(); 
  if (! data_.get_rep() || ! gui_play_mode_.get()) return;
  int samples = (int)round(samp_rate * elapsed * sweep_speed);
  cur_idx_ += samples;
  if (cur_idx_ >= data_->nrrd->axis[1].size) {
    cur_idx_ = data_->nrrd->axis[1].size - 1;
    gui_play_mode_.set(0);
    //cur_idx_ = 0;
  }

  gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
  gui_time_.reset();
  gui->execute("update idletasks");

  setTimeLabel();
}

bool
ExecutiveState::make_current()
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
ExecutiveState::pow2(const unsigned int dim) {
  unsigned int val = 1;
  while (val < dim) { val = val << 1; };
  return val;
}

void 
ExecutiveState::draw_counter(float x, float y) 
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
ExecutiveState::synch_plot_vars(int s)
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
  clear_vector(gui_lines_, s);
  clear_vector(gui_draw_aux_data, s);
  clear_vector(gui_auxidx_, s);
  clear_vector(gui_red_, s);
  clear_vector(gui_green_, s);
  clear_vector(gui_blue_, s);
 
}



void 
ExecutiveState::init_plots()
{
  if (! plots_dirty_) return;
  plots_dirty_ = false;

  createDecisionSpace();

  FreeTypeFace *font = fonts_["anatomical"];
  if (! font) return;

  font->set_points(18.0);
  status_label1a = scinew LabelTex("Estimated time to death: ");
  status_label1a->bind(font);
  status_label1b = scinew LabelTex(" ");
  status_label1b->bind(font);
  status_label1c = scinew LabelTex(" ");
  status_label1c->bind(font);
  status_label2 = scinew LabelTex(" ");
  status_label2->bind(font);
  status_label3 = scinew LabelTex(" ");
  status_label3->bind(font);

  font->set_points(18.0);
  if (name_label) delete name_label;
  name_label = scinew LabelTex(name_text);
  name_label->bind(font);

  font->set_points(36.0);
  if (alarm_label) delete alarm_label;
  alarm_label = scinew LabelTex("Alarm");
  alarm_label->bind(font);

  reset_vars();
  synch_plot_vars(gui_plot_count_.get());
  if (! gui_plot_count_.get()) return;

  //FreeTypeFace *font = fonts_["anatomical"];
  //if (! font) return;

  //font->set_points(18.0);
  //status_label = scinew LabelTex("Test Label");
  //status_label->bind(font);

  int i = 0;
  vector<Plot>::iterator iter = plots_.begin();
  while (iter != plots_.end()) {
    const string num = to_string(i);
    Plot &g = *iter++;

    font->set_points(36.0);
    if (g.aux_data_) delete g.aux_data_;
      g.aux_data_ = scinew LabelTex(" ");
      g.aux_data_->bind(font);

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
    if (! gui_aux_data_label_[i]) {
      gui_aux_data_label_[i] = scinew GuiString(ctx->subVar("aux_data_label-" + num));
    }
    if (gui_aux_data_label_[i]->get() != string("")) {
      if (g.aux_data_label_) delete g.aux_data_label_;
      g.aux_data_label_ = scinew LabelTex(gui_aux_data_label_[i]->get());
      g.aux_data_label_->bind(font);
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
    if (! gui_lines_[i]) {
      gui_lines_[i] = scinew GuiInt(ctx->subVar("lines-" + num));
    }
    g.lines_ = gui_lines_[i]->get();
    if (! gui_auxidx_[i]) {
      gui_auxidx_[i] = scinew GuiInt(ctx->subVar("auxidx-" + num));
    }
    g.auxindex_ = gui_auxidx_[i]->get();
    if (! gui_draw_aux_data[i]) {
      gui_draw_aux_data[i] = scinew GuiInt(ctx->subVar("draw_aux_data-" + num));
    }
    g.draw_aux_data = gui_draw_aux_data[i]->get();

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
ExecutiveState::draw_plots()
{
  reset_vars();
  //const int gr_ht = (int)gui_plot_height_.get();

  CHECK_OPENGL_ERROR("start draw_plots")

  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  const float w = width_;
  const float h = height_;
  const float sx = 1.0 / w;
  const float sy = 1.0 / h;

  //float cur_x = 20;
  //float cur_y = h - 10;

  glLineWidth(1.0);
  //glLineStipple(3, 0x1001);

  const float scale_factor = 2.0 * scale_;
  //glScalef(scale_factor, scale_factor, scale_factor);
  glPushMatrix();
    glTranslatef(pan_x_, pan_y_, scale_factor);

    //glTranslatef(-1.5, -2.5, 1.0);

    //glDisable(GL_TEXTURE_2D);
     // glCallList(CONTROL_SPACE_LIST);
    //glEnable(GL_TEXTURE_2D);

    if (data2_.get_rep()) {
      float *dat2 = (float *)data2_->nrrd->data;
      int dat2_id = cur_idx_ * data2_->nrrd->axis[0].size + alarm_index_;

      alarm_now = ((int)dat2[dat2_id] == 1);
    } 

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_POINT_SMOOTH);

    // origin, axes, background, alarm flash
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
      glLoadIdentity();
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
        glLoadIdentity();
        glLineWidth(6.0);
        glColor4f(1.0, 0.0, 0.0, (alarm_now) ? 0.6 : 0.0);
        glRectf(-1, -1, 1, 1);
        //glBegin(GL_LINE_LOOP);
        //glVertex3i(-1, -1, -1);
        //glVertex3i(1, -1, -1);
        //glVertex3i(1, 1, -1);
        //glVertex3i(-1, 1, -1);
        //glEnd();
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glLineWidth(1.0);
    glBegin(GL_LINE_LOOP);
      glColor4f(1.0, 1.0, 0.0, 1.0);
      glVertex3f(0.0, 0.0, 0.0);
      glVertex3f(AXIS_LENGTH, 0.0, 0.0);
      glVertex3f(AXIS_LENGTH, AXIS_LENGTH, 0.0);
      glVertex3f(0.0, AXIS_LENGTH, 0.0);
    glEnd();

    int i(1);
    glBegin(GL_LINES);
    glColor4f(1.0, 1.0, 0.0, 1.0);
    for (i = 1; i < AXIS_LENGTH; i++) {
      glVertex3f(i, 0.0, 0.0);
      glVertex3f(i, -0.1, 0.0);
    }
    glEnd();

    glBegin(GL_LINES);
    glColor4f(1.0, 1.0, 0.0, 1.0);
    for (i = 3; i < AXIS_LENGTH; i+=3) {
      glVertex3f(0.0, i, 0.0);
      glVertex3f(-0.1, i, 0.0);
    }
    glEnd();

    //glPointSize(6);
    //glBegin(GL_POINTS);
    //  glColor4f(1.0, 1.0, 0.0, 1.0);
    //  glVertex3f(0.0, 0.0, 0.0);
    //glEnd();
    //if (name_label) {
    //  glPushMatrix();
    //  glMatrixMode(GL_PROJECTION);
    //  glLoadIdentity();

    //  glMatrixMode(GL_MODELVIEW);
    //  glLoadIdentity();

    //  glScaled(2.0, 2.0, 2.0);
    //  glEnable(GL_TEXTURE_2D);
    //    name_label->draw(pan_x_, pan_y_, sx, sy);
    //  glDisable(GL_TEXTURE_2D);
    //  glPopMatrix();
    //}

    // path, points on path, crosshairs, second difference vector
    if (data_.get_rep()) {
      //if (cur_idx_ > 0) {
      if (cur_idx_ >= 0) {
        float *dat = (float *)data_->nrrd->data;

        //glPushMatrix();
         // glTranslatef(dat[cur_idx_*3], dat[cur_idx_*3+1], dat[cur_idx_*3+2]);
         // glDisable(GL_TEXTURE_2D);
         //   glCallList(CONTROL_SPACE_LIST);
         // glEnable(GL_TEXTURE_2D);
        //glPopMatrix();

        glPushMatrix();
          //glDisable(GL_TEXTURE_2D);
          //glEnable(GL_POINT_SMOOTH);

          glPointSize(3);
          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDepthMask(GL_FALSE);
            glDrawArrays(GL_LINE_STRIP, 0, cur_idx_ + 1);
            //glDrawArrays(GL_POINTS, 0, cur_idx_);
          //glDepthMask(GL_TRUE);

          float x = dat[cur_idx_*3];
          float y = dat[cur_idx_*3+1];
          float z = dat[cur_idx_*3+2];

          if (data2_.get_rep()) {
            float *dat2 = (float *)data2_->nrrd->data;
            int dat2_id = cur_idx_ * data2_->nrrd->axis[0].size + vector_index_;

            float x2 = dat2[dat2_id];
            float y2 = dat2[dat2_id+1];
            float z2 = 0.0;

            glBegin(GL_LINES);
              glColor4f(0.0, 1.0, 1.0, 1.0);
              glVertex3f(x, y, z);
              glVertex3f(x+x2, y+y2, z+z2);
            glEnd();

            glBegin(GL_LINES);
              glColor4f(0.0, 1.0, 1.0, 1.0);
              glVertex3f(x, y, z);
              glVertex3f(x+x2, y, z+z2);
            glEnd();
          }

          glPointSize(6);
          glBegin(GL_POINTS);
            glColor4f(0.0, 1.0, 1.0, 1.0);
            glVertex3f(x, y, z);
          glEnd();

          glBegin(GL_LINES);
            glColor4f(0.0, 1.0, 1.0, 1.0);
            glVertex3f(x-0.15, y, z);
            glVertex3f(x+0.15, y, z);

            glVertex3f(x, y-0.15, z);
            glVertex3f(x, y+0.15, z);
          glEnd();
        glPopMatrix();
      }
    }

    glDisable(GL_POINT_SMOOTH);
    glEnable(GL_TEXTURE_2D);
  glPopMatrix();

  float x_margin = 5.0;
  float y_margin = 7.0;
  glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glScaled(2.0, 2.0, 2.0);
    glTranslated(-.5, -.5, -.5);

    if (alarm_label && alarm_now) {
      float xoff = alarm_label->tex_width_ * alarm_label->u_;
      float yoff = alarm_label->tex_height_ * alarm_label->v_ * 1.5;
      //alarm_label->draw(w - xoff - x_margin, h - yoff, sx, sy);
      //alarm_label->draw(w/2 - xoff/2, h - yoff, sx, sy);
      glColor4f(1.0, 0.0, 0.0, 0.4);
      alarm_label->draw(w/2 - xoff/2, h/2 - yoff*0.5, sx, sy);
    }

    if (name_label) {
      float yoff = name_label->tex_height_ * name_label->v_ * 1.5;
      glColor4f(1.0, 1.0, 1.0, 1.0);
      //(alarm_now)?glDisable(GL_BLEND):glEnable(GL_BLEND);
      name_label->draw(x_margin, h - yoff, sx, sy);
      //(alarm_now)?glEnable(GL_BLEND):glDisable(GL_BLEND);
    }

    if (data2_.get_rep()) {
      float *dat2 = (float *)data2_->nrrd->data;
      float yoff(0.0);

      float lvp = dat2[cur_idx_ * data2_->nrrd->axis[0].size + lvp_index_];
      float rvp = dat2[cur_idx_ * data2_->nrrd->axis[0].size + rvp_index_];
      float lvs = dat2[cur_idx_ * data2_->nrrd->axis[0].size + lvs_index_];
      float rvs = dat2[cur_idx_ * data2_->nrrd->axis[0].size + rvs_index_];
      float ttd = dat2[cur_idx_ * data2_->nrrd->axis[0].size + ttd_index_];
      float pos = dat2[cur_idx_ * data2_->nrrd->axis[0].size + pos_index_];
      lvp *= 100;
      rvp *= 100;
      lvs *= 100;
      rvs *= 100;
      pos *= 100;

      FreeTypeFace *font = fonts_["anatomical"];
      font->set_points(18.0);

      yoff += y_margin;

      if (status_label3) {
        ostringstream stat;

        stat << "Power Loss: ";

        if (lvp > rvp) {
          stat << "LV (" << lvs << "%)";
          stat << ", RV (" << rvs << "%)";
        } else {
          stat << "RV (" << rvs << "%)";
          stat << ", LV (" << lvs << "%)";
        }

        status_label3->set(stat.str());
        status_label3->bind(font);

        if (lvs == lvs && rvs == rvs) {
          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDisable(GL_BLEND);
          status_label3->draw(x_margin, yoff, sx, sy);
          //glEnable(GL_BLEND);
        }

        yoff += status_label3->tex_height_*status_label3->v_ + y_margin;
      }

      if (status_label2) {
        ostringstream stat;

        //string hit = (lvp > rvp)?"Left":"Right";
        stat << "Injury: ";

        if (lvp > rvp) {
          stat << "LV (" << lvp << "% prob.)";
          stat << ", RV (" << rvp << "% prob.)";
        } else {
          stat << "RV (" << rvp << "% prob.)";
          stat << ", LV (" << lvp << "% prob.)";
        }

        status_label2->set(stat.str());
        status_label2->bind(font);

        if (lvp == lvp && rvp == rvp) {
          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDisable(GL_BLEND);
          status_label2->draw(x_margin, yoff, sx, sy);
          //glEnable(GL_BLEND);
        }

        yoff += status_label2->tex_height_*status_label2->v_ + y_margin + 2;
      }

      if (status_label1a && status_label1b && status_label1c) {
        float prob = 100 - pos;

        //ostringstream stata;
        //stata << "Estimated time to death: ";
        //status_label1a->set(stata.str());
        //status_label1a->bind(font);

        ostringstream statb;
        statb << " " << ttd << " mins.";
        status_label1b->set(statb.str());
        status_label1b->bind(font);

        ostringstream statc;
        statc << " Prob. of Death: " << prob << "%";
        status_label1c->set(statc.str());
        status_label1c->bind(font);

        float xoff(x_margin);
        if (ttd == ttd && pos == pos) {
          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDisable(GL_BLEND);
          status_label1a->draw(xoff, yoff, sx, sy);
          //glEnable(GL_BLEND);

          xoff += status_label1a->tex_width_ * status_label1a->u_;

          if (ttd < 20)
            glColor4f(1.0, 0.0, 0.0, 1.0);
          else
            glColor4f(1.0, 1.0, 1.0, 1.0);

          //glDisable(GL_BLEND);
          status_label1b->draw(xoff, yoff, sx, sy);
          //glEnable(GL_BLEND);

          xoff += status_label1b->tex_width_ * status_label1b->u_;

          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDisable(GL_BLEND);
          status_label1c->draw(xoff, yoff, sx, sy);
          //glEnable(GL_BLEND);
        }
      }

      //glPushMatrix();
      //glRotatef(69, 0.0, 0.0, 1.0);

      //glPopMatrix();
      //glColor4f(0.6, 0.0, 0.0, 1.0);
      //float gp = status_label1->tex_height_ * status_label1->v_ * sy * 0.25;
      //glRectf(x_margin * sx, yoff * sy - gp, (status_label1->tex_width_ * status_label1->u_ * sx) + (x_margin * sx), (status_label1->tex_height_ * status_label1->v_ * sy) + (yoff * sy) + gp);

      //status_label1->draw(x_margin, yoff, sx, sy);

      //glPushMatrix();
      //glMatrixMode(GL_MODELVIEW);
      //glLoadIdentity();
      //glRotatef(45, 0.0, 0.0, 1.0);
    //glScaled(2.0, 2.0, 2.0);
    //glTranslated(-.5, -.5, -.5);
      //status_label1->draw(10, h/2, sx, sy);
      //glPopMatrix();
    }
  glPopMatrix();

/*
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
      if (g.lines_ == 1) {
        glDisable(GL_TEXTURE_2D);
        glBegin(GL_LINES);
        glVertex2f((cur_x + xoff) * sx, (cur_y - gr_ht) * sy);
        glVertex2f((cur_x + (w * .70)) * sx, (cur_y - gr_ht) * sy);
        glEnd();
        glEnable(GL_TEXTURE_2D);
      }
    }    
    if (g.max_ref_label_) { 
      float xoff = g.max_ref_label_->tex_width_ * g.max_ref_label_->u_ + 3;
      float yoff = g.max_ref_label_->tex_height_ * g.max_ref_label_->v_ * 0.5;
      g.max_ref_label_->draw(cur_x, cur_y - yoff, sx, sy);
      if (g.lines_ == 1) {
        glDisable(GL_TEXTURE_2D);
        glBegin(GL_LINES);
        glVertex2f((cur_x + xoff) * sx, cur_y * sy);
        glVertex2f((cur_x + (w * .70)) * sx, cur_y * sy);
        glEnd();
        glEnable(GL_TEXTURE_2D);
      }
    }
    if (g.label_) { 
      g.label_->draw(cur_x + w * 0.70, cur_y, sx, sy);
    }
    if (g.draw_aux_data == 1) { 
      if (g.aux_data_label_)
        g.aux_data_label_->draw(cur_x + w * 0.70 + 80, cur_y, sx, sy);

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

         g.aux_data_->draw(cur_x + w * 0.70 + 70, cur_y - 50, sx, sy);
      }
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
	  if (gui_time_markers_mode_.get()) {
	     //glColor4f(0.0, 0.1, 0.9, 1.0);
	     glColor4f(1.0, 1.0, 1.0, 1.0);
	     glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, 
	   	     (start_y + val + tick) * sy);
	     glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, 
	   	     (start_y + val - tick) * sy);
	     glVertex2f((cur_x + 15 + (i * pix_per_sample)) * sx, 
	   	     (start_y + val) * sy);
	  }
	  glColor4f(g.r_, g.g_, g.b_, 1.0);
	}
      }
      glEnd();

      if (data2_.get_rep() && data2_->nrrd->axis[1].size == data_->nrrd->axis[1].size && g.snd_ == 1) {
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
*/

  CHECK_OPENGL_ERROR("end draw_plots")
}


void 
ExecutiveState::get_places(vector<int> &places, int num) const
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
ExecutiveState::redraw_all()
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
ExecutiveState::setup_gl_view()
{
  //glClearColor(0.0, 0.0, 0.0, 1.0);
  //glClear(GL_COLOR_BUFFER_BIT);

  glViewport(0, 0, width_, height_);

  //glMatrixMode(GL_MODELVIEW);
  //glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  double ratio = (double)width_/(double)height_;
  //double fovy = RtoD(2*Atan(1.0/ratio*Tan(DtoR(40.0/2.0))));
  //gluPerspective(fovy, ratio, 0.001, 10000.0);
  gluPerspective(40.0, ratio, 0.001, 10000.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  //glFlush();

  //glScaled(2.0, 2.0, 2.0);
  //glTranslated(-.5, -.5, -.5);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);

  //if (data_.get_rep()) {
   // float *dat = (float *)data_->nrrd->data;
    //glEnableClientState(GL_VERTEX_ARRAY);
    //glVertexPointer(3, GL_FLOAT, 0, dat);
  //}
}

void 
ExecutiveState::setTimeLabel()
{
    int hrs, min, sec;
    bool neg;

    int val = (int)(cur_idx_ / gui_sample_rate_.get());
    val -= injury_offset_;

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
}

void 
ExecutiveState::addMarkersToMenu()
{
  int value;
  hash_map<int, string> tmpmkrs;
  set<int> keys;
  string val;

  for (unsigned int c = 0; c < data_->nproperties(); c++) {
     string name = data_->get_property_name(c);
     //data_->get_property(name, value);
     data_->get_property(name, val);

     stringstream ss(val);
     ss >> value;

     keys.insert(value);
     tmpmkrs[value] = name;
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
ExecutiveState::setNameAndTime()
{
  char *name = nrrdKeyValueGet(data_->nrrd, "name");

  if (name != NULL) {
    string title(name);
    gui->execute(id + " setWindowTitle {Executive State: " + name + "}");

    ostringstream titlestr;
    titlestr << "Decision Space: " << title;

    name_text.replace(0, name_text.length(), titlestr.str());
                                                                                
    plots_dirty_ = true;
  }

  char *inj = nrrdKeyValueGet(data_->nrrd, "injury");
  if (inj != NULL) {
     string injoff(inj);

     stringstream ss(injoff);
     ss >> injury_offset_;

     injury_offset_ /= gui_sample_rate_.get();

     setTimeLabel();
  }
}

void 
ExecutiveState::getIndices()
{
  if (!data2_.get_rep())
    return;

  char *alarm = nrrdKeyValueGet(data2_->nrrd, "alarm");
  if (alarm != NULL) {
     string alrm(alarm);

     stringstream ss(alrm);
     ss >> alarm_index_;
  }

  char *vector = nrrdKeyValueGet(data2_->nrrd, "vector");
  if (vector != NULL) {
     string vec(vector);

     stringstream ss(vec);
     ss >> vector_index_;
  }

  char *lvp = nrrdKeyValueGet(data2_->nrrd, "lvp");
  if (lvp != NULL) {
     string lftvp(lvp);

     stringstream ss(lftvp);
     ss >> lvp_index_;
  }

  char *rvp = nrrdKeyValueGet(data2_->nrrd, "rvp");
  if (rvp != NULL) {
     string rtvp(rvp);

     stringstream ss(rtvp);
     ss >> rvp_index_;
  }

  char *ettd = nrrdKeyValueGet(data2_->nrrd, "ettd");
  if (ettd != NULL) {
     string ettds(ettd);

     stringstream ss(ettds);
     ss >> ttd_index_;
  }

  char *pos = nrrdKeyValueGet(data2_->nrrd, "pos");
  if (pos != NULL) {
     string poss(pos);

     stringstream ss(poss);
     ss >> pos_index_;
  }

  char *lvs = nrrdKeyValueGet(data2_->nrrd, "lvs");
  if (lvs != NULL) {
     string lftvs(lvs);

     stringstream ss(lftvs);
     ss >> lvs_index_;
  }

  char *rvs = nrrdKeyValueGet(data2_->nrrd, "rvs");
  if (rvs != NULL) {
     string rtvs(rvs);

     stringstream ss(rtvs);
     ss >> rvs_index_;
  }
}

void
ExecutiveState::createDecisionSpace()
{
  if (data_.get_rep()) {
    float *dat = (float *)data_->nrrd->data;
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, dat);
  }

  CONTROL_SPACE_LIST = glGenLists(1);
                                                                                
  glNewList(CONTROL_SPACE_LIST, GL_COMPILE);
    //glTranslatef(-1.5, -2.5, 5.0); // position scene
    //glTranslatef(-1.5, -2.5, 1.0); // position scene
                                                                               
    glPushMatrix();
      //glTranslatef(1.0, 2.0, 0.0); // position control space
                                                                                
      glBegin(GL_TRIANGLES);
        glColor4f(0.0, 1.0, 0.0, 0.75);
        glVertex3f(-1.0, -1.0, 0.0);
        //glVertex3f(-1.0, 2.0, 0.0);
        //glVertex3f(2.0, 2.0, 0.0);
        glVertex3f(-1.0, 1.0, 0.0);
        glVertex3f(1.0, 1.0, 0.0);
      glEnd();

      glBegin(GL_TRIANGLES);
        glColor4f(1.0, 0.0, 0.0, 0.75);
        glVertex3f(-1.0, -1.0, 0.0);
        //glVertex3f(2.0, 2.0, 0.0);
        //glVertex3f(2.0, -1.0, 0.0);
        glVertex3f(1.0, 1.0, 0.0);
        glVertex3f(1.0, -1.0, 0.0);
      glEnd();
/*
      glColor4f(0.0, 0.0, 1.0, 1.0);
      glPushMatrix();
        glBegin(GL_LINE_STRIP);
glColor4f(1.0, 1.0, 0.0, 1.0);
          glVertex3f(-1.0, -1.0, 0.0);
glColor4f(0.0, 0.0, 1.0, 1.0);
          glVertex3f(1.0, 1.0, 0.0);
        glEnd();
      glPopMatrix();
                                                                                
      GLUquadricObj *qobj;
      qobj = gluNewQuadric();
      gluQuadricDrawStyle(qobj, GLU_SILHOUETTE);
      gluQuadricNormals(qobj, GLU_NONE);
      glColor4f(1.0, 0.0, 0.0, 1.0);
      glPushMatrix();
        glTranslatef(1.0, 0.0, 0.0);
        gluDisk(qobj, 0.0, 0.5, 100, 1);
                                                                                
        glPointSize(4);
        glEnable(GL_POINT_SMOOTH);
        glBegin(GL_POINTS);
          glVertex3f(0.0, 0.0, 0.0);
        glEnd();
        glDisable(GL_POINT_SMOOTH);
      glPopMatrix();
      gluDeleteQuadric(qobj);
             
      qobj = gluNewQuadric();
      gluQuadricDrawStyle(qobj, GLU_SILHOUETTE);
      gluQuadricNormals(qobj, GLU_NONE);
      glColor4f(0.0, 1.0, 0.0, 1.0);
      glPushMatrix();
        glTranslatef(0.0, 1.0, 0.0);
        gluDisk(qobj, 0.0, 0.5, 100, 1);
             
        glPointSize(4);
        glEnable(GL_POINT_SMOOTH);
        glBegin(GL_POINTS);
          glVertex3f(0.0, 0.0, 0.0);
        glEnd();
        glDisable(GL_POINT_SMOOTH);
      glPopMatrix();
      gluDeleteQuadric(qobj);
*/
   glPopMatrix();
   glEndList();
}

void
ExecutiveState::translate_start(int x, int y)
{
  mouse_last_x_ = x;
  mouse_last_y_ = y;
}

void
ExecutiveState::translate_motion(int x, int y)
{
  float xmtn = float(mouse_last_x_ - x) / float(width_ / scale_);
  float ymtn = float(y - mouse_last_y_) / float(height_ / scale_);
  mouse_last_x_ = x;
  mouse_last_y_ = y;

  //pan_x_ = (pan_x_ + xmtn / scale_);
  //pan_y_ = (pan_y_ + ymtn / scale_);
  pan_x_ = (pan_x_ + xmtn);
  pan_y_ = (pan_y_ + ymtn);

  redraw_all();
}

void
ExecutiveState::translate_end(int x, int y)
{
  redraw_all();
}

void
ExecutiveState::scale_start(int x, int y)
{
  mouse_last_y_ = y;
}

void
ExecutiveState::scale_motion(int x, int y)
{
  float ymtn = -float(mouse_last_y_ - y) / float(height_);
  mouse_last_y_ = y;

  scale_ = (scale_ + -ymtn);

  //if (scale_ < 0.0) scale_ = 0.0;

  redraw_all();
}

void
ExecutiveState::scale_end(int x, int y)
{
  redraw_all();
}

int
do_round(float d)
{
  if (d > 0.0) return (int)(d + 0.5);
  else return -(int)(-d + 0.5);
}

void
ExecutiveState::screen_val(int &x, int &y)
{
  const float cx = width_ * 0.5;
  const float cy = height_ * 0.5;
  const float sf_inv = 1.0 / scale_;
 
  x = do_round((x - cx) * sf_inv + cx + (pan_x_ * width_));
  y = do_round((y - cy) * sf_inv + cy + (pan_y_ * height_));
}

void
ExecutiveState::execute()
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

  double rt = data_->nrrd->axis[0].spacing;
  if (rt == rt)
    gui_sample_rate_.set(1/data_->nrrd->axis[0].spacing);

  addMarkersToMenu();

  setNameAndTime();

  //getIndices();

  plots_dirty_ = true;

  NrrdIPort *nrrd2_port = (NrrdIPort*)get_iport("Nrrd2");

  if (!nrrd2_port) 
  {
    error("Unable to initialize iport Nrrd2.");
    return;
  }

  nrrd2_port->get(data2_);

  if (data2_.get_rep() && data2_->nrrd->axis[1].size != data_->nrrd->axis[1].size)
  {
    error ("Axis 1 size for both NRRD files must be identical.");
  } 
  
  getIndices();

  if (!runner_) {
    runner_ = scinew RTDraw2(this);
    runner_thread_ = scinew Thread(runner_, string(id+" RTDraw2 OpenGL").c_str());
  }
}

void
ExecutiveState::tcl_command(GuiArgs& args, void* userdata) 
{
  if(args.count() < 2) {
    args.error("ExecutiveState needs a minor command");
    return;
  } else if(args[1] == "mouse") {
    int x, y;

    if (args[2] == "translate") {
      string_to_int(args[4], x);
      string_to_int(args[5], y);
      //screen_val(x, y);
      if (args[3] == "start") {
        translate_start(x, y);
      } else if (args[3] == "move") {
        translate_motion(x, y);
      } else { // end
        translate_end(x, y);
      }

    } else if (args[2] == "reset") {
      pan_x_ = -AXIS_LENGTH/2;
      pan_y_ = -AXIS_LENGTH/2;
      scale_ = -8.0;

      redraw_all();

    } else if (args[2] == "scale") {
      string_to_int(args[4], x);
      string_to_int(args[5], y);
      if (args[3] == "start") {
        scale_start(x, y);
      } else if (args[3] == "move") {
        scale_motion(x, y);
      } else { // end
        scale_end(x, y);
      }
    }

  } else if(args[1] == "time") {
    gui_time_.reset();
    if (data_.get_rep()) {
      cur_idx_ = (int)round(gui_time_.get() * (data_->nrrd->axis[1].size - 1));
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

    if (val == -1 || val >= data_->nrrd->axis[1].size) return;
    cur_idx_ = val;

    gui_time_.set((float)cur_idx_ / (float)data_->nrrd->axis[1].size);
    gui_time_.reset();
    gui->execute("update idletasks");

    setTimeLabel();

  } else if(args[1] == "increment") {
    if (!data_.get_rep()) return;

    cur_idx_ += (int)gui_sample_rate_.get();
    if (cur_idx_ >= data_->nrrd->axis[1].size) {
        cur_idx_ = data_->nrrd->axis[1].size - 1;
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
      gui->unlock();
    } else {
      width_ = Tk_Width(tkwin);
      height_ = Tk_Height(tkwin);

      setup_gl_view();
    }

  } else {
    Module::tcl_command(args, userdata);
  }
}



} // End namespace VS
