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
#include <Dataflow/Ports/TimePort.h>

#include <Core/Geom/FreeType.h>

#include <Core/Util/Environment.h>

#include <typeinfo>
#include <iostream>
#include <sstream>
#include <iomanip>

#define X_AXIS_LENGTH (GLdouble)10.0
#define Y_AXIS_LENGTH (GLdouble)5.0
//#define ALARM_OFFSET 5880

//#define TIME_IDX 0
#define LV_INJURY_IDX 1
#define RV_INJURY_IDX 2
#define TIME_TO_DEATH_IDX 3
#define PROB_OF_SURV_IDX 4
#define LV_POWER_IDX 5
#define RV_POWER_IDX 6
#define VECTOR_IDX 7
#define ALARM_IDX 9
//#define FIRST_ALARM_IDX 10

#define VM_SLIDING 0
#define VM_PATH 1

typedef struct {GLdouble x; GLdouble y; GLdouble z;} tuple;

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
    void draw(float px, float py, float sx, float sy);
    void draw(float px, float py, float sx, float sy, float rotation);
    void bind(FreeTypeFace *font);
    void set(string s);
    
    unsigned      tex_width_;
    unsigned      tex_height_;
    float         u_;
    float         v_;
    GLuint        tex_id_;
    string        text_;
    static const float NORMAL       = 0.0;
    static const float ROTATE_LEFT  = 90.0;
    static const float ROTATE_RIGHT = 270.0;
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
  GuiDouble                            gui_font_scale_;
  GuiInt                               gui_show_name_;
  GuiInt                               gui_show_vectors_;
  GuiInt                               gui_horiz_comp_;
  GuiInt                               gui_both_comp_;
  GuiInt											gui_vector_mode_;
  GuiInt                               gui_show_cross_;
  GuiInt											gui_show_trend_;
  GuiInt											gui_show_threshold_;
  GuiInt                               gui_injury_offset_;
  GuiInt											gui_geom_;
  GuiString                            *gui_x_axis_label_;
  GuiString                            *gui_x_trend_h_label_;
  GuiString                            *gui_x_trend_m_label_;
  GuiString                            *gui_x_trend_l_label_;
  GuiString                            *gui_y_axis_label_;
  GuiString                            *gui_y_trend_h_label_;
  GuiString                            *gui_y_trend_m_label_;
  GuiString                            *gui_y_trend_l_label_;

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
  NrrdDataHandle                        data_;
  NrrdDataHandle                        data2_;
  vector<int>                           markers_;
  int                                   cur_idx_;
  bool                                  plots_dirty_;
  bool                                  decision_space_dirty_;
  //GLuint				CONTROL_SPACE_LIST;
  int 					mouse_last_x_;
  int 					mouse_last_y_;
  double 				pan_x_;
  double 				pan_y_;
  double 				scale_;
  LabelTex 				*status_label1a;
  LabelTex 				*status_label1b;
  LabelTex 				*status_label1c;
  LabelTex 				*status_label1d;
  LabelTex 				*status_label2;
  LabelTex 				*status_label3;
  LabelTex				*name_label;
  LabelTex				*x_axis_label;
  LabelTex				*x_trend_h_label;
  LabelTex				*x_trend_m_label;
  LabelTex				*x_trend_l_label;
  LabelTex				*y_axis_label;
  LabelTex				*y_trend_h_label;
  LabelTex				*y_trend_m_label;
  LabelTex				*y_trend_l_label;
  string				name_text;
  LabelTex				*time_label;
  string				time_text;
  //int 					injury_offset_;
  //int 					alarm_offset_;
  bool					alarm_now;
  LabelTex				*alarm_label;
  //GLfloat				*color_dat;
  //tuple					*injury_point;
  //tuple					*alarm_point;
  tuple					*x_axis_point;
  tuple					*x_trend_h_point;
  tuple					*x_trend_m_point;
  tuple					*x_trend_l_point;
  tuple					*y_axis_point;
  tuple					*y_trend_h_point;
  tuple					*y_trend_m_point;
  tuple					*y_trend_l_point;
  TimeViewerHandle	time_viewer_h_;

  bool                  make_current();
  void                  init_plots();
  void                  draw_plots();
  void                  draw_counter(float x, float y);
  void                  get_places(vector<int> &places, int num) const;
  void			setup_gl_view();
  static unsigned int	pow2(const unsigned int);
  void 			setTimeLabel();
  void 			addMarkersToMenu();
  void 			getNrrd1KeyValues();
  void 			createDecisionSpaceArrays();
  void 			translate_start(int x, int y);
  void 			translate_motion(int x, int y);
  void 			translate_end(int x, int y);
  void 			scale_start(int x, int y);
  void 			scale_motion(int x, int y);
  void 			scale_end(int x, int y);
};


class RTDraw2 : public Runnable {
public:
  //RTDraw2(ExecutiveState* module) : 
  RTDraw2(ExecutiveState* module, TimeViewerHandle tvh) : 
    module_(module), 
    throttle_(), 
	 tvh_(tvh),
    dead_(0) 
  {};
  virtual ~RTDraw2();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  ExecutiveState *	module_;
  TimeThrottle	throttle_;
  TimeViewerHandle	tvh_;
  bool		dead_;
};

RTDraw2::~RTDraw2()
{
}

void
RTDraw2::run()
{
  throttle_.start();
  //const double inc = 1./75.;
  const double inc = 1./20.;
  double t = throttle_.time();
  //double tlast = t;
  while (!dead_) {
    t = throttle_.time();
    throttle_.wait_for_time(t + inc);
    //double elapsed = t - tlast;
    //module_->inc_time(elapsed);
    module_->inc_time(tvh_->view_elapsed_since_start());
    module_->redraw_all();
    //tlast = t;
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
    ExecutiveState::LabelTex::draw(px, py, sx, sy, LabelTex::NORMAL);
}

void 
ExecutiveState::LabelTex::draw(float px, float py, float sx, float sy, float rotation)
{
  glBindTexture(GL_TEXTURE_2D, tex_id_);
  
  //float qwidth = tex_width_  * u_ * sx;
  //float qheight = tex_height_ * v_ * sy;

  float qwidth = (rotation==LabelTex::NORMAL)?tex_width_*u_*sx : tex_width_*u_*sy;
  float qheight = (rotation==LabelTex::NORMAL)?tex_height_*v_*sy : tex_height_*v_*sx;
  float tx = px * sx;
  float ty = py * sy;
  
  glPushMatrix();
    glTranslatef(tx, ty, 0.0);
    glRotatef(rotation, 0.0, 0.0, 1.0);

    glBegin(GL_QUADS);
      glTexCoord2f(0.0, 0.0);
      //glVertex2f(px * sx, py * sy);
      glVertex2f(0.0, 0.0);
      glTexCoord2f(u_, 0.0);
      //glVertex2f(px * sx + qwidth, py * sy);
      glVertex2f(qwidth, 0.0);
      glTexCoord2f(u_, v_);
      //glVertex2f(px * sx + qwidth, py * sy + qheight);      
      glVertex2f(qwidth, qheight);
      glTexCoord2f(0.0, v_);
      //glVertex2f(px * sx, py * sy + qheight);
      glVertex2f(0.0, qheight);
    glEnd();
  glPopMatrix();

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
  gui_font_scale_(ctx->subVar("font_scale")),
  gui_show_name_(ctx->subVar("show_name")),
  gui_show_vectors_(ctx->subVar("show_vectors")),
  gui_horiz_comp_(ctx->subVar("horiz_comp")),
  gui_both_comp_(ctx->subVar("both_comp")),
  gui_vector_mode_(ctx->subVar("vector_mode")),
  gui_show_cross_(ctx->subVar("show_cross")),
  gui_show_trend_(ctx->subVar("show_trend")),
  gui_show_threshold_(ctx->subVar("show_threshold")),
  gui_injury_offset_(ctx->subVar("injury_offset")),
  gui_geom_(ctx->subVar("geom")),
  gui_x_axis_label_(0),
  gui_x_trend_h_label_(0),
  gui_x_trend_m_label_(0),
  gui_x_trend_l_label_(0),
  gui_y_axis_label_(0),
  gui_y_trend_h_label_(0),
  gui_y_trend_m_label_(0),
  gui_y_trend_l_label_(0),
  ctx_(0),
  dpy_(0),
  win_(0),
  freetype_lib_(0),
  fonts_(),
  runner_(0),
  runner_thread_(0),
  digits_(10),
  dig_init_(false),
  data_(0),
  data2_(0),
  markers_(0),
  cur_idx_(0),
  plots_dirty_(true),
  decision_space_dirty_(true),
  mouse_last_x_(0),
  mouse_last_y_(0),
  pan_x_(-X_AXIS_LENGTH/2),
  pan_y_(-Y_AXIS_LENGTH/2),
  scale_(-8.0),
  status_label1a(0),
  status_label1b(0),
  status_label1c(0),
  status_label1d(0),
  status_label2(0),
  status_label3(0),
  name_label(0),
  x_axis_label(0),
  x_trend_h_label(0),
  x_trend_m_label(0),
  x_trend_l_label(0),
  y_axis_label(0),
  y_trend_h_label(0),
  y_trend_m_label(0),
  y_trend_l_label(0),
  name_text(" "),
  time_label(0),
  time_text("Time: 00:00:00"),
  //injury_offset_(0),
  //alarm_offset_(ALARM_OFFSET),
  alarm_now(0),
  alarm_label(0),
  //color_dat(0),
  //injury_point(0),
  //alarm_point(0),
  x_axis_point(0),
  x_trend_h_point(0),
  x_trend_m_point(0),
  x_trend_l_point(0),
  y_axis_point(0),
  y_trend_h_point(0),
  y_trend_m_point(0),
  y_trend_l_point(0)
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

  //injury_point = new tuple;
  //alarm_point = new tuple;
  x_axis_point = new tuple;
  x_trend_h_point = new tuple;
  x_trend_m_point = new tuple;
  x_trend_l_point = new tuple;
  y_axis_point = new tuple;
  y_trend_h_point = new tuple;
  y_trend_m_point = new tuple;
  y_trend_l_point = new tuple;

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
  //gui_sweep_speed_.reset();

  float samp_rate = gui_sample_rate_.get();  // samples per second.
  //const float sweep_speed = gui_sweep_speed_.get(); 
  if (! data_.get_rep() || ! gui_play_mode_.get()) return;
  //int samples = (int)round(samp_rate * elapsed * sweep_speed);
  int samples = (int)round(samp_rate * elapsed);
  //cur_idx_ += samples;
  cur_idx_ = samples;
  if (cur_idx_ >= data_->nrrd->axis[1].size) {
    cur_idx_ = data_->nrrd->axis[1].size - 1;
    //gui_play_mode_.set(0);
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

void 
ExecutiveState::init_plots()
{
  if (! plots_dirty_)
    return;

  plots_dirty_ = false;

  createDecisionSpaceArrays();

  FreeTypeFace *font = fonts_["anatomical"];
  if (! font)
    return;

  font->set_points(18.0 * gui_font_scale_.get());
  if (status_label1a) delete status_label1a;
  status_label1a = scinew LabelTex("Estimated time to death: ");
  status_label1a->bind(font);
  if (status_label1b) delete status_label1b;
  status_label1b = scinew LabelTex(" ");
  status_label1b->bind(font);
  if (status_label1c) delete status_label1c;
  status_label1c = scinew LabelTex(" Prob. of death: ");
  status_label1c->bind(font);
  if (status_label1d) delete status_label1d;
  status_label1d = scinew LabelTex(" ");
  status_label1d->bind(font);
  if (status_label2) delete status_label2;
  status_label2 = scinew LabelTex(" ");
  status_label2->bind(font);
  if (status_label3) delete status_label3;
  status_label3 = scinew LabelTex(" ");
  status_label3->bind(font);

  font->set_points(18.0 * gui_font_scale_.get());
  if (name_label) delete name_label;
  name_label = scinew LabelTex(name_text);
  name_label->bind(font);

  if (time_label) delete time_label;
  time_label = scinew LabelTex(time_text);
  time_label->bind(font);

  if (!gui_x_axis_label_) {
    gui_x_axis_label_ = scinew GuiString(ctx->subVar("x_axis_label"));
  }
  //if (gui_x_axis_label_->get() != string("")) {
    if (x_axis_label) delete x_axis_label;
    //x_axis_label = scinew LabelTex("Amplitude, LV Waveform");
    //x_axis_label = scinew LabelTex(gui_x_axis_label_->get());
    x_axis_label = scinew LabelTex((gui_x_axis_label_->get() != string(""))?gui_x_axis_label_->get():string(" "));
    x_axis_label->bind(font);
  //}

  if (!gui_x_trend_h_label_) {
    gui_x_trend_h_label_ = scinew GuiString(ctx->subVar("x_trend_h_label"));
  }
  //if (gui_x_trend_h_label_->get() != string("")) {
    if (x_trend_h_label) delete x_trend_h_label;
    //x_trend_h_label = scinew LabelTex("Baseline");
    x_trend_h_label = scinew LabelTex((gui_x_trend_h_label_->get() != string(""))?gui_x_trend_h_label_->get():string(" "));
    x_trend_h_label->bind(font);
  //}

  if (!gui_x_trend_m_label_) {
    gui_x_trend_m_label_ = scinew GuiString(ctx->subVar("x_trend_m_label"));
  }
  //if (gui_x_trend_m_label_->get() != string("")) {
    if (x_trend_m_label) delete x_trend_m_label;
    //x_trend_m_label = scinew LabelTex("LV");
    x_trend_m_label = scinew LabelTex((gui_x_trend_m_label_->get() != string(""))?gui_x_trend_m_label_->get():string(" "));
    x_trend_m_label->bind(font);
  //}

  if (!gui_x_trend_l_label_) {
    gui_x_trend_l_label_ = scinew GuiString(ctx->subVar("x_trend_l_label"));
  }
  //if (gui_x_trend_l_label_->get() != string("")) {
    if (x_trend_l_label) delete x_trend_l_label;
    //x_trend_l_label = scinew LabelTex("Decline");
	 x_trend_l_label = scinew LabelTex((gui_x_trend_l_label_->get() != string(""))?gui_x_trend_l_label_->get():string(" "));
    x_trend_l_label->bind(font);
  //}

  if (!gui_y_axis_label_) {
    gui_y_axis_label_ = scinew GuiString(ctx->subVar("y_axis_label"));
  }
  //if (gui_y_axis_label_->get() != string("")) {
    if (y_axis_label) delete y_axis_label;
    //y_axis_label = scinew LabelTex("Amplitude, RV Waveform");
    y_axis_label = scinew LabelTex((gui_y_axis_label_->get() != string(""))?gui_y_axis_label_->get():string(" "));
    y_axis_label->bind(font);
  //}

  if (!gui_y_trend_h_label_) {
    gui_y_trend_h_label_ = scinew GuiString(ctx->subVar("y_trend_h_label"));
  }
  //if (gui_y_trend_h_label_->get() != string("")) {
    if (y_trend_h_label) delete y_trend_h_label;
    //y_trend_h_label = scinew LabelTex("Baseline");
    y_trend_h_label = scinew LabelTex((gui_y_trend_h_label_->get() != string(""))?gui_y_trend_h_label_->get():string(" "));
    y_trend_h_label->bind(font);
  //}

  if (!gui_y_trend_m_label_) {
    gui_y_trend_m_label_ = scinew GuiString(ctx->subVar("y_trend_m_label"));
  }
  //if (gui_y_trend_m_label_->get() != string("")) {
    if (y_trend_m_label) delete y_trend_m_label;
    //y_trend_m_label = scinew LabelTex("RV");
    y_trend_m_label = scinew LabelTex((gui_y_trend_m_label_->get() != string(""))?gui_y_trend_m_label_->get():string(" "));
    y_trend_m_label->bind(font);
  //}

  if (!gui_y_trend_l_label_) {
    gui_y_trend_l_label_ = scinew GuiString(ctx->subVar("y_trend_l_label"));
  }
  //if (gui_y_trend_l_label_->get() != string("")) {
    if (y_trend_l_label) delete y_trend_l_label;
    //y_trend_l_label = scinew LabelTex("Decline");
    y_trend_l_label = scinew LabelTex((gui_y_trend_l_label_->get() != string(""))?gui_y_trend_l_label_->get():string(" "));
    y_trend_l_label->bind(font);
  //}

  font->set_points(36.0 * gui_font_scale_.get());
  if (alarm_label) delete alarm_label;
  alarm_label = scinew LabelTex("Alarm");
  alarm_label->bind(font);
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
      int dat2_id = cur_idx_ * data2_->nrrd->axis[0].size + ALARM_IDX;

      alarm_now = ((int)dat2[dat2_id] == 1);

      //int inj_id = cur_idx_ * data2_->nrrd->axis[0].size + TIME_IDX;
      //float time_value = dat2[inj_id];
		//cerr << cur_idx_ << ":" << time_value << endl;
		//if (time_value == 0)
		//	cerr << cur_idx_ << ":" << time_value << endl;
    } 

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_POINT_SMOOTH);

    // axes, ticks, background, alarm flash
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
      glLoadIdentity();
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
        glLoadIdentity();
        glColor4f(1.0, 0.0, 0.0, (alarm_now) ? 0.6 : 0.0);
        glRectf(-1, -1, 1, 1);
        //glLineWidth(6.0);
        //glBegin(GL_LINE_LOOP);
        //glVertex3i(-1, -1, -1);
        //glVertex3i(1, -1, -1);
        //glVertex3i(1, 1, -1);
        //glVertex3i(-1, 1, -1);
        //glEnd();
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    //glLineWidth(1.0);
    glBegin(GL_LINE_LOOP);
      glColor4f(1.0, 1.0, 0.0, 1.0);
      glVertex3f(0.0, 0.0, 0.0);
      glVertex3f(X_AXIS_LENGTH, 0.0, 0.0);
      glVertex3f(X_AXIS_LENGTH, Y_AXIS_LENGTH, 0.0);
      glVertex3f(0.0, Y_AXIS_LENGTH, 0.0);
    glEnd();

	if (gui_show_trend_.get()) {
		glShadeModel(GL_SMOOTH);
		glBegin(GL_LINES);
			glColor4f(0.0, 1.0, 0.0, 1.0);
			glVertex3f(X_AXIS_LENGTH, Y_AXIS_LENGTH, 0.0);
			glColor4f(1.0, 0.0, 0.0, 1.0);
			glVertex3f(X_AXIS_LENGTH, 0.0, 0.0);

			glColor4f(0.0, 1.0, 0.0, 1.0);
			glVertex3f(X_AXIS_LENGTH, Y_AXIS_LENGTH, 0.0);
			glColor4f(1.0, 0.0, 0.0, 1.0);
			glVertex3f(0.0, Y_AXIS_LENGTH, 0.0);
		glEnd();
	}

		glShadeModel(GL_FLAT);

    int i(1);
    glBegin(GL_LINES);
      glColor4f(1.0, 1.0, 0.0, 1.0);
      for (i = 1; i < X_AXIS_LENGTH; i++) {
        glVertex3f(i, 0.0, 0.0);
        glVertex3f(i, -0.1, 0.0);
      }

      //for (i = 3; i < Y_AXIS_LENGTH; i+=3) {
      for (i = 1; i < Y_AXIS_LENGTH; i++) {
        glVertex3f(0.0, i, 0.0);
        glVertex3f(-0.1, i, 0.0);
      }
    glEnd();

    //glBegin(GL_LINES);
    //glColor4f(1.0, 1.0, 0.0, 1.0);
    //for (i = 3; i < AXIS_LENGTH; i+=3) {
     // glVertex3f(0.0, i, 0.0);
      //glVertex3f(-0.1, i, 0.0);
    //}
    //glEnd();

    //if (data2_.get_rep() && gui_show_vectors_.get() && gui_horiz_comp_.get() && gui_vector_mode_.get() == VM_SLIDING) {
     // glBegin(GL_LINES);
      //  glColor4f(1.0, 0.0, 0.0, 0.6);
      //  glVertex3f(-1.0, 0.0, 0.0);
       // glVertex3f(-1.0, Y_AXIS_LENGTH, 0.0);

      //  glVertex3f(-1.1, Y_AXIS_LENGTH, 0.0);
       // glVertex3f(-0.9, Y_AXIS_LENGTH, 0.0);

      //  glVertex3f(-1.1, 0.0, 0.0);
       // glVertex3f(-0.9, 0.0, 0.0);
      //glEnd();
    //}

    GLint vp[4];
    GLdouble mm[16], pm[16];
    glGetIntegerv(GL_VIEWPORT, vp);
    glGetDoublev(GL_MODELVIEW_MATRIX, mm);
    glGetDoublev(GL_PROJECTION_MATRIX, pm);

    gluProject(X_AXIS_LENGTH/2, (GLdouble)0.0 - 0.1, (GLdouble)0.0, mm, pm, vp,
	 	&x_axis_point->x, &x_axis_point->y, &x_axis_point->z);
    gluProject(X_AXIS_LENGTH, Y_AXIS_LENGTH, (GLdouble)0.0, mm, pm, vp,
		&x_trend_h_point->x, &x_trend_h_point->y, &x_trend_h_point->z);
    gluProject(X_AXIS_LENGTH/2, Y_AXIS_LENGTH, (GLdouble)0.0, mm, pm, vp,
	 	&x_trend_m_point->x, &x_trend_m_point->y, &x_trend_m_point->z);
    gluProject((GLdouble)0.0, Y_AXIS_LENGTH, (GLdouble)0.0, mm, pm, vp,
		&x_trend_l_point->x, &x_trend_l_point->y, &x_trend_l_point->z);

    gluProject((GLdouble)0.0 - 0.1, Y_AXIS_LENGTH/2, (GLdouble)0.0, mm, pm, vp,
	 	&y_axis_point->x, &y_axis_point->y, &y_axis_point->z);
    gluProject(X_AXIS_LENGTH, Y_AXIS_LENGTH, (GLdouble)0.0, mm, pm, vp,
		&y_trend_h_point->x, &y_trend_h_point->y, &y_trend_h_point->z);
    gluProject(X_AXIS_LENGTH, Y_AXIS_LENGTH/2, (GLdouble)0.0, mm, pm, vp,
		&y_trend_m_point->x, &y_trend_m_point->y, &y_trend_m_point->z);
    gluProject(X_AXIS_LENGTH, (GLdouble)0.0, (GLdouble)0.0, mm, pm, vp,
		&y_trend_l_point->x, &y_trend_l_point->y, &y_trend_l_point->z);

    // path, points on path, crosshairs, second difference vector
    if (data_.get_rep()) {
      //if (cur_idx_ > 0) {
      if (cur_idx_ >= 0) {
        float *dat = (float *)data_->nrrd->data;

        glPushMatrix();
          //glDisable(GL_TEXTURE_2D);
          //glEnable(GL_POINT_SMOOTH);

			 // scale the data to fit in the graph
			 glScaled(0.1, 0.1, 1.0);

          glPointSize(3);
          glColor4f(1.0, 1.0, 1.0, 1.0);
			 //glPushMatrix();
          //glDepthMask(GL_FALSE);
            glDrawArrays(GL_LINE_STRIP, 0, cur_idx_ + 1);
            //glDrawArrays(GL_POINTS, 0, cur_idx_);
          //glDepthMask(GL_TRUE);
			 //glPopMatrix();

          float x = dat[cur_idx_*6+3];
          float y = dat[cur_idx_*6+4];
          float z = dat[cur_idx_*6+5];

          if (data2_.get_rep() && gui_show_vectors_.get()) {
            float *dat2 = (float *)data2_->nrrd->data;
            int dat2_id = cur_idx_ * data2_->nrrd->axis[0].size + VECTOR_IDX;

            float x2 = dat2[dat2_id];
            float y2 = dat2[dat2_id+1];
            float z2 = 0.0;

				// threshold
				if (gui_show_threshold_.get() && gui_vector_mode_.get() == VM_SLIDING) {
					glBegin(GL_LINES);
						glColor4f(1.0, 0.0, 0.0, 0.6);
						glVertex3f(-10.0, 0.0, 0.0);
						glVertex3f(-10.0, Y_AXIS_LENGTH * 10, 0.0);

						glVertex3f(-11.0, Y_AXIS_LENGTH * 10, 0.0);
						glVertex3f(-9.0, Y_AXIS_LENGTH * 10, 0.0);

						glVertex3f(-11.0, 0.0, 0.0);
						glVertex3f(-9.0, 0.0, 0.0);
					glEnd();
				}

				// thermometer
				if (gui_show_threshold_.get() && gui_vector_mode_.get() == VM_PATH) {
              glBegin(GL_LINES);
                glColor4f(1.0, 0.0, 0.0, 0.6);
                glVertex3f(x, y, z);
                glVertex3f(x-10, y, z);

                glVertex3f(x-10, y+1, z);
                glVertex3f(x-10, y-1, z);
              glEnd();
				}

				if (x2 == x2 && y2 == y2) {
					// x and y 2nd difference vector
					if (gui_both_comp_.get()) {
						glBegin(GL_LINES);
							glColor4f(0.0, 1.0, 1.0, 1.0);
							glVertex3f(x, y, z);
							glVertex3f(x+x2, y+y2, z+z2);
						glEnd();
					}
				  
					// x 2nd difference vector translated to y axis
					if (gui_horiz_comp_.get()) {
						glPushMatrix();
							if (gui_vector_mode_.get() == VM_SLIDING) {
								glTranslatef(-x, 0.0, 0.0);
							}
							glBegin(GL_LINES);
								glColor4f(0.0, 1.0, 1.0, 1.0);
									glVertex3f(x, y, z);
									glVertex3f(x+x2, y, z+z2);
							glEnd();
						glPopMatrix();
					}
				}
			}

    //if (data2_.get_rep()) {
     // float *dat2 = (float *)data2_->nrrd->data;

      //int inj_id = cur_idx_ * data2_->nrrd->axis[0].size + TIME_IDX;
      //float time_value = dat2[inj_id];

		//if (time_value == 0) {
			//cerr << cur_idx_ << ":" << time_value << endl;
		//		injury_point->x = x;
		//		injury_point->y = y;
		//		injury_point->z = z;
		//}
	//}

			//if (cur_idx_ >= gui_injury_offset_.get() * 100 && cur_idx_ < gui_injury_offset_.get() * 100 + 6000) {
			//	injury_point->x = x;
			//	injury_point->y = y;
			//	injury_point->z = z;
			//}

			//if (cur_idx_ >= alarm_offset_ * 100 && cur_idx_ < alarm_offset_ * 100 + 6000) {
			//	alarm_point->x = x;
			//	alarm_point->y = y;
			//	alarm_point->z = z;
			//}

         glPointSize(6);
			//if (cur_idx_ >= gui_injury_offset_.get() * 100) {
			//if (injury_point->x != 0) {
			//	glBegin(GL_POINTS);
			//		glColor4f(0.0, 1.0, 0.0, 1.0);
			//		glVertex3f(injury_point->x, injury_point->y, injury_point->z);
			//	glEnd();
			//}

			//if (cur_idx_ >= alarm_offset_ * 100) {
			//	glBegin(GL_POINTS);
			//		glColor4f(1.0, 0.0, 0.0, 1.0);
			//		glVertex3f(alarm_point->x, alarm_point->y, alarm_point->z);
			//	glEnd();
			//}

			 // lead point
          glBegin(GL_POINTS);
            glColor4f(0.0, 1.0, 1.0, 1.0);
            glVertex3f(x, y, z);
          glEnd();

			 // cross hairs on lead point
          if (gui_show_cross_.get()) {
            glBegin(GL_LINES);
              glColor4f(0.0, 1.0, 1.0, 1.0);
              glVertex3f(x-1.5, y, z);
              glVertex3f(x+1.5, y, z);

              glVertex3f(x, y-1.5, z);
              glVertex3f(x, y+1.5, z);
            glEnd();
          }
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

    if (name_label && gui_show_name_.get()) {
      float yoff = name_label->tex_height_ * name_label->v_ * 1.5;
      glColor4f(1.0, 1.0, 1.0, 1.0);
      //(alarm_now)?glDisable(GL_BLEND):glEnable(GL_BLEND);
      name_label->draw(x_margin, h - yoff, sx, sy);
      //(alarm_now)?glEnable(GL_BLEND):glDisable(GL_BLEND);
    }

    if (x_axis_label) {
      glColor4f(1.0, 1.0, 0.0, 1.0);
      float xoff = x_axis_label->tex_width_ * x_axis_label->u_;
      float yoff = x_axis_label->tex_height_ * x_axis_label->v_ * 1.5;
      x_axis_label->draw(x_axis_point->x - xoff/2, x_axis_point->y - yoff, sx, sy);
    }

    if (y_axis_label) {
      glColor4f(1.0, 1.0, 0.0, 1.0);
      float xoff = y_axis_label->tex_width_ * y_axis_label->u_;
      float yoff = y_axis_label->tex_height_ * y_axis_label->v_ * 0.5;
      //y_axis_label->draw(y_axis_point->x - xoff, y_axis_point->y, sx, sy);
      //y_axis_label->draw(y_axis_point->x - xoff, y_axis_point->y, sx, sy, LabelTex::ROTATE_LEFT);
      y_axis_label->draw(y_axis_point->x - yoff, y_axis_point->y - xoff/2, sx,
			sy, LabelTex::ROTATE_LEFT);
    }

    if (x_trend_m_label) {
      glColor4f(1.0, 1.0, 0.0, 1.0);
      float xoff = x_trend_m_label->tex_width_ * x_trend_m_label->u_;
      float yoff = x_trend_m_label->tex_height_ * x_trend_m_label->v_ * 0.5;
      x_trend_m_label->draw(x_trend_m_point->x - xoff/2, x_trend_m_point->y + yoff, sx, sy);
    }

    if (y_trend_m_label) {
      glColor4f(1.0, 1.0, 0.0, 1.0);
      float xoff = y_trend_m_label->tex_width_ * y_trend_m_label->u_;
      float yoff = y_trend_m_label->tex_height_ * y_trend_m_label->v_ * 0.5;
      //y_trend_m_label->draw(y_trend_m_point->x, y_trend_m_point->y, sx, sy);
      y_trend_m_label->draw(y_trend_m_point->x + yoff, y_trend_m_point->y + xoff/2, sx, sy, LabelTex::ROTATE_RIGHT);
    }

    if (x_trend_h_label) {
      glColor4f(0.0, 1.0, 0.0, 1.0);
      float xoff = x_trend_h_label->tex_width_ * x_trend_h_label->u_;
      float yoff = x_trend_h_label->tex_height_ * x_trend_h_label->v_ * 0.5;
      x_trend_h_label->draw(x_trend_h_point->x - xoff, x_trend_h_point->y + yoff, sx, sy);
    }

    if (y_trend_h_label) {
      glColor4f(0.0, 1.0, 0.0, 1.0);
      //float xoff = y_trend_h_label->tex_width_ * y_trend_h_label->u_;
      float yoff = y_trend_h_label->tex_height_ * y_trend_h_label->v_ * 0.5;
      //y_trend_h_label->draw(y_trend_h_point->x, y_trend_h_point->y - yoff, sx, sy);
      y_trend_h_label->draw(y_trend_h_point->x + yoff, y_trend_h_point->y, sx, sy, LabelTex::ROTATE_RIGHT);
    }

    if (x_trend_l_label) {
      glColor4f(1.0, 0.0, 0.0, 1.0);
      //float xoff = x_trend_l_label->tex_width_ * x_trend_l_label->u_;
      float yoff = x_trend_l_label->tex_height_ * x_trend_l_label->v_ * 0.5;
      x_trend_l_label->draw(x_trend_l_point->x, x_trend_l_point->y + yoff, sx, sy);
    }

    if (y_trend_l_label) {
      glColor4f(1.0, 0.0, 0.0, 1.0);
      float xoff = y_trend_l_label->tex_width_ * y_trend_l_label->u_;
      float yoff = y_trend_l_label->tex_height_ * y_trend_l_label->v_ * 0.5;
      y_trend_l_label->draw(y_trend_l_point->x + yoff, y_trend_l_point->y + xoff, sx, sy, LabelTex::ROTATE_RIGHT);
    }

    if (time_label) {
      FreeTypeFace *font = fonts_["anatomical"];
      font->set_points(18.0 * gui_font_scale_.get());
      time_label->set(time_text);
      time_label->bind(font);
                                                                                
      float yoff = time_label->tex_height_ * time_label->v_ * 1.5;
      float xoff = time_label->tex_width_ * time_label->u_;
                                                                                
      glColor4f(1.0, 1.0, 1.0, 1.0);
      time_label->draw(w - x_margin - xoff, h - yoff, sx, sy);
    }

    if (data2_.get_rep()) {
      float *dat2 = (float *)data2_->nrrd->data;
      float yoff(0.0);

      float lv_inj = dat2[cur_idx_*data2_->nrrd->axis[0].size + LV_INJURY_IDX];
      float rv_inj = dat2[cur_idx_*data2_->nrrd->axis[0].size + RV_INJURY_IDX];
      float lv_pwr = dat2[cur_idx_*data2_->nrrd->axis[0].size + LV_POWER_IDX];
      float rv_pwr = dat2[cur_idx_*data2_->nrrd->axis[0].size + RV_POWER_IDX];
      float ttd = dat2[cur_idx_*data2_->nrrd->axis[0].size + TIME_TO_DEATH_IDX];
      float pos = dat2[cur_idx_*data2_->nrrd->axis[0].size + PROB_OF_SURV_IDX];
      lv_inj *= 100;
      rv_inj *= 100;
      lv_pwr *= 100;
      rv_pwr *= 100;
      pos *= 100;
      float prob = 100 - pos;

      FreeTypeFace *font = fonts_["anatomical"];
      font->set_points(18.0 * gui_font_scale_.get());

      yoff += y_margin;

      if (status_label3) {
        ostringstream stat;

        stat << "Power Loss: ";

        if (lv_inj > rv_inj) {
          stat << "LV (" << lv_pwr << "%)";
          stat << ", RV (" << rv_pwr << "%)";
        } else {
          stat << "RV (" << rv_pwr << "%)";
          stat << ", LV (" << lv_pwr << "%)";
        }

        status_label3->set(stat.str());
        status_label3->bind(font);

        if (lv_pwr == lv_pwr && rv_pwr == rv_pwr) {
          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDisable(GL_BLEND);
          status_label3->draw(x_margin, yoff, sx, sy);
          //glEnable(GL_BLEND);
        }

        yoff += status_label3->tex_height_*status_label3->v_ + y_margin;
      }

      if (status_label2) {
        ostringstream stat;

        //string hit = (lv_inj > rv_inj)?"Left":"Right";
        stat << "Injury: ";

        if (lv_inj > rv_inj) {
          stat << "LV (" << lv_inj << "% prob.)";
          stat << ", RV (" << rv_inj << "% prob.)";
        } else {
          stat << "RV (" << rv_inj << "% prob.)";
          stat << ", LV (" << lv_inj << "% prob.)";
        }

        status_label2->set(stat.str());
        status_label2->bind(font);

        if (lv_inj == lv_inj && rv_inj == rv_inj) {
          glColor4f(1.0, 1.0, 1.0, 1.0);
          //glDisable(GL_BLEND);
          status_label2->draw(x_margin, yoff, sx, sy);
          //glEnable(GL_BLEND);
        }

        yoff += status_label2->tex_height_*status_label2->v_ + y_margin + 2;
      }

      if (status_label1a && status_label1b && status_label1c && status_label1d) {
        //float prob = 100 - pos;

        //ostringstream stata;
        //stata << "Estimated time to death: ";
        //status_label1a->set(stata.str());
        //status_label1a->bind(font);

        ostringstream statb;
		  if (prob >= 50) 
          statb << " " << ttd << " mins.";
		  else
          statb << " NA.";
        status_label1b->set(statb.str());
        status_label1b->bind(font);

        //ostringstream statc;
        //statc << " Prob. of Death: ";
        //status_label1c->set(statc.str());
        //status_label1c->bind(font);

        ostringstream statd;
        statd << " " << prob << "%";
        status_label1d->set(statd.str());
        status_label1d->bind(font);

        float xoff(x_margin);
        if (ttd == ttd && pos == pos) {
          glColor4f(1.0, 1.0, 1.0, 1.0);
          status_label1a->draw(xoff, yoff, sx, sy);

          xoff += status_label1a->tex_width_ * status_label1a->u_;

          if (ttd < 20 && prob >= 50)
            glColor4f(1.0, 0.0, 0.0, 1.0);
          else
            glColor4f(1.0, 1.0, 1.0, 1.0);

          status_label1b->draw(xoff, yoff, sx, sy);

          xoff += status_label1b->tex_width_ * status_label1b->u_;

          glColor4f(1.0, 1.0, 1.0, 1.0);
          status_label1c->draw(xoff, yoff, sx, sy);

          xoff += status_label1c->tex_width_ * status_label1c->u_;

          if (prob > 80)
            glColor4f(1.0, 0.0, 0.0, 1.0);
          else
            glColor4f(1.0, 1.0, 1.0, 1.0);

          status_label1d->draw(xoff, yoff, sx, sy);
        }
      }
    }
  glPopMatrix();

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

  glShadeModel(GL_FLAT);
}

void 
ExecutiveState::setTimeLabel()
{
    int hrs, min, sec;
    bool neg;

    int val = (int)(cur_idx_ / gui_sample_rate_.get());
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
ExecutiveState::addMarkersToMenu()
{
  int value;
  hash_map<int, string> tmpmkrs;
  set<int> keys;
  string val;

  for (unsigned int c = 0; c < data_->nproperties(); c++) {
     string name = data_->get_property_name(c);

	  if (string(name, 0, 4) == "MKR_") {
     //data_->get_property(name, value);
     data_->get_property(name, val);

     stringstream ss(val);
     ss >> value;

     keys.insert(value);
     //tmpmkrs[value] = name;
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
ExecutiveState::getNrrd1KeyValues()
{
  char *name = nrrdKeyValueGet(data_->nrrd, "name");

  if (name != NULL) {
    string title(name);
    //gui->execute(id + " setWindowTitle {Decision Space: " + name + "}");

    ostringstream titlestr;
    titlestr << "Decision Space: " << title;

    gui->execute(id + " setWindowTitle {" + titlestr.str().c_str() + "}");

    name_text.replace(0, name_text.length(), titlestr.str());
                                                                                
    plots_dirty_ = true;
  }
}

void
ExecutiveState::createDecisionSpaceArrays()
{
  if (! decision_space_dirty_)
    return;

  decision_space_dirty_ = false;

  if (data_.get_rep()) {
    float *dat = (float *)data_->nrrd->data;

    glEnableClientState(GL_VERTEX_ARRAY);
    //glVertexPointer(3, GL_FLOAT, 0, dat);

	 //if (color_dat) delete[] color_dat;
    //int size = data_->nrrd->axis[1].size;
    //color_dat = scinew GLfloat[size * 4];

    //int inj = gui_injury_offset_.get() * 100 * 4;
    //int falrm = alarm_offset_ * 100 * 4;
    //for (int i = 0; i < size * 4; i+=4) {
       //if (i <= inj) {
       //  color_dat[i] = 0.0;
       //  color_dat[i+1] = 1.0;
       //  color_dat[i+2] = 0.0;
       //} else if (i <= falrm) {
       //  color_dat[i] = 0.0;
       //  color_dat[i+1] = 0.0;
       //  color_dat[i+2] = 1.0;
       //} else {
       //  color_dat[i] = 1.0;
       //  color_dat[i+1] = 0.0;
       //  color_dat[i+2] = 0.0;
       //}
       //color_dat[i+3] = 1.0;

       //color_dat[i]   = 1.0;
       //color_dat[i+1] = 1.0;
       //color_dat[i+2] = 1.0;
       //color_dat[i+3] = 1.0;
    //}

    glEnableClientState(GL_COLOR_ARRAY);
    //glColorPointer(4, GL_FLOAT, 0, color_dat);

	 glInterleavedArrays(GL_C3F_V3F, 0, dat);
  }

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
ExecutiveState::execute()
{
  TimeIPort *time_port = (TimeIPort*)get_iport("Time");
  
  if (!time_port)
  {
    error("Unable to initialize iport Time.");
	 return;
  }
  
  time_port->get(time_viewer_h_);
  if (time_viewer_h_.get_rep() == 0)
  {
    error("No data in the Time port.  It is required.");
	 return;
  }

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

  getNrrd1KeyValues();

  plots_dirty_ = true;
  decision_space_dirty_ = true;

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
  
  if (!runner_) {
    //runner_ = scinew RTDraw2(this);
    runner_ = scinew RTDraw2(this, time_viewer_h_);
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

      if (args[3] == "start") {
        translate_start(x, y);
      } else if (args[3] == "move") {
        translate_motion(x, y);
      } else { // end
        translate_end(x, y);
      }

    } else if (args[2] == "reset") {
      pan_x_ = -X_AXIS_LENGTH/2;
      pan_y_ = -Y_AXIS_LENGTH/2;
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
    setTimeLabel();
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
      //gui->unlock();
    } else {
      width_ = Tk_Width(tkwin);
      height_ = Tk_Height(tkwin);

      //setup_gl_view();
		redraw_all();
    }

  } else {
    Module::tcl_command(args, userdata);
  }
}



} // End namespace VS
