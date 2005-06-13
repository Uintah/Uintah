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
//    File   : StatForecast.cc
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

#define X_AXIS_LENGTH (GLdouble)5.0
#define Y_AXIS_LENGTH (GLdouble)5.0
//#define ALARM_OFFSET 5880

#define TIME_IDX 0
#define PROB_OF_LV_INJURY_IDX 1
#define PROB_OF_RV_INJURY_IDX 2
#define TIME_TO_DEATH_IDX 3
#define PROB_OF_SURV_IDX 4
#define ABP_POWER_IDX 5
#define LVP_POWER_IDX 6
//#define VECTOR_IDX 7
#define ALARM_IDX 11
//#define FIRST_ALARM_IDX 10

//#define VM_SLIDING 0
//#define VM_PATH 1

typedef struct {GLdouble x; GLdouble y; GLdouble z;} tuple;

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;


namespace VS {
using namespace SCIRun;
using std::cerr;
using std::endl;

class RTDraw3;

class StatForecast : public Module
{
public:
  StatForecast(GuiContext* ctx);
  virtual ~StatForecast();
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
  GuiInt                               gui_play_mode_;
  GuiInt                               gui_time_markers_mode_;
  GuiInt                               gui_selected_marker_;
  GuiDouble                            gui_font_scale_;
  GuiInt                               gui_show_name_;
  GuiInt                               gui_show_alarm_;
  GuiInt                               gui_injury_offset_;
  GuiInt                               gui_threshold_squ_;
  GuiInt                               gui_threshold_sod_;
  GuiInt                               gui_threshold_dqu_;
  GuiInt                               gui_threshold_ttd_;
  GuiInt											gui_geom_;

  GLXContext                            ctx_;
  Display*                              dpy_;
  Window                                win_;
  int                                   width_;
  int                                   height_;
  FreeTypeLibrary *	                freetype_lib_;
  map<string, FreeTypeFace *>		fonts_;
  RTDraw3 *		                runner_;
  Thread *		                runner_thread_;
  //! 0-9 digit textures.
  vector<LabelTex>                      digits_;
  bool                                  dig_init_;
  NrrdDataHandle                        data_;
  vector<int>                           markers_;
  int                                   cur_idx_;
  bool                                  plots_dirty_;
  bool                                  decision_space_dirty_;
  GLuint				CONTROL_SPACE_LIST;
  enum Textures {
  	WhiteSquare,
	YellowQuestionSquare,
	GreenCircle,
	GreenQuestionCircle,
	YellowTriangle, 
	YellowQuestionTriangle,
	RedOctagon,
	RedQuestionOctagon,
	GreenECircle,
	GreenQuestionECircle,
	YellowETriangle,
	YellowQuestionETriangle,
	RedEOctagon,
	RedQuestionEOctagon
  };
  GLuint texName[14];
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
  LabelTex				*forecast_label1;
  LabelTex				*forecast_label2;
  string				name_text;
  LabelTex				*time_label;
  string				time_text;
  //int 					injury_offset_;
  //int 					alarm_offset_;
  //bool					alarm_now;
  LabelTex				*alarm_label;
  tuple					*forecast_point;
  TimeViewerHandle	time_viewer_h_;

  bool                  make_current();
  void                  init_plots();
  void                  draw_plots();
  void			setup_gl_view();
  static unsigned int	pow2(const unsigned int);
  void 			setTimeLabel();
  void 			addMarkersToMenu();
  void 			getNrrd1KeyValues();
  void 			createDecisionSpaceArrays();
  void 			createTextureMap(string path, int width, int height, Textures tex);
  void 			translate_start(int x, int y);
  void 			translate_motion(int x, int y);
  void 			translate_end(int x, int y);
  void 			scale_start(int x, int y);
  void 			scale_motion(int x, int y);
  void 			scale_end(int x, int y);
};


class RTDraw3 : public Runnable {
public:
  //RTDraw3(StatForecast* module) : 
  RTDraw3(StatForecast* module, TimeViewerHandle tvh) : 
    module_(module), 
    throttle_(), 
	 tvh_(tvh),
    dead_(0) 
  {};
  virtual ~RTDraw3();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  StatForecast *	module_;
  TimeThrottle	throttle_;
  TimeViewerHandle	tvh_;
  bool		dead_;
};

RTDraw3::~RTDraw3()
{
}

void
RTDraw3::run()
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
StatForecast::LabelTex::set(string s)
{
  text_.replace(0, text_.length(), s);
}

void
StatForecast::LabelTex::draw(float px, float py, float sx, float sy)
{
    StatForecast::LabelTex::draw(px, py, sx, sy, LabelTex::NORMAL);
}

void 
StatForecast::LabelTex::draw(float px, float py, float sx, float sy, float rotation)
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
StatForecast::LabelTex::bind(FreeTypeFace *font) 
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


DECLARE_MAKER(StatForecast)

StatForecast::StatForecast(GuiContext* ctx) :
  Module("StatForecast", ctx, Filter, "Render", "VS"),
  gui_time_(ctx->subVar("time")),
  gui_sample_rate_(ctx->subVar("sample_rate")),
  gui_sweep_speed_(ctx->subVar("sweep_speed")),
  gui_play_mode_(ctx->subVar("play_mode")),
  gui_time_markers_mode_(ctx->subVar("time_markers_mode")),
  gui_selected_marker_(ctx->subVar("selected_marker")),
  gui_font_scale_(ctx->subVar("font_scale")),
  gui_show_name_(ctx->subVar("show_name")),
  gui_show_alarm_(ctx->subVar("show_alarm")),
  gui_injury_offset_(ctx->subVar("injury_offset")),
  gui_threshold_squ_(ctx->subVar("threshold_squ")),
  gui_threshold_sod_(ctx->subVar("threshold_sod")),
  gui_threshold_dqu_(ctx->subVar("threshold_dqu")),
  gui_threshold_ttd_(ctx->subVar("threshold_ttd")),
  gui_geom_(ctx->subVar("geom")),
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
  markers_(0),
  cur_idx_(0),
  plots_dirty_(true),
  decision_space_dirty_(true),
  //texName(0),
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
  forecast_label1(0),
  forecast_label2(0),
  name_text(" "),
  time_label(0),
  time_text("Time: 00:00:00"),
  //injury_offset_(0),
  //alarm_offset_(ALARM_OFFSET),
  //alarm_now(0),
  alarm_label(0),
  forecast_point(0)
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

      //fonts_["default"] = freetype_lib_->load_face(sdir+"/scirun.ttf");
      fonts_["default"] = freetype_lib_->load_face(sdir+"/VS/LucidaSansRegular.ttf");
      fonts_["default_bold"] = freetype_lib_->load_face(sdir+"/VS/LucidaSansDemiBold.ttf");
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

  forecast_point = new tuple;

}

StatForecast::~StatForecast()
{
  if (runner_thread_) {
    runner_->set_dead(true);
    runner_thread_->join();
    runner_thread_ = 0;
  }
}

void 
StatForecast::inc_time(double elapsed)
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
StatForecast::make_current()
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
StatForecast::pow2(const unsigned int dim) {
	unsigned int val = 1;
	while (val < dim) { val = val << 1; };
	return val;
}

void 
StatForecast::init_plots()
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

  if (forecast_label1) delete forecast_label1;
  forecast_label1 = scinew LabelTex(" ");
  forecast_label1->bind(font);

  if (forecast_label2) delete forecast_label2;
  forecast_label2 = scinew LabelTex(" ");
  forecast_label2->bind(font);

  font = fonts_["default_bold"];
  font->set_points(36.0 * gui_font_scale_.get());
  if (alarm_label) delete alarm_label;
  alarm_label = scinew LabelTex("Alarm");
  alarm_label->bind(font);
}

void 
StatForecast::draw_plots()
{
	//cerr << "Draw Plots" << endl;
	reset_vars();

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

	if (data_.get_rep()) {
		float *dat2 = (float *)data_->nrrd->data;

		int nrrd_idx_ = cur_idx_ * data_->nrrd->axis[0].size;

		bool alarm_now = ((int)dat2[nrrd_idx_ + ALARM_IDX] == 1);
		float file_time = dat2[nrrd_idx_ + TIME_IDX];
		float lv_inj = dat2[nrrd_idx_ + PROB_OF_LV_INJURY_IDX];
		float rv_inj = dat2[nrrd_idx_ + PROB_OF_RV_INJURY_IDX];
		float lvp_pwr = dat2[nrrd_idx_ + LVP_POWER_IDX];
		float abp_pwr = dat2[nrrd_idx_ + ABP_POWER_IDX];
		float ttd    = dat2[nrrd_idx_ + TIME_TO_DEATH_IDX];
		float pos    = dat2[nrrd_idx_ + PROB_OF_SURV_IDX];

		lv_inj *= 100;
		rv_inj *= 100;
		lvp_pwr *= 100;
		abp_pwr *= 100;
		pos *= 100;
		float prob = 100 - pos;

		GLuint cur_forecast(0);

		//
		// responds to mouse movement
		//
		glPushMatrix();
			glTranslatef(pan_x_, pan_y_, scale_factor);

			glDisable(GL_TEXTURE_2D);
			//glEnable(GL_POINT_SMOOTH);

			// axes
			//glBegin(GL_LINE_LOOP);
				//glColor4f(1.0, 1.0, 0.0, 1.0);
				//glVertex3f(0.0, 0.0, 0.0);
				//glVertex3f(X_AXIS_LENGTH, 0.0, 0.0);
				//glVertex3f(X_AXIS_LENGTH, Y_AXIS_LENGTH, 0.0);
				//glVertex3f(0.0, Y_AXIS_LENGTH, 0.0);
			//glEnd();

			// forecast indicator
			glEnable(GL_TEXTURE_2D);

			GLuint t_squ = gui_threshold_squ_.get();
			GLuint t_sod = gui_threshold_sod_.get();
			GLuint t_dqu = gui_threshold_dqu_.get();
			GLuint t_ttd = gui_threshold_ttd_.get();

			// no injury
			if (file_time < 0) {
				cur_forecast = WhiteSquare;

			// injury but no valid prob
			} else if (pos != pos) {
				cur_forecast = YellowQuestionSquare;

			// prob between t_sod (50) and t_dqu (60)
			} else if (prob > t_sod && prob < t_dqu) {
				if (alarm_now)
					cur_forecast =
						(ttd > t_ttd)?YellowQuestionETriangle:RedQuestionEOctagon;
				else
					cur_forecast =
						(ttd > t_ttd)?YellowQuestionTriangle:RedQuestionOctagon;

			// prob between t_dqu (60) and 100
			} else if (prob >= t_dqu) {
				if (alarm_now)
					cur_forecast = (ttd > t_ttd)?YellowETriangle:RedEOctagon;
				else
					cur_forecast = (ttd > t_ttd)?YellowTriangle:RedOctagon;

			// prob between t_squ (40) and t_sod (50)
			} else if (prob <= t_sod && prob > t_squ) {
				if (alarm_now)
					cur_forecast = GreenQuestionECircle;
				else
					cur_forecast = GreenQuestionCircle;

			// prob between 0 and t_squ (40)
			} else if (prob <= t_squ) {
				if (alarm_now)
					cur_forecast = GreenECircle;
				else
					cur_forecast = GreenCircle;
			}

			glBindTexture(GL_TEXTURE_2D, texName[cur_forecast]);
			glBegin(GL_QUADS);
				glColor4f(1.0, 1.0, 1.0, 1.0);
				glTexCoord2f(0.0, 1.0); glVertex3f(0.0, 0.0, 0.0);
				glTexCoord2f(1.0, 1.0); glVertex3f(X_AXIS_LENGTH, 0.0, 0.0);
				glTexCoord2f(1.0, 0.0); glVertex3f(X_AXIS_LENGTH, Y_AXIS_LENGTH,
					0.0);
				glTexCoord2f(0.0, 0.0); glVertex3f(0.0, Y_AXIS_LENGTH, 0.0);
			glEnd();
			glDisable(GL_TEXTURE_2D);

			// alarm flash
			if (alarm_now && gui_show_alarm_.get()) {
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
					glLoadIdentity();
					glMatrixMode(GL_PROJECTION);
					glPushMatrix();
						glLoadIdentity();
						//glColor4f(1.0, 0.0, 0.0, (alarm_now) ? 0.5 : 0.0);
						glColor4f(1.0, 0.0, 0.0, 0.6);
						glRectf(-1, -1, 1, 1);
					glPopMatrix();
					glMatrixMode(GL_MODELVIEW);
				glPopMatrix();
			}

			// text location
			GLint vp[4];
			GLdouble mm[16], pm[16];
			glGetIntegerv(GL_VIEWPORT, vp);
			glGetDoublev(GL_MODELVIEW_MATRIX, mm);
			glGetDoublev(GL_PROJECTION_MATRIX, pm);

			gluProject(X_AXIS_LENGTH, Y_AXIS_LENGTH/2, (GLdouble)0.0, mm, pm, vp,
				&forecast_point->x, &forecast_point->y, &forecast_point->z);

			//glDisable(GL_POINT_SMOOTH);
			glEnable(GL_TEXTURE_2D);
		glPopMatrix();

		//
		// text layer
		//
		float x_margin = 5.0;
		float y_margin = 7.0;
		glPushMatrix();
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glScaled(2.0, 2.0, 2.0);
			glTranslated(-.5, -.5, -.5);

			if (alarm_label && alarm_now && gui_show_alarm_.get()) {
				float xoff = alarm_label->tex_width_ * alarm_label->u_;
				float yoff = alarm_label->tex_height_ * alarm_label->v_ * 1.5;
				//alarm_label->draw(w - xoff - x_margin, h - yoff, sx, sy);
				//alarm_label->draw(w/2 - xoff/2, h - yoff, sx, sy);
				glColor4f(1.0, 1.0, 1.0, 0.6);
				alarm_label->draw(w/2 - xoff/2, h/2 - yoff*0.5, sx, sy);
			}

			if (name_label && gui_show_name_.get()) {
				float yoff = name_label->tex_height_ * name_label->v_ * 1.5;
				glColor4f(1.0, 1.0, 1.0, 1.0);
				//(alarm_now)?glDisable(GL_BLEND):glEnable(GL_BLEND);
				name_label->draw(x_margin, h - yoff, sx, sy);
				//(alarm_now)?glEnable(GL_BLEND):glDisable(GL_BLEND);
			}

			if (forecast_label1 && forecast_label2) {
				FreeTypeFace *font = fonts_["anatomical"];
				font->set_points(24.0 * gui_font_scale_.get());

				ostringstream info1;
				ostringstream info2;

				switch (cur_forecast) {
					case WhiteSquare:
						glColor4f(0.772, 0.772, 0.772, 1.0);
						info1 << "No injury";
						info2 << " ";
						break;
					case YellowQuestionSquare:
						glColor4f(1.0, 1.0, 0.0, 1.0);
						info1 << "Acquiring/analyzing data";
						info2 << " ";
						break;
					case GreenCircle:
					case GreenECircle:
						glColor4f(0.0, 1.0, 0.0, 1.0);
						info1 << pos << "% probability of survival";
						info2 << " ";
						break;
					case GreenQuestionCircle:
					case GreenQuestionECircle:
						glColor4f(0.0, 1.0, 0.0, 1.0);
						info1 << pos << "% probability of survival";
						info2 << " ";
						break;
					case YellowTriangle:
					case YellowETriangle:
						glColor4f(1.0, 1.0, 0.0, 1.0);
						info1 << ttd << " mins. before death (est.)";
						info2 << prob << "% probability of death";
						break;
					case YellowQuestionTriangle:
					case YellowQuestionETriangle:
						glColor4f(1.0, 1.0, 0.0, 1.0);
						info1 << prob << "% probability of death";
						info2 << " ";
						break;
					case RedOctagon:
					case RedEOctagon:
						glColor4f(1.0, 0.0, 0.0, 1.0);
						info1 << ttd << " mins. before death (est.)";
						info2 << prob << "% probability of death";
						break;
					case RedQuestionOctagon:
					case RedQuestionEOctagon:
						glColor4f(1.0, 0.0, 0.0, 1.0);
						info1 << prob << "% probability of death";
						info2 << " ";
						break;
					default:
						glColor4f(1.0, 1.0, 1.0, 1.0);
						info1 << " ";
						info2 << " ";
						break;
				}

				forecast_label1->set(info1.str());
				forecast_label1->bind(font);

				forecast_label2->set(info2.str());
				forecast_label2->bind(font);

				//float xoff = forecast_label1->tex_width_ * forecast_label1->u_;
				float yoff = forecast_label1->tex_height_*forecast_label1->v_;
				forecast_label1->draw(forecast_point->x + yoff*0.5,
					forecast_point->y, sx, sy);
				//forecast_label1->draw(forecast_point->x + yoff,
				//	forecast_point->y + xoff/2, sx, sy, LabelTex::ROTATE_RIGHT);
				forecast_label2->draw(forecast_point->x + yoff*0.5,
					forecast_point->y-yoff-y_margin, sx, sy);
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

			float yoff(0.0);

			FreeTypeFace *font = fonts_["anatomical"];
			font->set_points(18.0 * gui_font_scale_.get());

			yoff += y_margin;

			if (status_label3) {
				ostringstream stat;

				stat << "Power Loss: ";

				if (lv_inj > rv_inj) {
					stat << "LVP (" << lvp_pwr << "%)";
					//stat << ", RV (" << abp_pwr << "%)";
					stat << ", ABP (" << abp_pwr << "%)";
				} else {
					//stat << "RV (" << abp_pwr << "%)";
					stat << "ABP (" << abp_pwr << "%)";
					stat << ", LVP (" << lvp_pwr << "%)";
				}

				status_label3->set(stat.str());
				status_label3->bind(font);

				if (lvp_pwr == lvp_pwr && abp_pwr == abp_pwr) {
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
				if (prob >= t_sod) 
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
StatForecast::redraw_all()
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
StatForecast::setup_gl_view()
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
StatForecast::setTimeLabel()
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
StatForecast::addMarkersToMenu()
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
StatForecast::getNrrd1KeyValues()
{
  char *name = nrrdKeyValueGet(data_->nrrd, "name");

  if (name != NULL) {
    string title(name);
    //gui->execute(id + " setWindowTitle {Decision Space: " + name + "}");
	 gui->execute(id + " setWindowTitle {Statistical Forecast: " + name + "}");

    ostringstream titlestr;
    titlestr << "Name: " << title;

    //gui->execute(id + " setWindowTitle {" + titlestr.str().c_str() + "}");

    name_text.replace(0, name_text.length(), titlestr.str());
                                                                                
    plots_dirty_ = true;
  }
}

void
StatForecast::createTextureMap(string path, int width, int height, Textures tex)
{
  GLubyte *data;
  FILE *file;

  file = fopen(path.c_str(), "rb");
  if (file == NULL) cerr << "Can't open file: " << path << endl;
  data = (GLubyte *)malloc(width*height*3);
  if (fread(data, sizeof(GLubyte), width*height*3, file) == 0) {
     cerr << "Can't read file: " << path << endl;
  }
  fclose(file);

  glGenTextures(1, &texName[tex]);
  glBindTexture(GL_TEXTURE_2D, texName[tex]);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
  free(data);
}

void
StatForecast::createDecisionSpaceArrays()
{
  if (! decision_space_dirty_)
    return;

  decision_space_dirty_ = false;

  string images = string(sci_getenv("SCIRUN_SRCDIR")) + string("/pixmaps/VS/");
  string path;

  path = images + string("whitesquare.raw");
  createTextureMap(path, 256, 256, WhiteSquare);

  path = images + string("yellowquestionsquare.raw");
  createTextureMap(path, 256, 256, YellowQuestionSquare);

  path = images + string("greencircle.raw");
  createTextureMap(path, 256, 256, GreenCircle);

  path = images + string("greenquestioncircle.raw");
  createTextureMap(path, 256, 256, GreenQuestionCircle);

  path = images + string("yellowtriangle.raw");
  createTextureMap(path, 256, 256, YellowTriangle);

  path = images + string("yellowquestiontriangle.raw");
  createTextureMap(path, 256, 256, YellowQuestionTriangle);

  path = images + string("redoctagon.raw");
  createTextureMap(path, 256, 256, RedOctagon);

  path = images + string("redquestionoctagon.raw");
  createTextureMap(path, 256, 256, RedQuestionOctagon);

  path = images + string("greenecircle.raw");
  createTextureMap(path, 256, 256, GreenECircle);

  path = images + string("greenecircle.raw");
  createTextureMap(path, 256, 256, GreenQuestionECircle);

  path = images + string("yellowetriangle.raw");
  createTextureMap(path, 256, 256, YellowETriangle);

  path = images + string("yellowetriangle.raw");
  createTextureMap(path, 256, 256, YellowQuestionETriangle);

  path = images + string("redeoctagon.raw");
  createTextureMap(path, 256, 256, RedEOctagon);

  path = images + string("redeoctagon.raw");
  createTextureMap(path, 256, 256, RedQuestionEOctagon);
}

void
StatForecast::translate_start(int x, int y)
{
  mouse_last_x_ = x;
  mouse_last_y_ = y;
}

void
StatForecast::translate_motion(int x, int y)
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
StatForecast::translate_end(int x, int y)
{
  redraw_all();
}

void
StatForecast::scale_start(int x, int y)
{
  mouse_last_y_ = y;
}

void
StatForecast::scale_motion(int x, int y)
{
  float ymtn = -float(mouse_last_y_ - y) / float(height_);
  mouse_last_y_ = y;

  scale_ = (scale_ + -ymtn);

  //if (scale_ < 0.0) scale_ = 0.0;

  redraw_all();
}

void
StatForecast::scale_end(int x, int y)
{
  redraw_all();
}

void
StatForecast::execute()
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
  }// else {
  	//cerr << "Data Rep" << endl;
  //}

  double rt = data_->nrrd->axis[0].spacing;
  if (rt == rt)
    gui_sample_rate_.set(1/data_->nrrd->axis[0].spacing);

  addMarkersToMenu();

  getNrrd1KeyValues();

  plots_dirty_ = true;
  //decision_space_dirty_ = true;

  if (!runner_) {
    //runner_ = scinew RTDraw3(this);
    runner_ = scinew RTDraw3(this, time_viewer_h_);
    runner_thread_ = scinew Thread(runner_, string(id+" RTDraw3 OpenGL").c_str());
  }
}

void
StatForecast::tcl_command(GuiArgs& args, void* userdata) 
{
  if(args.count() < 2) {
    args.error("StatForecast needs a minor command");
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
