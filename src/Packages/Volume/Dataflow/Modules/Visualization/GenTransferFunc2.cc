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


#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array3.h>

#include <sci_gl.h>
//#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Geom/FragmentProgramARB.h>
#include <Packages/Volume/Dataflow/Ports/Colormap2Port.h>
#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Geom/GeomOpenGL.h>
#include <tcl.h>
#include <tk.h>

#include <iostream>

// tcl interpreter corresponding to this module
extern Tcl_Interp* the_interp;

// the OpenGL context structure
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

using namespace SCIRun;
using namespace SCITeem;

namespace Volume {

class Widget
{
public:
  Widget();
  virtual ~Widget();

  // appearance
  virtual void draw() = 0;
  virtual void rasterize() = 0;
  virtual void rasterize(Array3<float>& array) = 0;
  // behavior
//   virtual void pick (int x, int y) = 0;
//   virtual void move (int x, int y) = 0;
//   virtual void release (int x, int y) = 0;

  // actions
  //virtual void rasterize () = 0;
  
protected:
  Color line_color_;
  float line_alpha_;
  float thin_line_width_;
  float thick_line_width_;
  float point_size_;
  Color color_;
  float alpha_;
};

class TriangleWidget : public Widget
{
public:
  TriangleWidget();
  TriangleWidget(float base, float top_x, float top_y,
                 float width, float bottom);
  ~TriangleWidget();

  // appearance
  void draw();
  void rasterize();
  void rasterize(Array3<float>& array);
  
  static bool Init();
  static void Exit();
  
protected:
  float base_;
  float top_x_, top_y_;
  float width_;
  float bottom_;
  static FragmentProgramARB* shader_;
};

enum RectangleType {
  RECTANGLE_ELLIPSOID,
  RECTANGLE_1D
};

class RectangleWidget : public Widget
{
public:
  RectangleWidget();
  RectangleWidget(RectangleType type, float left_x, float left_y,
                  float width, float height, float offset);
  ~RectangleWidget();

  // appearance
  void draw();
  void rasterize();
  void rasterize(Array3<float>& array);
  
  static bool Init();
  static void Exit();
  
protected:
  RectangleType type_;
  float left_x_, left_y_;
  float width_, height_, offset_;
  static FragmentProgramARB* shader_;
};

FragmentProgramARB* TriangleWidget::shader_ = 0;
FragmentProgramARB* RectangleWidget::shader_ = 0;

const char* TriangleWidgetShader =
"!!ARBfp1.0 \n"
"PARAM color = program.local[0]; \n"
"PARAM geom0 = program.local[1]; # {base, top_x, top_y, 0.0} \n"
"PARAM geom1 = program.local[2]; # {width, bottom, 0.0, 0.0} \n"
"PARAM sz = program.local[3]; # {1/sx, 1/sy, 0.0, 0.0} \n"
"TEMP c, p, t;"
"MUL p.xy, fragment.position.xyyy, sz.xyyy; \n"
"MUL p.z, geom1.y, geom0.z; \n"
"SUB p.z, p.y, p.z; \n"
"KIL p.z; \n"
"RCP t.z, geom0.z; \n"
"MUL t.x, p.y, t.z; \n"
"LRP c.x, t.x, geom0.y, geom0.x; \n"
"MUL c.y, t.x, geom1.x; \n"
"MUL c.y, c.y, 0.5; \n"
"RCP c.y, c.y; \n"
"SUB c.z, p.x, c.x; \n"
"MUL c.z, c.y, c.z; \n"
"ABS c.z, c.z; \n"
"SUB c.z, 1.0, c.z; \n"
"MUL c.w, color.w, c.z; \n"
"MOV c.xyz, color.xyzz; \n"
"MOV result.color, c; \n"
"END";

const char* RectangleWidgetShader1D =
"!!ARBfp1.0 \n"
"PARAM color = program.local[0]; \n"
"PARAM geom0 = program.local[1]; # {left_x, left_y, width, height} \n"
"PARAM geom1 = program.local[2]; # {offset, 1/offset, 1/(1-offset), 0.0} \n"
"PARAM sz = program.local[3]; # {1/sx, 1/sy, 0.0, 0.0} \n"
"TEMP c, p, t; \n"
"MUL p.xy, fragment.position.xyyy, sz.xyyy; \n"
"SUB p.xy, p.xyyy, geom0.xyyy; \n"
"RCP p.z, geom0.z; \n"
"RCP p.w, geom0.w; \n"
"MUL p.xy, p.xyyy, p.zwww; \n"
"SUB t.x, p.x, geom1.x; \n"
"MUL t.y, t.x, geom1.y; \n"
"MUL t.z, t.x, geom1.z; \n"
"CMP t.w, t.y, t.y, t.z; \n"
"ABS t.w, t.w; \n"
"SUB t.w, 1.0, t.w; \n"
"MUL c.w, color.w, t.w; \n"
"MOV c.xyz, color.xyzz; \n"
"MOV result.color, c; \n"
"END";

Widget::Widget()
  : line_color_(0.75, 0.75, 0.75), line_alpha_(1.0), thin_line_width_(0.75),
    thick_line_width_(2.0), point_size_(7.0), color_(0.5, 0.5, 1.0), alpha_(0.7)
{}

Widget::~Widget()
{}

TriangleWidget::TriangleWidget()
  : base_(0.5), top_x_(0.15), top_y_(0.5), width_(0.25), bottom_(0.5)
{}

TriangleWidget::TriangleWidget(float base, float top_x, float top_y,
                               float width, float bottom)
  : base_(base), top_x_(top_x), top_y_(top_y), width_(width), bottom_(bottom)
{}

TriangleWidget::~TriangleWidget()
{}

bool
TriangleWidget::Init()
{
  if (!shader_) {
    shader_ = new FragmentProgramARB(TriangleWidgetShader);
    shader_->create();
    //return true;
  }
  return false;
}

void
TriangleWidget::Exit()
{
  shader_->destroy();
  delete shader_;
  shader_ = 0;
}

void
TriangleWidget::rasterize()
{
#if 0  
  if(shader_->valid()) {
    shader_->bind();
    shader_->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
    shader_->setLocalParam(1, base_, top_x_, top_y_, 0.0);
    shader_->setLocalParam(2, width_, bottom_, 0.0, 0.0);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    shader_->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
    
    glBegin(GL_TRIANGLES);
    {
      glVertex2f(base_, 0.0);
      glVertex2f(top_x_+width_/2, top_y_);
      glVertex2f(top_x_-width_/2, top_y_);
    }
    glEnd();
    shader_->release();
  }
#endif
}

void
TriangleWidget::rasterize(Array3<float>& array)
{
  //std::cerr << tex->size(0) << " " << tex->size(1) << std::endl;
  if(array.dim3() != 4) return;
  int size_x = array.dim2();
  int size_y = array.dim1();
  float top_left = top_x_-width_/2;
  float top_right = top_x_+width_/2;
  int lb = (int)(bottom_*top_y_*size_y);
  int le = (int)(top_y_*size_y);
  //cerr << lb << " | " << le << endl;
  for(int i=lb; i<=le; i++) {
    float fb = (i/(float)le)*top_left + base_;
    float fe = (i/(float)le)*top_right + base_;
    float fm = (i/(float)le)*top_x_ + base_;
    int rb = (int)(fb*size_x);
    int re = (int)(fe*size_x);
    int rm = (int)(fm*size_x);
    float a = 0.0;
    float da = alpha_/(rm-rb);
    for(int j=rb; j<rm; j++, a+=da) {
      array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
      array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
      array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
      array(i,j,3) = array(i,j,3)*(1-a) + a;
    }
    a = alpha_;
    da = alpha_/(re-rm);
    //cerr << mTop.x << " " << fm << " -> " << fe << std::endl;
    for (int j=rm; j<re; j++, a-=da)
    {
      array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
      array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
      array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
      array(i,j,3) = array(i,j,3)*(1-a) + a;
    }
  }
}

void
TriangleWidget::draw()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  glColor4f(line_color_.r(), line_color_.g(), line_color_.b(), line_alpha_);
  glEnable(GL_LINE_SMOOTH);
  glLineWidth(thin_line_width_);
  glBegin(GL_LINES);
  {
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_-width_/2, top_y_);
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_+width_/2, top_y_);
  }
  glEnd();
  glLineWidth(thick_line_width_);
  float b_x = bottom_*top_x_ + base_;
  float b_y = bottom_*top_y_;
  float w = bottom_*width_;
  glBegin(GL_LINES);
  {
    glVertex2f(base_+top_x_-width_/2, top_y_);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    glVertex2f(b_x-w/2, b_y);
    glVertex2f(b_x+w/2, b_y);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glEnable(GL_POINT_SMOOTH);
  glPointSize(point_size_);
  glBegin(GL_POINTS);
  {
    glVertex2f(base_+top_x_+width_/2, top_y_);
    glVertex2f(b_x-w/2, b_y);
  }
  glEnd();
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}


RectangleWidget::RectangleWidget()
  : type_(RECTANGLE_1D), left_x_(0.25), left_y_(0.5), width_(0.25), height_(0.25),
    offset_(0.25)
{}

RectangleWidget::RectangleWidget(RectangleType type, float left_x, float left_y,
                                 float width, float height, float offset)
  : type_(type), left_x_(left_x), left_y_(left_y), width_(width), height_(height),
    offset_(offset)
{}

RectangleWidget::~RectangleWidget()
{}

bool
RectangleWidget::Init()
{
  if (!shader_) {
    shader_ = new FragmentProgramARB(RectangleWidgetShader1D);
    shader_->create();
    //return true;
  }
  return false;
}

void
RectangleWidget::Exit()
{
  shader_->destroy();
  delete shader_;
  shader_ = 0;
}

void
RectangleWidget::rasterize()
{
  shader_->bind();
  shader_->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
  shader_->setLocalParam(1, left_x_, left_y_, width_, height_);
  shader_->setLocalParam(2, offset_, 1/offset_, 1/(1-offset_), 0.0);

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  shader_->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
  
  glBegin(GL_QUADS);
  {
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_+height_);
    glVertex2f(left_x_, left_y_+height_);
  }
  glEnd();
  shader_->release();
}

void
RectangleWidget::rasterize(Array3<float>& array)
{
  if(array.dim3() != 4) return;
  int size_x = array.dim2();
  int size_y = array.dim1();
  float left = left_x_;
  float right = left_x_+width_;
  float bottom = left_y_;
  float top = left_y_+height_;
  left = left < 0.0 ? 0.0 : left;
  right = right > 1.0 ? 1.0 : right;
  bottom = bottom < 0.0 ? 0.0 : bottom;
  top = top > 1.0 ? 1.0 : top;

  int lb = int(bottom*size_y);
  int le = int(top*size_y);
  //int la = int((mBall.y*mSize.y+bottom)*size.y);
  int rb = int(left*size_x);
  int re = int(right*size_x);
  int ra = int((offset_*width_+left)*size_x);
  switch(type_) {
    case RECTANGLE_ELLIPSOID: {
      for(int i=lb; i<=le; i++) {
        for(int j=rb; j<re; j++) {
          float x = j/(float)size_x;
          float y = i/(float)size_y;
          x -= (left+right)/2;
          y -= (bottom+top)/2;
          x *= height_/width_;
          float w = 1-2*sqrt(x*x+y*y)/size_y;
          if (w < 0) w = 0;
          float a = alpha_*w;
          array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
          array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
          array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
          array(i,j,3) = array(i,j,3)*(1-a) + a;
        }
      }
    } break;

    case RECTANGLE_1D: {
      float a = ra <= rb+1 ? alpha_ : 0;
      float da = ra <= rb+1 ? 0.0 : alpha_/(ra-rb-1);
      for(int j=rb; j<ra; j++, a+=da) {
        for(int i=lb; i<le; i++) {
          array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
          array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
          array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
          array(i,j,3) = array(i,j,3)*(1-a) + a;
        }
      }
      a = alpha_;
      da = ra <= re-1 ? alpha_/(re-ra-1) : 0.0;
      for(int j=ra; j<re; j++, a-=da) {
        for(int i=lb; i<le; i++) {
          array(i,j,0) = array(i,j,0)*(1-a) + color_.r();
          array(i,j,1) = array(i,j,1)*(1-a) + color_.g();
          array(i,j,2) = array(i,j,2)*(1-a) + color_.b();
          array(i,j,3) = array(i,j,3)*(1-a) + a;
        }
      }
    } break;

  default:
    break;
  }
}

void
RectangleWidget::draw()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  glColor4f(line_color_.r(), line_color_.g(), line_color_.b(), line_alpha_);
  glEnable(GL_LINE_SMOOTH);
  glLineWidth(thin_line_width_);
  glBegin(GL_LINES);
  {
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_+height_);
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_, left_y_+height_);
  }
  glEnd();
  glLineWidth(thick_line_width_);
  glBegin(GL_LINES);
  {
    glVertex2f(left_x_, left_y_+height_);
    glVertex2f(left_x_+width_, left_y_+height_);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glEnable(GL_POINT_SMOOTH);
  glPointSize(point_size_);
  glBegin(GL_POINTS);
  {
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_+height_);
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_, left_y_+height_);
    glVertex2f(left_x_+offset_*width_, left_y_+height_/2);
  }
  glEnd();
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}


class GenTransferFunc2 : public Module {

  GLXContext ctx_;
  Display* dpy_;
  Window win_;
  int width_, height_;
  int button_;
  Widget* widget_[2];
  //Pbuffer* pbuffer_;

  Nrrd* histo_;
  bool histo_dirty_;
  GLuint histo_tex_;

  Colormap2Handle cmap_;
  bool cmap_dirty_;
  bool cmap_size_dirty_;
  bool cmap_out_dirty_;
  GLuint cmap_tex_;
  
public:
  GenTransferFunc2(GuiContext* ctx);
  virtual ~GenTransferFunc2();

  virtual void execute();

  void tcl_command(GuiArgs&, void*);

  bool create_histo();
  
  void update();
  void redraw();

  void push(int x, int y, int button);
  void motion(int x, int y);
  void release(int x, int y, int button);
};

DECLARE_MAKER(GenTransferFunc2)
GenTransferFunc2::GenTransferFunc2(GuiContext* ctx)
  : Module("GenTransferFunc2", ctx, Filter, "Visualization", "Volume"),
    ctx_(0), dpy_(0), win_(0), button_(0), //pbuffer_(0),
    histo_(0), histo_dirty_(false), histo_tex_(0),
    cmap_(new Colormap2),
    cmap_dirty_(true), cmap_size_dirty_(true), cmap_out_dirty_(true), cmap_tex_(0)
{
  widget_[0] = new TriangleWidget();
  widget_[1] = new RectangleWidget();
}

GenTransferFunc2::~GenTransferFunc2()
{}

void
GenTransferFunc2::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for GenTransferFunc");
    return;
  }

  // mouse commands are motion, down, release
  if (args[1] == "mouse") {
    //cerr << "EVENT: mouse" << endl;
    int x, y, b;
    string_to_int(args[3], x);
    string_to_int(args[4], y);
    if (args[2] == "motion") {
      if (button_ == 0) // not buttons down!
	return;
      motion(x, y);
    } else {
      string_to_int(args[5], b); // which button it was
      if (args[2] == "push") {
	push(x, y, b);
      } else {
	release(x, y, b);
      }
    }
  } else if (args[1] == "resize") {
    //cerr << "EVENT: resize" << endl;
    string_to_int(args[2], width_);
    string_to_int(args[3], height_);
    update();
    redraw();
  } else if (args[1] == "expose") {
    //cerr << "EVENT: expose" << endl;
    update();
    redraw();
  } else if (args[1] == "closewindow") {
    //cerr << "EVENT: close" << endl;
    ctx_ = 0;
  } else {
    Module::tcl_command(args, userdata);
  }
}

void
GenTransferFunc2::motion(int x, int y)
{
  //cerr << "motion: " << x << " " << y << endl;
  update();
  redraw();
}

void
GenTransferFunc2::push(int x, int y, int button)
{
  //cerr << "push: " << x << " " << y << " " << button << endl;
  button_ = button;
  update();
  redraw();
}

void
GenTransferFunc2::release(int x, int y, int button)
{
  //cerr << "release: " << x << " " << y << " " << button << endl;
  button_ = 0;
  update();
  redraw();
}

void
GenTransferFunc2::execute()
{
  //cerr << "execute" << endl;
  
  NrrdIPort* histo_port = (NrrdIPort*)get_iport("Histogram");
  if(histo_port) {
    NrrdDataHandle h;
    histo_port->get(h);
    if(h.get_rep()) {
      if(h->nrrd->dim != 2 && h->nrrd->dim != 3) {
        error("Invalid input dimension.");
      }
      if(histo_ != h->nrrd) {
        histo_ = h->nrrd;
        histo_dirty_ = true;
      }
    } else {
      if(histo_ != 0)
        histo_dirty_ = true;
      histo_ = 0;
    }
  } else {
    if(histo_ != 0)
      histo_dirty_ = true;
    histo_ = 0;
  }

  update();
  redraw();
  
  Colormap2OPort* cmap_port = (Colormap2OPort*)get_oport("Colormap");
  if(cmap_port) {
    cmap_port->send(cmap_);
  }
}

void
GenTransferFunc2::update()
{
  //cerr << "update" << endl;

  bool do_execute = false;
  
  gui->lock();

  //----------------------------------------------------------------
  // obtain rendering ctx 
  if (!ctx_) {
    const string myname(".ui" + id + ".f.gl.gl");
    Tk_Window tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
                                      Tk_MainWindow(the_interp));
    if (!tkwin) {
      warning("Unable to locate window!");
      gui->unlock();
      return;
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
      return;
    }
  }
  glXMakeCurrent(dpy_, win_, ctx_);

  //----------------------------------------------------------------
  // update colormap array
  Array3<float>& cmap = cmap_->array();
  if(width_ != cmap.dim2() || height_ != cmap.dim1()) {
    cmap_size_dirty_ = true;
    cmap_dirty_ = true;
  }

  if(cmap_dirty_) {
    cmap_->lock_array();
    // realloc cmap
    if(cmap_size_dirty_)
      cmap.resize(height_, width_, 4);
    // clear cmap
    for(int i=0; i<cmap.dim1(); i++) {
      for(int j=0; j<cmap.dim2(); j++) {
        cmap(i,j,0) = 0.0;
        cmap(i,j,1) = 0.0;
        cmap(i,j,2) = 0.0;
        cmap(i,j,3) = 0.0;
      }
    }
    // rasterize widgets
    widget_[0]->rasterize(cmap);
    widget_[1]->rasterize(cmap);
    // update textures
    if(cmap_size_dirty_) {
      if(glIsTexture(cmap_tex_)) {
        glDeleteTextures(1, &cmap_tex_);
        cmap_tex_ = 0;
      }
      glGenTextures(1, &cmap_tex_);
      glBindTexture(GL_TEXTURE_2D, cmap_tex_);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cmap.dim2(), cmap.dim1(),
                   0, GL_RGBA, GL_FLOAT, &cmap(0,0,0));
      glBindTexture(GL_TEXTURE_2D, 0);
      cmap_size_dirty_ = false;
    } else {
      glBindTexture(GL_TEXTURE_2D, cmap_tex_);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cmap.dim2(), cmap.dim1(),
                   0, GL_RGBA, GL_FLOAT, &cmap(0,0,0));
      glBindTexture(GL_TEXTURE_2D, 0);
    }
    cmap_dirty_ = false;
    cmap_->unlock_array();
    do_execute = true;
  }

  //----------------------------------------------------------------
  // update histo tex
  if (histo_dirty_) {
    if(glIsTexture(histo_tex_)) {
      glDeleteTextures(1, &histo_tex_);
      histo_tex_ = 0;
    }
    glGenTextures(1, &histo_tex_);
    glBindTexture(GL_TEXTURE_2D, histo_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    int axis_size[3];
    nrrdAxisInfoGet_nva(histo_, nrrdAxisInfoSize, axis_size);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, axis_size[histo_->dim-2],
                 axis_size[histo_->dim-1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                 histo_->data);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  gui->unlock();

  if (do_execute) {
    want_to_execute();
  }
}


void
GenTransferFunc2::redraw()
{
  //cerr << "redraw" << endl;

  gui->lock();
  
  //----------------------------------------------------------------
  // draw
  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  
  glViewport(0, 0, width_, height_);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1.0, -1.0, 0.0);
  glScalef(2.0, 2.0, 2.0);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);

  // draw histo
  if(histo_) {
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, histo_tex_);
    glColor4f(0.7, 0.7, 0.7, 1.0);
    glBegin(GL_QUADS);
    {
      glTexCoord2f( 0.0,  0.0);
      glVertex2f( 0.0,  0.0);
      glTexCoord2f( 1.0,  0.0);
      glVertex2f( 1.0,  0.0);
      glTexCoord2f( 1.0,  1.0);
      glVertex2f( 1.0,  1.0);
      glTexCoord2f( 0.0,  1.0);
      glVertex2f( 0.0,  1.0);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }

  // draw cmap
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, cmap_tex_);
  glBegin(GL_QUADS);
  {
    glTexCoord2f( 0.0,  0.0);
    glVertex2f( 0.0,  0.0);
    glTexCoord2f( 1.0,  0.0);
    glVertex2f( 1.0,  0.0);
    glTexCoord2f( 1.0,  1.0);
    glVertex2f( 1.0,  1.0);
    glTexCoord2f( 0.0,  1.0);
    glVertex2f( 0.0,  1.0);
  }
  glEnd();
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
  
  glDisable(GL_BLEND);
  
  widget_[0]->draw();
  widget_[1]->draw();
  
  glXSwapBuffers(dpy_, win_);
  glXMakeCurrent(dpy_, 0, 0);

  gui->unlock();
}



#if 0  

#if defined(HAVE_GLEW)
  if (!glew_init) {
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err )
    {
      /* problem: glewInit failed, something is seriously wrong */
      fprintf(stderr, "Error: %s\n", glewGetErrorString(err)); 
    }
    fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    glew_init = GL_TRUE;
  }
#endif

  
  // create pbuffer
  if (!pbuffer_
      || pbuffer_->getWidth() != width_
      || pbuffer_->getHeight() != height_) {
    if (pbuffer_) {
      pbuffer_->destroy();
      delete pbuffer_;
    }
    pbuffer_ = new Pbuffer(width_, height_, GL_INT, 8, true, GL_FALSE);
    if (pbuffer_->create()) {
      pbuffer_->destroy();
      delete pbuffer_;
      pbuffer_ = 0;
      return false;
    }
    //cerr << "Using visual: " << pbuffer_->getVisualId() << endl;
  }

  //pbuffer_->makeCurrent();
  
  TriangleWidget::Init();
  RectangleWidget::Init();
#endif


  //pbuffer_->makeCurrent();

  //glClear(GL_COLOR_BUFFER_BIT);

//   glColor3f(1.0, 0.0, 0.0);
//   glBegin(GL_TRIANGLES);
//   {
//     glVertex2f(0.0, 0.0);
//     glVertex2f(1.0, 0.0);
//     glVertex2f(1.0, 1.0);
//   }
//   glEnd();

  //glFlush();

  //update();


  //-------------------------------------------------------
#if 0
  pbuffer_->makeCurrent();
  
  glDrawBuffer(GL_FRONT);
  glClearColor(0.0, 0.0, 0.0, 0.0);

  glViewport(0, 0,width_, height_);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1.0, -1.0, 0.0);
  glScalef(2.0, 2.0, 2.0);

//   glEnable(GL_COLOR_MATERIAL);
//   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDisable(GL_TEXTURE_2D);
#endif


  
#if 0
  pbuffer_->bind(GL_FRONT);
  glBegin(GL_QUADS);
  {
    glTexCoord2f( 0.0,  0.0);
    glVertex2f( 0.0,  0.0);
    glTexCoord2f( 1.0,  0.0);
    glVertex2f( 1.0,  0.0);
    glTexCoord2f( 1.0,  1.0);
    glVertex2f( 1.0,  1.0);
    glTexCoord2f( 0.0,  1.0);
    glVertex2f( 0.0,  1.0);
  }
  glEnd();
  pbuffer_->release(GL_FRONT);
#endif

} // End namespace Volume
