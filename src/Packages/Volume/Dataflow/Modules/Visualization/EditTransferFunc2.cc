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
//    File   : EditTransferFunc2.cc
//    Author : Milan Ikits
//    Author : Michael Callahan
//    Date   : Thu Jul  8 01:50:58 2004

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array3.h>

#include <sci_gl.h>
#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
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


inline double
CLAMP(double x, double l, double u)
{
  if (x < l) return l;
  if (x > u) return u;
  return x;
}


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
  virtual int pick (int x, int y, int w, int h) = 0;
  virtual void move (int obj, int x, int y, int w, int h) = 0;
  virtual void release (int obj, int x, int y, int w, int h) = 0;

  void select(int obj) { selected_ = obj; }
  void unselect_all() { selected_ = 0; }
  
protected:
  void selectcolor(int obj);

  Color line_color_;
  float line_alpha_;
  Color selected_color_;
  float selected_alpha_;
  float thin_line_width_;
  float thick_line_width_;
  float point_size_;
  Color color_;
  float alpha_;
  int selected_;
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
  
  // behavior
  virtual int pick (int x, int y, int w, int h);
  virtual void move (int obj, int x, int y, int w, int h);
  virtual void release (int obj, int x, int y, int w, int h);

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
  
  // behavior
  virtual int pick (int x, int y, int w, int h);
  virtual void move (int obj, int x, int y, int w, int h);
  virtual void release (int obj, int x, int y, int w, int h);

  static bool Init();
  static void Exit();
  
protected:

  RectangleType type_;
  float left_x_, left_y_;
  float width_, height_, offset_;

  // Used by picking.
  float last_x_, last_y_;
  int pick_ix_, pick_iy_;
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
  : line_color_(0.75, 0.75, 0.75),
    line_alpha_(1.0),
    selected_color_(1.0, 0.0, 0.0),
    selected_alpha_(1.0),
    thin_line_width_(0.75),
    thick_line_width_(2.0),
    point_size_(7.0),
    color_(0.5, 0.5, 1.0),
    alpha_(0.7),
    selected_(0)
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
    return shader_->create();
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
  shader_->bind();
  shader_->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
  shader_->setLocalParam(1, base_, base_+top_x_, top_y_, 0.0);
  shader_->setLocalParam(2, width_, bottom_, 0.0, 0.0);

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  shader_->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
    
  glBegin(GL_TRIANGLES);
  {
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    glVertex2f(base_+top_x_-width_/2, top_y_);
  }
  glEnd();
  shader_->release();
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

  glEnable(GL_LINE_SMOOTH);
  glLineWidth(thin_line_width_);
  glBegin(GL_LINES);
  {
    selectcolor(1);
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_-width_/2, top_y_);
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_+width_/2, top_y_);
  }
  glEnd();
  glLineWidth(thick_line_width_);
  const float b_x = bottom_*top_x_ + base_;
  const float b_y = bottom_*top_y_;
  const float w = bottom_*width_;
  glBegin(GL_LINES);
  {
    selectcolor(4);
    glVertex2f(base_+top_x_-width_/2, top_y_);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    selectcolor(1);
    glVertex2f(b_x-w/2, b_y);
    glVertex2f(b_x+w/2, b_y);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glEnable(GL_POINT_SMOOTH);
  glPointSize(point_size_);
  glBegin(GL_POINTS);
  {
    selectcolor(2);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    selectcolor(3);
    glVertex2f(b_x-w/2, b_y);
  }
  glEnd();
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}


int
TriangleWidget::pick (int ix, int iy, int sw, int sh)
{
  const double x = ix / (double)sw;
  const double y = iy / (double)sh;
  const double xeps = point_size_ / sw * 0.5;
  const double yeps = point_size_ / sh * 0.5;

  const float b_x = bottom_*top_x_ + base_;
  const float b_y = bottom_*top_y_;
  const float w = bottom_*width_;

  const double cp1x = base_ + top_x_ + width_/2;
  const double cp1y = top_y_;
  if (fabs(x - cp1x) < xeps &&
      fabs(y - cp1y) < yeps)
  {
    return 2;
  }

  const double cp2x = b_x - w/2;
  const double cp2y = b_y;
  if (fabs(x - cp2x) < xeps &&
      fabs(y - cp2y) < yeps)
  {
    return 3;
  }
  
  return 0;
}


void
TriangleWidget::move (int obj, int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  
  switch (selected_)
  {
  case 2:
    width_ = (x - top_x_ - base_) * 2.0;
    break;

  case 3:
    bottom_ = y / top_y_;
    break;
  }
}


void
TriangleWidget::release (int obj, int x, int y, int w, int h)
{
  move(obj, x, y, w, h);
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
    return shader_->create();
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
Widget::selectcolor(int obj)
{
  if (selected_ == obj)
  {
    glColor4f(selected_color_.r(), selected_color_.g(),
	      selected_color_.b(), selected_alpha_);
  }
  else
  {
    glColor4f(line_color_.r(), line_color_.g(), line_color_.b(), line_alpha_);
  }
}


void
RectangleWidget::draw()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  selectcolor(1);

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
    selectcolor(2);
    glVertex2f(left_x_, left_y_);
    selectcolor(3);
    glVertex2f(left_x_+width_, left_y_);
    selectcolor(4);
    glVertex2f(left_x_+width_, left_y_+height_);
    selectcolor(5);
    glVertex2f(left_x_, left_y_+height_);
    selectcolor(6);
    glVertex2f(left_x_+offset_*width_, left_y_+height_*0.5);
  }
  glEnd();
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}


int
RectangleWidget::pick (int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  const double xeps = point_size_ / w * 0.5;
  const double yeps = point_size_ / h * 0.5;

  if (fabs(x - left_x_) < xeps &&
      fabs(y - left_y_) < yeps)
  {
    return 2;
  }

  if (fabs(x - left_x_ - width_) < xeps &&
      fabs(y - left_y_) < yeps)
  {
    return 3;
  }

  if (fabs(x - left_x_ - width_) < xeps &&
      fabs(y - left_y_ - height_) < yeps)
  {
    return 4;
  }

  if (fabs(x - left_x_) < xeps &&
      fabs(y - left_y_ - height_) < yeps)
  {
    return 5;
  }

  if (fabs(x - left_x_ - offset_ * width_) < xeps &&
      fabs(y - left_y_ - height_ * 0.5) < yeps)
  {
    return 6;
  }

  if (x > Min(left_x_, left_x_ + width_) &&
      x < Max(left_x_, left_x_ + width_) &&
      y > Min(left_y_, left_y_ + height_) &&
      y < Max(left_y_, left_y_ + height_))
  {
    last_x_ = left_x_;
    last_y_ = left_y_;
    pick_ix_ = ix;
    pick_iy_ = iy;
    return 1;
  }

  return 0;
}


void
RectangleWidget::move (int obj, int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  
  switch (selected_)
  {
  case 1:
    left_x_ = last_x_ + x - pick_ix_ / (double)w;
    left_y_ = last_y_ + y - pick_iy_ / (double)h;
    break;
      
  case 2:
    width_ = width_ + left_x_ - x;
    left_x_ = x;
    height_ = height_ + left_y_ - y;
    left_y_ = y;
    break;

  case 3:
    width_ = x - left_x_;
    height_ = height_ + left_y_ - y;
    left_y_ = y;
    break;

  case 4:
    width_ = x - left_x_;
    height_ = y - left_y_;
    break;

  case 5:
    width_ = width_ + left_x_ - x;
    left_x_ = x;
    height_ = y - left_y_;
    break;

  case 6:
    offset_ = CLAMP((x - left_x_) / width_, 0.0, 1.0);
    break;
  }
}


void
RectangleWidget::release (int obj, int x, int y, int w, int h)
{
  move(obj, x, y, w, h);
}



class EditTransferFunc2 : public Module {

  GLXContext ctx_;
  Display* dpy_;
  Window win_;
  int width_, height_;
  bool button_;
  vector<Widget*> widget_;
  Pbuffer* pbuffer_;
  bool use_pbuffer_;
  
  Nrrd* histo_;
  bool histo_dirty_;
  GLuint histo_tex_;

  Colormap2Handle cmap_;
  bool cmap_dirty_;
  bool cmap_size_dirty_;
  bool cmap_out_dirty_;
  GLuint cmap_tex_;
  
  int pick_widget_; // Which widget is selected.
  int pick_object_; // The part of the widget that is selected.

public:
  EditTransferFunc2(GuiContext* ctx);
  virtual ~EditTransferFunc2();

  virtual void execute();

  void tcl_command(GuiArgs&, void*);

  bool create_histo();
  
  void update();
  void redraw();

  void push(int x, int y, int button);
  void motion(int x, int y);
  void release(int x, int y, int button);
};



DECLARE_MAKER(EditTransferFunc2)

EditTransferFunc2::EditTransferFunc2(GuiContext* ctx)
  : Module("EditTransferFunc2", ctx, Filter, "Visualization", "Volume"),
    ctx_(0), dpy_(0), win_(0), button_(0), pbuffer_(0), use_pbuffer_(true),
    histo_(0), histo_dirty_(false), histo_tex_(0),
    cmap_(new Colormap2),
    cmap_dirty_(true), cmap_size_dirty_(true), cmap_out_dirty_(true), cmap_tex_(0),
    pick_widget_(-1), pick_object_(0)
{
  widget_.push_back(scinew TriangleWidget());
  widget_.push_back(scinew RectangleWidget());
  //widget_.push_back(scinew RectangleWidget());
}


EditTransferFunc2::~EditTransferFunc2()
{
  // Clean up currently unmemorymanaged widgets.
  for (unsigned int i = 0; i < widget_.size(); i++)
  {
    delete widget_[i];
  }
  widget_.clear();
}


void
EditTransferFunc2::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for EditTransferFunc");
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
EditTransferFunc2::push(int x, int y, int button)
{
  //cerr << "push: " << x << " " << y << " " << button << endl;

  unsigned int i;

  button_ = button;

  for (i = 0; i < widget_.size(); i++)
  {
    widget_[i]->unselect_all();
  }

  pick_widget_ = -1;
  pick_object_ = 0;
  for (unsigned int i = 0; i < widget_.size(); i++)
  {
    const int tmp = widget_[i]->pick(x, 255-y, 512, 256);
    if (tmp)
    {
      pick_widget_ = i;
      pick_object_ = tmp;
      widget_[i]->select(tmp);
      break;
    }
  }
  update();
  redraw();
}



void
EditTransferFunc2::motion(int x, int y)
{
  //cerr << "motion: " << x << " " << y << endl;

  if (pick_widget_ != -1)
  {
    widget_[pick_widget_]->move(pick_object_, x, 255-y, 512, 256);
    cmap_dirty_ = true;
  }
  update();
  redraw();
}



void
EditTransferFunc2::release(int x, int y, int button)
{
  //cerr << "release: " << x << " " << y << " " << button << endl;

  button_ = 0;
  if (pick_widget_ != -1)
  {
    widget_[pick_widget_]->release(pick_object_, x, 255-y, 512, 256);
    cmap_dirty_ = true;
  }

  update();
  redraw();
}



void
EditTransferFunc2::execute()
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
EditTransferFunc2::update()
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

  if(use_pbuffer_) {
    if (TriangleWidget::Init()
        || RectangleWidget::Init()) {
      use_pbuffer_ = false;
      cerr << "Shaders not supported; switching to software rasterization" << endl;
    }
  }

  // create pbuffer
  if(use_pbuffer_ && (!pbuffer_
      || pbuffer_->width() != width_
      || pbuffer_->height() != height_)) {
    if(pbuffer_) {
      pbuffer_->destroy();
      delete pbuffer_;
    }
    pbuffer_ = new Pbuffer(width_, height_, GL_INT, 8, true, GL_FALSE);
    if(pbuffer_->create()) {
      use_pbuffer_ = false;
      pbuffer_->destroy();
      delete pbuffer_;
      pbuffer_ = 0;
      cerr << "Pbuffers not supported; switching to software rasterization" << endl;
    }
  }
  
  //----------------------------------------------------------------
  // update colormap array
  if(use_pbuffer_) {
    pbuffer_->makeCurrent();
    glDrawBuffer(GL_FRONT);
    glViewport(0, 0, width_, height_);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-1.0, -1.0, 0.0);
    glScalef(2.0, 2.0, 2.0);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);

    // Rasterize widgets
    for (unsigned int i = 0; i < widget_.size(); i++)
    {
      widget_[i]->rasterize();
    }

    glFlush();
    
    glXMakeCurrent(dpy_, win_, ctx_);
  } else {
    Array3<float>& cmap = cmap_->array();
    if(width_ != cmap.dim2() || height_ != cmap.dim1()) {
      cmap_size_dirty_ = true;
      cmap_dirty_ = true;
    }

    if (cmap_dirty_) {
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
      // Rasterize widgets
      for (unsigned int i = 0; i < widget_.size(); i++)
      {
	widget_[i]->rasterize(cmap);
      }

      // Update textures
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
EditTransferFunc2::redraw()
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
  if(use_pbuffer_) {
    pbuffer_->bind(GL_FRONT);
  } else {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, cmap_tex_);
  }
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
  if(use_pbuffer_) {
    pbuffer_->release(GL_FRONT);
  } else {
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }
  glDisable(GL_BLEND);

  // draw widgets
  for (unsigned int i = 0; i < widget_.size(); i++)
  {
    widget_[i]->draw();
  }
  
  glXSwapBuffers(dpy_, win_);
  glXMakeCurrent(dpy_, 0, 0);

  gui->unlock();
}


} // end namespace Volume
