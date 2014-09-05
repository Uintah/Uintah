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
//    File   : CM2Widget.cc
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:29 2004


#include <Core/Volume/CM2Widget.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Volume/Pbuffer.h>
#include <Core/Volume/Utils.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>

#include <sci_gl.h>
#include <GL/glu.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sstream>

// This constant should be migrated to Math something...
#define SCI_FLOAT_EPSILON 0.000001

#include <sgi_stl_warnings_on.h>

#include <math.h>
#include <stdlib.h>

using namespace std;
using namespace SCIRun;

#ifdef _WIN32
  // windows doesn't have drand48
  // we can make a better solution if it becomes necessary
  // (i.e., more files use it than this one)
#include <Core/Thread/Time.h>

double drand48()
{
  srand((int) Time::currentTicks());
  return ((double) rand())/ RAND_MAX;
}

#endif

PersistentTypeID CM2Widget::type_id("CM2Widget", "Datatype", 0);

CM2Widget::CM2Widget()
  : name_("default"),  
    color_(1.0, 1.0, 1.0),
    alpha_(0.8),
    selected_(0),
    shadeType_(CM2_SHADE_REGULAR),
    onState_(1),
    faux_(true),
    value_range_(0.0, -1.0)
    
{
  // Generates a bright random color
  while (fabs(color_[0] - color_[1]) + 
	 fabs(color_[0] - color_[2]) + 
	 fabs(color_[1] - color_[2]) < 1.0) {
    color_[0] = 1.0 - sqrt(1.0 - drand48());
    color_[1] = 1.0 - sqrt(1.0 - drand48());
    color_[2] = 1.0 - sqrt(1.0 - drand48());
  }
}

CM2Widget::~CM2Widget()
{}

CM2Widget::CM2Widget(CM2Widget& copy)
  : name_(copy.name_),
    color_(copy.color_),
    alpha_(copy.alpha_),
    selected_(copy.selected_),
    shadeType_(copy.shadeType_),
    onState_(copy.onState_),
    faux_(copy.faux_),
    value_range_(copy.value_range_)
{}

void
CM2Widget::set_value_range(range_t range) {
  if (range.first > range.second) return;
  const bool recompute = (value_range_.first > value_range_.second);
  value_range_ = range;
  if (recompute) un_normalize();
}


void
CM2Widget::draw_thick_gl_line(double x1, double y1, double x2, double y2,
			      double r, double g, double b)
{
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glLineWidth(5.0);
  glColor4d(0.0, 0.0, 0.0, 1.0);
  glBegin(GL_LINES);
  glVertex2d(x1,y1);
  glVertex2d(x2,y2);
  glEnd();

  glLineWidth(3.0);
  glColor4d(r,g,b,1.0);
  glBegin(GL_LINES);
  glVertex2d(x1,y1);
  glVertex2d(x2,y2);
  glEnd();

  glLineWidth(1.0);
  glColor4d(1.0, 1.0, 1.0, 1.0);
  glBegin(GL_LINES);
  glVertex2d(x1,y1);
  glVertex2d(x2,y2);
  glEnd();

  glDisable(GL_BLEND);
  glDisable(GL_LINE_SMOOTH);
}


void
CM2Widget::draw_thick_gl_point(double x1, double y1,
			       double r, double g, double b)
{
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_POINT_SMOOTH);

  glPointSize(7.0);
  glColor4d(0.0, 0.0, 0.0, 1.0);
  glBegin(GL_POINTS);
    glVertex2f(x1, y1);
  glEnd();

  glPointSize(5.0);
  glColor4d(r,g,b,1.0);
  glBegin(GL_POINTS);
    glVertex2f(x1, y1);
  glEnd();

  glPointSize(3.0);
  glColor4d(1.0, 1.0, 1.0, 1.0);
  glBegin(GL_POINTS);
    glVertex2f(x1, y1);
  glEnd();

  glDisable(GL_BLEND);
  glDisable(GL_POINT_SMOOTH);
}



static Persistent* RectangleCM2Widget_maker()
{
  return scinew RectangleCM2Widget;
}

PersistentTypeID RectangleCM2Widget::type_id("RectangleCM2Widget", "CM2Widget",
					     RectangleCM2Widget_maker);

#define RECTANGLECM2WIDGET_VERSION 3

void
RectangleCM2Widget::io(Piostream &stream)
{
  const int version = 
    stream.begin_class("RectangleCM2Widget", RECTANGLECM2WIDGET_VERSION);

  // Originally used "Pio(stream, (int)type_);", but this did not
  // compile on the SGI, so needed to do it this way.
  int tmp = (int)type_;
  Pio(stream, tmp);
  if (stream.reading())
  {
    type_ = (CM2RectangleType)tmp;
  }

  Pio(stream, left_x_);
  Pio(stream, left_y_);
  Pio(stream, width_);
  Pio(stream, height_);
  Pio(stream, offset_);
  Pio(stream, shadeType_);
  Pio(stream, onState_);
  Pio(stream, color_);
  Pio(stream, alpha_);

  if (version == 2) {
    Pio(stream, name_);
    double temp;
    Pio(stream, temp);
    Pio(stream, temp);
    value_range_.first = 0.0;
    value_range_.second = -1.0;
  }

  if (version == 3) {
    Pio(stream, name_);
    Pio(stream, value_range_.first);
    Pio(stream, value_range_.second);
  }

  stream.end_class();
}

RectangleCM2Widget::RectangleCM2Widget() : 
  CM2Widget(),
  type_(CM2_RECTANGLE_1D), 
  left_x_(0.25), 
  left_y_(0.5),
  width_(0.25), 
  height_(0.25),
  offset_(0.25),
  last_x_(0),
  last_y_(0),
  pick_ix_(0),
  pick_iy_(0)
{
  left_x_ = drand48()*0.9;
  left_y_ = drand48()*0.9;
  width_ = Clamp(0.1+(0.9-left_x_)*drand48(), 0.1, 0.5);
  height_ = Clamp(0.1+(0.9-left_y_)*drand48(), 0.5*width_, 1.5*width_);
  offset_ = 0.25+0.5*drand48();
  name_ = "Rectangle";
}

RectangleCM2Widget::RectangleCM2Widget(CM2RectangleType type, float left_x, 
				       float left_y, float width, float height,
				       float offset) : 
  CM2Widget(),
  type_(type), 
  left_x_(left_x), 
  left_y_(left_y), 
  width_(width), 
  height_(height),
  offset_(offset),
  last_x_(0),
  last_y_(0),
  pick_ix_(0),
  pick_iy_(0)

{
  name_ = "Rectangle";
}

RectangleCM2Widget::~RectangleCM2Widget()
{}

RectangleCM2Widget::RectangleCM2Widget(RectangleCM2Widget& copy) : 
  CM2Widget(copy),
  type_(copy.type_),
  left_x_(copy.left_x_),
  left_y_(copy.left_y_),
  width_(copy.width_),
  height_(copy.height_),
  offset_(copy.offset_),
  last_x_(copy.last_x_),
  last_y_(copy.last_y_),
  pick_ix_(copy.pick_ix_),
  pick_iy_(copy.pick_iy_)
{}

CM2Widget*
RectangleCM2Widget::clone()
{
  return new RectangleCM2Widget(*this);
}

void
RectangleCM2Widget::rasterize(CM2ShaderFactory& factory, 
			      Pbuffer* pbuffer)
{
  if (!onState_) return;
 
  CM2BlendType blend = CM2_BLEND_RASTER;
  if (pbuffer) {
    if(pbuffer->need_shader())
      blend = CM2_BLEND_FRAGMENT_NV;
    else
      blend = CM2_BLEND_FRAGMENT_ATI;
  }
  CM2ShaderType type = CM2_SHADER_RECTANGLE_1D;
  if (type_ == CM2_RECTANGLE_ELLIPSOID)
    type = CM2_SHADER_RECTANGLE_ELLIPSOID;
  FragmentProgramARB* shader = factory.shader(type, shadeType_, faux_, blend);

  if (!shader) return;
  
  if(!shader->valid()) {
    shader->create();
  }
 
  normalize();
    
  GLdouble modelview[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  double panx = (modelview[12]+1.0)/2.0;
  double pany = (modelview[13]+1.0)/2.0;
  double scalex = (modelview[0])/2.0;
  double scaley = (modelview[5])/2.0;
  double left_x = left_x_*scalex+panx;
  double left_y = left_y_*scaley+pany;
  double width = width_*scalex;
  double height = height_*scaley;
    
  
  shader->bind();
  shader->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
  shader->setLocalParam(1, left_x, left_y, width, height);

  if(offset_ < SCI_FLOAT_EPSILON )
    shader->setLocalParam(2, offset_, 0.0, 1.0, 0.0);
  else if((1.0-offset_) < SCI_FLOAT_EPSILON )
    shader->setLocalParam(2, offset_, 1.0, 0.0, 0.0);
  else
    shader->setLocalParam(2, offset_, 1/offset_, 1/(1-offset_), 0.0);
  
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  shader->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], 0.0, 0.0);
  if(pbuffer)
    shader->setLocalParam(4, 1.0/pbuffer->width(), 1.0/pbuffer->height(), 
			  0.0, 0.0);
  
  
  glBegin(GL_QUADS);
  {
    glVertex2f(left_x_, left_y_);
    glVertex2f(left_x_+width_, left_y_);
    glVertex2f(left_x_+width_, left_y_+height_);
    glVertex2f(left_x_, left_y_+height_);
  }
  glEnd();
  un_normalize();
  shader->release();
}

void
RectangleCM2Widget::rasterize(Array3<float>& array)
{
  if(!onState_) return;
  if(array.dim3() != 4) return;
  normalize();
  int size_x = array.dim2();
  int size_y = array.dim1();
  float left = left_x_;
  float right = left_x_+width_;
  float bottom = left_y_;
  float top = left_y_+height_;

  int lb = int(bottom*size_y);
  int le = int(top*size_y);
  int ilb = Clamp(lb, 0, size_y-1);
  int ile = Clamp(le, 0, size_y-1);
//  int la = int((mBall.y*mSize.y+bottom)*size.y);
  int rb = int(left*size_x);
  int re = int(right*size_x);
  int ra = int((offset_*width_+left)*size_x);
  int jrb = Clamp(rb, 0, size_x-1);
  int jre = Clamp(re, 0, size_x-1);
  int jra = Clamp(ra, 0, size_x-1);
  switch(type_) {
    case CM2_RECTANGLE_ELLIPSOID: {
      for(int i=ilb; i<=ile; i++) {
        for(int j=jrb; j<jre; j++) {
          float x = j/(float)size_x;
          float y = i/(float)size_y;
          x -= (left+right)/2;
          y -= (bottom+top)/2;
          x *= height_/width_;
          float w = 1-2*sqrt(x*x+y*y)/size_y;
          if (w < 0) w = 0;
          float a = alpha_*w;
          float r = faux_ ? color_.r()*w : color_.r();
          float g = faux_ ? color_.r()*w : color_.g();
          float b = faux_ ? color_.r()*w : color_.b();
          array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + r, 0.0f, 1.0f);
          array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + g, 0.0f, 1.0f);
          array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + b, 0.0f, 1.0f);
          array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
        }
      }
    } break;

    case CM2_RECTANGLE_1D: {
      if (shadeType_ == CM2_SHADE_FLAT) 
      {
        for(int i=ilb; i<=ile; i++) {
          for(int j=jrb; j<jre; j++) {
            array(i,j,0) = Clamp((float)color_.r(), 0.0f, 1.0f);
            array(i,j,1) = Clamp((float)color_.g(), 0.0f, 1.0f);
            array(i,j,2) = Clamp((float)color_.b(), 0.0f, 1.0f);
            array(i,j,3) = Clamp(alpha_, 0.0f, 1.0f);
          }
        }
      } else if (faux_) {
        float da = ra <= rb+1 ? 0.0 : alpha_/(ra-rb-1);
        float dr = ra <= rb+1 ? 0.0 : color_.r()/(ra-rb-1);
        float dg = ra <= rb+1 ? 0.0 : color_.g()/(ra-rb-1);
        float db = ra <= rb+1 ? 0.0 : color_.b()/(ra-rb-1);
        float a = ra <= rb+1 ? alpha_ : alpha_ - abs(ra-jra)*da;
        float r = ra <= rb+1 ? color_.r() : color_.r() - abs(ra-jra)*dr;
        float g = ra <= rb+1 ? color_.g() : color_.g() - abs(ra-jra)*dg;
        float b = ra <= rb+1 ? color_.b() : color_.b() - abs(ra-jra)*db;
        for(int j=jra-1; j>=jrb; j--, a-=da, r-=dr, b-=db, g-=dg) {
          for(int i=ilb; i<=ile; i++) {
          
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + r, 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + g, 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + b, 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
          }
        }
        da = ra < re-1 ? alpha_/(re-ra-1) : 0.0;
        dr = ra < re-1 ? color_.r()/(re-ra-1) : 0.0;
        dg = ra < re-1 ? color_.g()/(re-ra-1) : 0.0;
        db = ra < re-1 ? color_.b()/(re-ra-1) : 0.0;
        a = alpha_ - abs(ra-jra)*da;
        r = color_.r() - abs(ra-jra)*dr;
        g = color_.g() - abs(ra-jra)*dg;
        b = color_.b() - abs(ra-jra)*db;
        for(int j=jra; j<=jre; j++, a-=da, r-=dr, b-=db, g-=dg) {
          for(int i=ilb; i<=ile; i++) {
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + r, 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + g, 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + b, 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
          }
        }
      } else { // !faux
        float da = ra <= rb+1 ? 0.0 : alpha_/(ra-rb-1);
        float a = ra <= rb+1 ? alpha_ : alpha_ - abs(ra-jra)*da;
        for(int j=jra-1; j>=jrb; j--, a-=da) {
          for(int i=ilb; i<=ile; i++) {
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + 
				 (float)color_.r(), 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + 
				 (float)color_.g(), 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + 
				 (float)color_.b(), 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + 
				 a, 0.0f, 1.0f);
          }
        }
        da = ra < re-1 ? alpha_/(re-ra-1) : 0.0;
        a = alpha_ - abs(ra-jra)*da;
        for(int j=jra; j<=jre; j++, a-=da) {
          for(int i=ilb; i<=ile; i++) {
            array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + 
				 (float)color_.r(), 0.0f, 1.0f);
            array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + 
				 (float)color_.g(), 0.0f, 1.0f);
            array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + 
				 (float)color_.b(), 0.0f, 1.0f);
            array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + 
				 a, 0.0f, 1.0f);
          }
        }
      }  // end !faux
    } break;

  default:
    break;
  }
  un_normalize();
}

void
RectangleCM2Widget::draw()
{
  if(!onState_) return;

  double r = 0.5, g = 0.5, b = 0.5;
  if (selected_) {
    r = 0.9; g = 0.8; b = 0.1;
  }
  normalize();
  draw_thick_gl_line(left_x_, left_y_, left_x_+width_, left_y_, r,g,b);
  draw_thick_gl_line(left_x_, left_y_+height_, left_x_+width_, left_y_+height_,
		     r,g,b);

  draw_thick_gl_line(left_x_, left_y_, left_x_, left_y_+height_, r,g,b);
  draw_thick_gl_line(left_x_+width_, left_y_, left_x_+width_, left_y_+height_, 
		     r,g,b);

  if (selected_) {
    r = 0.9; g = 0.6; b = 0.4;
  }

  draw_thick_gl_point(left_x_, left_y_,r,g,b);
  draw_thick_gl_point(left_x_+width_, left_y_,r,g,b);
  draw_thick_gl_point(left_x_, left_y_+height_,r,g,b);
  draw_thick_gl_point(left_x_+width_, left_y_+height_,r,g,b);
  draw_thick_gl_point(left_x_+offset_*width_, left_y_+height_*0.5,r,g,b);
  un_normalize();
}



string
RectangleCM2Widget::tk_cursorname(int obj)
{
  switch (obj) {
  case 0: return string("left_ptr"); break;
  case 1: return string("fleur"); break;
  case 2: return string("bottom_left_corner"); break;
  case 3: return string("bottom_right_corner"); break;
  case 4: return string("top_right_corner"); break;
  case 5: return string("top_left_corner"); break;
  case 6: return string("sb_h_double_arrow"); break;
  case 7: return string("fleur"); break;
  case 8: return string("sb_h_double_arrow"); break;
  case 9: return string("sb_h_double_arrow"); break;
  case 10: return string("sb_v_double_arrow"); break;
  case 11: return string("sb_v_double_arrow"); break;
  default: break;
  }
  return string("left_ptr");
}


int
RectangleCM2Widget::pick1 (int ix, int iy, int w, int h)
{
  //todo
  double point_size_ = 5.0;
  normalize();
  last_x_ = left_x_;
  last_y_ = left_y_;
  pick_ix_ = ix;
  pick_iy_ = iy;
  int ret_val = 0;
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  const double xeps = point_size_ / w * 0.5;
  const double yeps = point_size_ / h * 0.5;

  double left = fabs(x - left_x_);
  double right = fabs(x - left_x_ - width_);
  double bottom = fabs(y - left_y_);
  double top = fabs(y - left_y_ - height_);

  if (left < xeps && bottom < yeps) ret_val = 2;
  else if (right < xeps && bottom < yeps) ret_val = 3;
  else if (right < xeps && top < yeps) ret_val = 4;
  else if (left < xeps && top < yeps) ret_val = 5;
  else if (fabs(x - left_x_ - offset_ * width_) < xeps &&
	   fabs(y - left_y_ - height_ * 0.5) < yeps) ret_val = 6;
  else if (left < xeps && y > left_y_ && y < (left_y_+height_)) ret_val = 8;
  else if (right < xeps && y > left_y_ && y < (left_y_+height_)) ret_val = 9;
  else if (x > left_x_ && x < (left_x_+width_) && bottom < yeps) ret_val = 10;
  else if (x > left_x_ && x < (left_x_+width_) && top < yeps) ret_val = 11;

  un_normalize();
  return ret_val;
}


int
RectangleCM2Widget::pick2 (int ix, int iy, int w, int h, int m)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  int ret_val = 0;
  normalize();
  if (x > Min(left_x_, left_x_ + width_) &&
      x < Max(left_x_, left_x_ + width_) &&
      y > Min(left_y_, left_y_ + height_) &&
      y < Max(left_y_, left_y_ + height_))
  {
    last_x_ = left_x_;
    last_y_ = left_y_;
    pick_ix_ = ix;
    pick_iy_ = iy;
    last_hsv_ = HSVColor(color_);
    ret_val =  m?7:1;
  }
  un_normalize();
  return ret_val;
}


void
RectangleCM2Widget::move (int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  normalize();
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
    offset_ = Clamp((x - left_x_) / width_, 0.0, 1.0);
    break;

  case 7:
    {
      // Hue controls on x axis
      const double hdelta = x - pick_ix_ / (double)w;
      double hue = last_hsv_[0] + hdelta * 360.0 * 2.0;
      while (hue < 0.0) hue += 360.0;
      while (hue > 360.0) hue -= 360;

      // Saturation controls on y axis
      const double sdelta = y - pick_iy_ / (double)h;
      double sat = Clamp(last_hsv_[1] - sdelta * 2.0, 0.0, 1.0);

      HSVColor hsv(hue, sat, last_hsv_.val());
      color_ = Color(hsv);
    }
    break;

  case 8:
    width_ = width_ + left_x_ - x;
    left_x_ = x;
    break;

  case 9:
    width_ = x - left_x_;
    break;

  case 10:
    height_ = height_ + left_y_ - y;
    left_y_ = y;
    break;

  case 11:
    height_ = y - left_y_;
    break;
  }

  if (width_ < 0.0) {
    left_x_ += width_;
    width_ *= -1;
    switch (selected_) {
    case 2: selected_ = 3; break;
    case 3: selected_ = 2; break;
    case 4: selected_ = 5; break;
    case 5: selected_ = 4; break;
    case 8: selected_ = 9; break;
    case 9: selected_ = 8; break;
    default: break;
    }
  }

  if (height_ < 0.0) {
    left_y_ += height_;
    height_ *= -1;
    switch (selected_) {
    case 2: selected_ = 5; break;
    case 3: selected_ = 4; break;
    case 4: selected_ = 3; break;
    case 5: selected_ = 2; break;
    case 10: selected_ = 11; break;
    case 11: selected_ = 10; break;
    default: break;
    }
  }  
  un_normalize();
}


void
RectangleCM2Widget::release (int /*x*/, int /*y*/,
                             int /*w*/, int /*h*/)
{
  normalize();
  if (width_ < 0.0) {
    left_x_ += width_;
    width_ *= -1.0;
  }

  if (height_ < 0.0) {
    left_y_ += height_;
    height_ *= -1.0;
  }
  un_normalize();
  // Don't need to do anything here.
}


string
RectangleCM2Widget::tcl_pickle()
{
  normalize();
  ostringstream s;
  s << "r ";
  s << (int)type_ << " ";
  s << left_x_ << " ";
  s << left_y_ << " ";
  s << width_ << " ";
  s << height_ << " ";
  s << offset_;
  un_normalize();
  return s.str();
}

void
RectangleCM2Widget::tcl_unpickle(const string &p)
{
  istringstream s(p);
  char c;
  s >> c;
  int t;
  s >> t;
  type_ = (CM2RectangleType)t;
  s >> left_x_;
  s >> left_y_;
  s >> width_;
  s >> height_;
  s >> offset_;
  value_range_.first = 0.0;
  value_range_.second = -1.0;
}


void
RectangleCM2Widget::normalize()
{
  if (value_range_.first > value_range_.second) return;
  const float offset = -value_range_.first;
  const float scale = 1.0/(value_range_.second-value_range_.first);

  left_x_ = (left_x_ + offset) * scale;
  width_ = width_ * scale;
}


void
RectangleCM2Widget::un_normalize()
{
  if (value_range_.first > value_range_.second) return;
  const float offset = -value_range_.first;
  const float scale = 1.0/(value_range_.second-value_range_.first);
  
  left_x_ = left_x_/scale - offset;
  width_ = width_/scale;
}


static Persistent* TriangleCM2Widget_maker()
{
  return scinew TriangleCM2Widget;
}

PersistentTypeID TriangleCM2Widget::type_id("TriangleCM2Widget", "CM2Widget",
					    TriangleCM2Widget_maker);

#define TRIANGLECM2WIDGET_VERSION 3

void
TriangleCM2Widget::io(Piostream &stream)
{
  const int version = 
    stream.begin_class("TriangleCM2Widget", TRIANGLECM2WIDGET_VERSION);

  Pio(stream, base_);
  Pio(stream, top_x_);
  Pio(stream, top_y_);
  Pio(stream, width_);
  Pio(stream, bottom_);
  Pio(stream, shadeType_);
  Pio(stream, onState_);
  Pio(stream, color_);
  Pio(stream, alpha_);

  if (version == 2) {
    Pio(stream, name_);
    double temp;
    Pio(stream, temp);
    Pio(stream, temp);
    value_range_.first = 0.0;
    value_range_.second = -1.0;
  }

  if (version == 3) {
    Pio(stream, name_);
    Pio(stream, value_range_.first);
    Pio(stream, value_range_.second);
  }
    
  stream.end_class();
}

TriangleCM2Widget::TriangleCM2Widget() : 
  CM2Widget(),
  base_(0.5), 
  top_x_(0.15), 
  top_y_(0.5), 
  width_(0.25), 
  bottom_(0.5)
{
  base_ = 0.1+drand48()*0.8;
  top_x_ = -0.1 + drand48()*0.2;
  top_y_ = 0.2 + drand48()*0.8;
  width_ = 0.1 + drand48()*0.4;
  bottom_ = drand48()*0.4+0.2;
  name_ = "Triangle";
}

TriangleCM2Widget::TriangleCM2Widget(float base, float top_x, float top_y,
                                     float width, float bottom) : 
  CM2Widget(),
  base_(base), 
  top_x_(top_x), 
  top_y_(top_y), 
  width_(width), 
  bottom_(bottom)
{}

TriangleCM2Widget::~TriangleCM2Widget()
{}

TriangleCM2Widget::TriangleCM2Widget(TriangleCM2Widget& copy) : 
  CM2Widget(copy),
  base_(copy.base_),
  top_x_(copy.top_x_),
  top_y_(copy.top_y_),
  width_(copy.width_),
  bottom_(copy.bottom_),
  last_x_(copy.last_x_),
  last_y_(copy.last_y_),
  last_width_(copy.last_width_),
  pick_ix_(copy.pick_ix_),
  pick_iy_(copy.pick_iy_)
{}

CM2Widget*
TriangleCM2Widget::clone()
{
  return new TriangleCM2Widget(*this);
}

void
TriangleCM2Widget::rasterize(CM2ShaderFactory& factory, 
			     Pbuffer* pbuffer)
{
  if(!onState_) return;

  CM2BlendType blend = CM2_BLEND_RASTER;
  if(pbuffer) {
    if(pbuffer->need_shader())
      blend = CM2_BLEND_FRAGMENT_NV;
    else
      blend = CM2_BLEND_FRAGMENT_ATI;
  }

  FragmentProgramARB* shader = 
    factory.shader(CM2_SHADER_TRIANGLE, shadeType_, faux_, blend);

  if (!shader) return;
  
  if(!shader->valid()) shader->create();
    
  GLdouble modelview[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  double panx = (modelview[12]+1.0)/2.0;
  double pany = (modelview[13]+1.0)/2.0;
  double scalex = (modelview[0])/2.0;
  double scaley = (modelview[5])/2.0;
  normalize();  
  shader->bind();
  shader->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
  shader->setLocalParam(1, base_*scalex+panx, 
			scalex*(base_+top_x_)+panx,
			top_y_*scaley+pany, pany);
  shader->setLocalParam(2, width_*scalex, bottom_, pany, 0.0);
  
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  shader->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], pany, pany);
  if(pbuffer)
    shader->setLocalParam(4, 1.0/pbuffer->width(), 1.0/pbuffer->height(), 
			  0.0, 0.0);
  
  glBegin(GL_TRIANGLES);
  {
    glVertex2f(base_, 0.0);
    glVertex2f(base_+top_x_+width_/2, top_y_);
    glVertex2f(base_+top_x_-width_/2, top_y_);
  }
  glEnd();
  shader->release();
  un_normalize();
}

void
TriangleCM2Widget::rasterize(Array3<float>& array)
{
  if(!onState_) return;
  if(array.dim3() != 4) return;
  normalize();
  int size_x = array.dim2();
  int size_y = array.dim1();
  float top_left = top_x_ - fabs(width_)/2;
  float top_right = top_x_ + fabs(width_)/2;
  int lb = (int)(bottom_*top_y_*size_y);
  int le = (int)(top_y_*size_y);
  int ilb = Clamp(lb, 0, size_y-1);
  int ile = Clamp(le, 0, size_y-1);
  if (shadeType_ == CM2_SHADE_FLAT) 
  {
    for(int i=ilb; i<=ile; i++) {
      float fb = (i/(float)le)*top_left + base_;
      float fe = (i/(float)le)*top_right + base_;
//      float fm = (i/(float)le)*top_x_ + base_;
      int rb = (int)(fb*size_x);
      int re = (int)(fe*size_x);
      int jrb = Clamp(rb, 0, size_x-1);
      int jre = Clamp(re, 0, size_x-1);
      
      for(int j=jrb; j<jre; j++) {
        array(i,j,0) = Clamp((float)color_.r(), 0.0f, 1.0f);
        array(i,j,1) = Clamp((float)color_.g(), 0.0f, 1.0f);
        array(i,j,2) = Clamp((float)color_.b(), 0.0f, 1.0f);
        array(i,j,3) = Clamp(alpha_, 0.0f, 1.0f);
      }
    }
  }

  if (faux_) {
    for(int i=ilb; i<=ile; i++) {
      float fb = (i/(float)le)*top_left + base_;
      float fe = (i/(float)le)*top_right + base_;
      float fm = (i/(float)le)*top_x_ + base_;
      int rb = (int)(fb*size_x);
      int re = (int)(fe*size_x);
      int rm = (int)(fm*size_x);
      int jrb = Clamp(rb, 0, size_x-1);
      int jre = Clamp(re, 0, size_x-1);
      int jrm = Clamp(rm, 0, size_x-1);
      float da = alpha_/(rm-rb);
      float dr = color_.r()/(rm-rb);
      float dg = color_.g()/(rm-rb);
      float db = color_.b()/(rm-rb);
      float a = alpha_ - abs(rm-jrm+1)*da;
      float r = color_.r() - abs(rm-jrm+1)*dr;
      float g = color_.g() - abs(rm-jrm+1)*dg;
      float b = color_.b() - abs(rm-jrm+1)*db;
      for(int j=jrm-1; j>=jrb; j--, a-=da, r-=dr, b-=db, g-=dg) {
        array(i,j,0) = Clamp(array(i,j,0)*(1-a) + r, 0.0f, 1.0f);
        array(i,j,1) = Clamp(array(i,j,1)*(1-a) + g, 0.0f, 1.0f);
        array(i,j,2) = Clamp(array(i,j,2)*(1-a) + b, 0.0f, 1.0f);
        array(i,j,3) = Clamp(array(i,j,3)*(1-a) + a, 0.0f, 1.0f);
      }
      da = alpha_/(re-rm);
      dr = color_.r()/(re-rm);
      dg = color_.g()/(re-rm);
      db = color_.b()/(re-rm);
      a = alpha_ - abs(rm-jrm)*da;
      r = color_.r() - abs(rm-jrm)*dr;
      g = color_.g() - abs(rm-jrm)*dg;
      b = color_.b() - abs(rm-jrm)*db;
      for (int j=jrm; j<=jre; j++, a-=da, r-=dr, b-=db, g-=dg)
      {
        array(i,j,0) = array(i,j,0)*(1-a) + r;
        array(i,j,1) = array(i,j,1)*(1-a) + g;
        array(i,j,2) = array(i,j,2)*(1-a) + b;
        array(i,j,3) = array(i,j,3)*(1-a) + a;
      }
    }
  } else {
    for(int i=ilb; i<=ile; i++) {
      float fb = (i/(float)le)*top_left + base_;
      float fe = (i/(float)le)*top_right + base_;
      float fm = (i/(float)le)*top_x_ + base_;
      int rb = (int)(fb*size_x);
      int re = (int)(fe*size_x);
      int rm = (int)(fm*size_x);
      int jrb = Clamp(rb, 0, size_x-1);
      int jre = Clamp(re, 0, size_x-1);
      int jrm = Clamp(rm, 0, size_x-1);
      float da = alpha_/(rm-rb);
      float a = alpha_ - abs(rm-jrm+1)*da;
      for(int j=jrm-1; j>=jrb; j--, a-=da) {
        array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + (float)color_.r(), 
			     0.0f, 1.0f);
        array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + (float)color_.g(), 
			     0.0f, 1.0f);
        array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + (float)color_.b(), 
			     0.0f, 1.0f);
        array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
      }
      da = alpha_/(re-rm);
      a = alpha_ - abs(rm-jrm)*da;
      for (int j=jrm; j<=jre; j++, a-=da)
      {
        array(i,j,0) = Clamp(array(i,j,0)*(1.0f-a) + (float)color_.r(), 
			     0.0f, 1.0f);
        array(i,j,1) = Clamp(array(i,j,1)*(1.0f-a) + (float)color_.g(), 
			     0.0f, 1.0f);
        array(i,j,2) = Clamp(array(i,j,2)*(1.0f-a) + (float)color_.b(), 
			     0.0f, 1.0f);
        array(i,j,3) = Clamp(array(i,j,3)*(1.0f-a) + a, 0.0f, 1.0f);
      }
    }
  }
  un_normalize();
}


void
TriangleCM2Widget::draw()
{
  if(!onState_) return;
  normalize();
  const float b_x = bottom_*top_x_ + base_;
  const float b_y = bottom_*top_y_;
  const float w = bottom_*width_;

  double r = 0.5, g = 0.5, b = 0.5;
  if (selected_) {
    r = 0.8; g = 0.7; b = 0.1;
  }
  draw_thick_gl_line(b_x-w/2, b_y, b_x+w/2, b_y, r,g,b);
  draw_thick_gl_line(base_+top_x_-width_/2, top_y_,
		     base_+top_x_+width_/2, top_y_,  r, g, b);
  draw_thick_gl_line(base_, 0.0, base_+top_x_-width_/2, top_y_, r,g,b);
  draw_thick_gl_line(base_, 0.0, base_+top_x_+width_/2, top_y_, r,g,b);

  if (selected_) {
    r = 0.9; g = 0.6; b = 0.4;
  }

  draw_thick_gl_point(base_, 0.0, r,g,b);
  draw_thick_gl_point(b_x-w/2, b_y,r,g,b);
  draw_thick_gl_point(b_x+w/2, b_y,r,g,b);
  draw_thick_gl_point(base_+top_x_-width_/2, top_y_,r,g,b);
  draw_thick_gl_point(base_+top_x_+width_/2, top_y_,r,g,b);
  un_normalize();
}


int
TriangleCM2Widget::pick1 (int ix, int iy, int sw, int sh)
{
  int ret_val = 0;
  normalize();
  //todo
  double point_size_ = 5.0;
  double thick_line_width_ = 5.0;
  last_x_ = top_x_;
  last_y_ = top_y_;
  pick_ix_ = ix;
  pick_iy_ = iy;
  last_width_ = width_;

  const double x = ix / (double)sw;
  const double y = iy / (double)sh;
  const double xeps = point_size_ / sw * 0.5;
  const double yeps = point_size_ / sh * 0.5;
  const double yleps = thick_line_width_ / sh;
  const float b_x = bottom_*top_x_ + base_;
  const float b_y = bottom_*top_y_;
  const float w = bottom_*width_;
  const double top_right_x = base_ + top_x_ + width_/2;
  const double top_left_x = base_ + top_x_ - width_/2;
  const double bot_left_x = b_x - w/2;
  const double bot_right_x = b_x + w/2;


  // upper right corner
  if (fabs(x - top_right_x) < xeps && fabs(y - top_y_) < yeps) ret_val = 2; 
  // upper left corner
  else if (fabs(x - top_left_x) < xeps && fabs(y - top_y_) < yeps) ret_val = 6;
  // middle left corner
  else if (fabs(x - bot_left_x) < xeps && fabs(y - b_y) < yeps) ret_val = 3;
  // middle right corner
  else if (fabs(x - bot_right_x) < xeps && fabs(y - b_y) < yeps) ret_val = 8; 
  // top bar
  else if (fabs(y - top_y_) < yleps && x > Min(top_right_x, top_left_x) && 
	   x < Max(top_right_x, top_left_x)) ret_val = 4; 
  // bottom bar
  else if (fabs(y - b_y) < yleps && x > Min(bot_left_x, bot_right_x) && 
	   x < Max(bot_left_x, bot_right_x)) ret_val = 7;
  un_normalize();
  return ret_val;
}


int
TriangleCM2Widget::pick2 (int ix, int iy, int sw, int sh, int m)
{
  normalize();
  const double x = ix / (double)sw;
  const double y = iy / (double)sh;
  const double x1top = top_x_ + width_ * 0.5;
  const double x2top = top_x_ - width_ * 0.5;
  const double x1 = base_ + x1top * (y / top_y_);
  const double x2 = base_ + x2top * (y / top_y_);
  int ret_val = 0;
  if (y < top_y_ && x > Min(x1, x2) && x < Max(x1, x2))
  {
    last_x_ = base_;
    pick_ix_ = ix;
    pick_iy_ = iy; // modified only.
    last_hsv_ = HSVColor(color_);  // modified only
    ret_val = m?5:1;
  }
  un_normalize();
  return ret_val;
}


string
TriangleCM2Widget::tk_cursorname(int obj)
{
  switch (obj) {
  case 0: return string("left_ptr"); break;
  case 1: return string("sb_h_double_arrow"); break;
  case 2: return string("fleur"); break;
  case 3: return string("sb_v_double_arrow"); break;
  case 4: return string("fleur"); break;
  case 5: return string("left_ptr"); break;
  case 6: return string("fleur"); break;
  case 7: return string("sb_v_double_arrow"); break;
  case 8: return string("sb_v_double_arrow"); break;
  default: break;
  }
  return string("left_ptr");
}
    
    


void
TriangleCM2Widget::move (int ix, int iy, int w, int h)
{
  const double x = ix / (double)w;
  const double y = iy / (double)h;
  normalize();
  switch (selected_)
  {
  case 1:
    base_ = last_x_ + x - pick_ix_ / (double)w;
    break;

  case 2:
    width_ = (x - top_x_ - base_) * 2.0;
    top_y_ = last_y_ + y - pick_iy_ / (double)h;
    break;

  case 3:
    bottom_ = Clamp(y / top_y_, 0.0, 1.0);
    break;

  case 4:
    top_x_ = last_x_ + x - pick_ix_ / (double)w;
    top_y_ = last_y_ + y - pick_iy_ / (double)h;
    width_ = last_width_ * top_y_ / last_y_;
    break;
    
  case 5:
    {
      // Hue controls on x axis
      const double hdelta = x - pick_ix_ / (double)w;
      double hue = last_hsv_[0] + hdelta * 360.0 * 2.0;
      while (hue < 0.0) hue += 360.0;
      while (hue > 360.0) hue -= 360;

      // Saturation controls on y axis
      const double sdelta = y - pick_iy_ / (double)h;
      double sat = Clamp(last_hsv_[1] - sdelta * 2.0, 0.0, 1.0);

      HSVColor hsv(hue, sat, last_hsv_.val());
      color_ = Color(hsv);
    }
    break;
  case 6:
    width_ = (x - top_x_ - base_) * 2.0;
    top_y_ = last_y_ + y - pick_iy_ / (double)h;
    break;

  case 7:
    bottom_ = Clamp(y / top_y_, 0.0, 1.0);
    break;

  case 8:
    bottom_ = Clamp(y / top_y_, 0.0, 1.0);
    break;

  default:
    break;
  }
  un_normalize();
}


void
TriangleCM2Widget::release (int /*x*/, int /*y*/, 
                            int /*w*/, int /*h*/)
{
  // Don't need to do anything here.
}


string
TriangleCM2Widget::tcl_pickle()
{
  normalize();
  ostringstream s;
  s << "t ";
  s << base_ << " ";
  s << top_x_ << " ";
  s << top_y_ << " ";
  s << width_ << " ";
  s << bottom_;
  un_normalize();
  return s.str();
}

void
TriangleCM2Widget::tcl_unpickle(const string &p)
{
  istringstream s(p);
  char c;
  s >> c;
  s >> base_;
  s >> top_x_;
  s >> top_y_;
  s >> width_;
  s >> bottom_;
  value_range_.first = 0.0;
  value_range_.second = -1.0;
}


void
TriangleCM2Widget::normalize()
{
  if (value_range_.first > value_range_.second) return;
  const float offset = -value_range_.first;
  const float scale = 1.0/(value_range_.second-value_range_.first);

  base_ = (base_ + offset) * scale;
  top_x_ = top_x_ * scale;
  width_ = width_ * scale;
}


void
TriangleCM2Widget::un_normalize()
{
  if (value_range_.first > value_range_.second) return;
  const float offset = -value_range_.first;
  const float scale = 1.0/(value_range_.second-value_range_.first);
  
  base_ = base_/scale - offset;
  top_x_ = top_x_/scale;
  width_ = width_/scale;
}


// Image --
#define IMAGECM2WIDGET_VERSION 1

static Persistent* ImageCM2Widget_maker()
{
  return scinew ImageCM2Widget;
}

PersistentTypeID ImageCM2Widget::type_id("ImageCM2Widget", "CM2Widget",
					 ImageCM2Widget_maker);

void
ImageCM2Widget::io(Piostream &stream)
{
  stream.begin_class("ImageCM2Widget", IMAGECM2WIDGET_VERSION);
  Pio(stream, pixels_);
  stream.end_class();
}

ImageCM2Widget::ImageCM2Widget() : 
  pixels_(0)
{
}

ImageCM2Widget::ImageCM2Widget(NrrdDataHandle p) : 
  pixels_(p)
{}

ImageCM2Widget::~ImageCM2Widget()
{}

ImageCM2Widget::ImageCM2Widget(ImageCM2Widget& copy) : 
  CM2Widget(copy),
  pixels_(copy.pixels_)
{
  pixels_.detach();
}

CM2Widget*
ImageCM2Widget::clone()
{
  return new ImageCM2Widget(*this);
}

const float trans = 1.0/255.0;

void
ImageCM2Widget::rasterize(CM2ShaderFactory& /*factory*/, Pbuffer* pbuffer)
{
  CHECK_OPENGL_ERROR("ImageCM2Widget::rasterize - - start");
  //assume images draw first.
  
  if (! pixels_.get_rep()) return;
  
  glDisable(GL_BLEND);
  glRasterPos2i(0,0);

  // float data in the range 0 - 255 
  glPixelTransferf(GL_RED_SCALE, trans);
  glPixelTransferf(GL_GREEN_SCALE, trans);
  glPixelTransferf(GL_BLUE_SCALE, trans);
  glPixelTransferf(GL_ALPHA_SCALE, trans);


  Nrrd *nout = pixels_->nrrd;
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);

  if (vp[2] != nout->axis[1].size || vp[3] != nout->axis[2].size) 
  {
    nout = resize(vp[2], vp[3]);
  }
  if (!nout) return;

  glDrawPixels(vp[2], vp[3], GL_RGBA, GL_FLOAT, (float*)nout->data);

  if (nout != pixels_->nrrd)
    nrrdNuke(nout);

  // restore default values
  glPixelTransferf(GL_RED_SCALE, 1.0);
  glPixelTransferf(GL_GREEN_SCALE, 1.0);
  glPixelTransferf(GL_BLUE_SCALE, 1.0);
  glPixelTransferf(GL_ALPHA_SCALE, 1.0);

  glEnable(GL_BLEND);
  CHECK_OPENGL_ERROR("ImageCM2Widget::rasterize - - end");
}

Nrrd*
ImageCM2Widget::resize(int width, int height) 
{
  Nrrd *nin   = pixels_->nrrd;
  NrrdResampleInfo *info = nrrdResampleInfoNew();
  NrrdKernel *kern;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0L;
  kern = nrrdKernelBCCubic; 
  p[1] = 1.0L; 
  p[2] = 0.0L; 
  info->kernel[0] = kern;
  info->kernel[1] = kern;
  info->kernel[2] = kern;
  info->samples[0]= 4;
  info->samples[1]= width;
  info->samples[2]= height;

  for (int a = 0; a < 3; a++) {

    if (nrrdKindSize(nin->axis[a].kind) > 1) {
      std::cerr << "Trying to resample along axis " << a 
		<< " which is not of nrrdKindDomain or nrrdKindUnknown." 
		<< std::endl;
    }

    memcpy(info->parm[a], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));
    if (info->kernel[a] && 
	(!(airExists(nin->axis[a].min) && 
	   airExists(nin->axis[a].max)))) {
      nrrdAxisInfoMinMaxSet(nin, a, nin->axis[a].center ? 
			    nin->axis[a].center : nrrdDefCenter);
    }
    info->min[a] = nin->axis[a].min;
    info->max[a] = nin->axis[a].max;
  } 

  info->boundary = nrrdBoundaryBleed;
  info->type = nin->type;
  info->renormalize = AIR_TRUE;

  bool fail = false;
  Nrrd *nrrd_resamp = nrrdNew();
  if (nrrdSpatialResample(nrrd_resamp, nin, info)) {
    char *err = biffGetDone(NRRD);
    std::cerr << "Resample Failed in Core/Volume/CM2Widget.cc: " 
	      << err << std::endl;
    free(err);
    fail = true;
  }
  nrrdResampleInfoNix(info); 
  if (fail) {
    nrrdNuke(nrrd_resamp);
    return 0;
  } else {
    return nrrd_resamp;
  }
}

void
ImageCM2Widget::rasterize(Array3<float>& array)
{
  if (! pixels_.get_rep()) return;
  ASSERT(pixels_->nrrd->type == nrrdTypeFloat);
  if(array.dim3() != 4) return;

  Nrrd *nout = 0;
  if (array.dim2() != pixels_->nrrd->axis[1].size || 
      array.dim1() != pixels_->nrrd->axis[2].size) 
  {
    nout = resize(array.dim2(), array.dim1());
  }

  // return if resample fails.
  if (!nout) return;

  const int chans = 4;
  const int ysz = nout->axis[2].size;
  const int xsz = nout->axis[1].size;
  for (int j = 0; j < ysz; ++j) {
    const int yoff = j * xsz * chans;
    for (int i = 0; i < xsz; ++i) {
      float *dat = ((float*)nout->data) + yoff + (i * chans);
      array(j,i,0) = trans * *dat++;
      array(j,i,1) = trans * *dat++;
      array(j,i,2) = trans * *dat++;
      array(j,i,3) = trans * *dat;
    }
  }
}

void
ImageCM2Widget::draw()
{
  // no widget controls to draw.
}



#define PAINTCM2WIDGET_VERSION 1

static Persistent* PaintCM2Widget_maker()
{
  return scinew PaintCM2Widget;
}

PersistentTypeID PaintCM2Widget::type_id("PaintCM2Widget", "CM2Widget",
					 PaintCM2Widget_maker);

void
PaintCM2Widget::io(Piostream &stream)
{
  stream.begin_class("PaintCM2Widget", PAINTCM2WIDGET_VERSION);
  
  Pio(stream, strokes_);
  Pio(stream, name_);
  stream.end_class();
}

PaintCM2Widget::PaintCM2Widget() : 
  CM2Widget(),
  strokes_()
{
  name_ = "Paint";
}

PaintCM2Widget::~PaintCM2Widget()
{}

PaintCM2Widget::PaintCM2Widget(PaintCM2Widget& copy) : 
  CM2Widget(copy),
  strokes_(copy.strokes_)
{
}

CM2Widget*
PaintCM2Widget::clone()
{
  return new PaintCM2Widget(*this);
}

void
PaintCM2Widget::rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer)
{
  if(!onState_) return;
  normalize();
  CM2BlendType blend = CM2_BLEND_RASTER;
  if(pbuffer) {
    if(pbuffer->need_shader())
      blend = CM2_BLEND_FRAGMENT_NV;
    else
      blend = CM2_BLEND_FRAGMENT_ATI;
  }

  FragmentProgramARB* shader = 
    factory.shader(CM2_SHADER_PAINT, shadeType_, faux_, blend);
  if (!shader) return;

  if(!shader->valid()) {
    shader->create();
  }
    
  GLdouble modelview[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  double pany = (modelview[13]+1.0)/2.0;
  
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  
  shader->bind();
  shader->setLocalParam(0, color_.r(), color_.g(), color_.b(), alpha_);
  shader->setLocalParam(3, 1.0/vp[2], 1.0/vp[3], pany, pany);
  if(pbuffer)
    shader->setLocalParam(4, 1.0/pbuffer->width(), 1.0/pbuffer->height(), 
			  0.0, 0.0);
  
  GLdouble mid[4] = { color_.r(), color_.g(), color_.b(), alpha_ };
  GLdouble edg[4] = { color_.r(), color_.g(), color_.b(), 0 };
  glShadeModel(GL_SMOOTH);    
  double range = 1.0;
  if (value_range_.first < value_range_.second)
    range = value_range_.second - value_range_.first;
  for (unsigned int s = 0; s < strokes_.size(); ++s)
  {
    const double halfx = strokes_[s].first/range;
    Stroke &stroke = strokes_[s].second;
    const unsigned int coordinates = stroke.size();
    for (unsigned c = 1; c < coordinates; ++c)
    {
      glBegin(GL_QUADS);
      glColor4dv(edg);
      glVertex2d(stroke[c-1].first-halfx, stroke[c-1].second);
      
      glColor4dv(mid);
      glVertex2d(stroke[c-1].first, stroke[c-1].second);
      
      glColor4dv(mid);
      glVertex2d(stroke[c].first, stroke[c].second);
      
      glColor4dv(edg);
      glVertex2d(stroke[c].first-halfx, stroke[c].second);
      
      
      glColor4dv(mid);
      glVertex2d(stroke[c-1].first, stroke[c-1].second);
      
      glColor4dv(edg);
      glVertex2d(stroke[c-1].first+halfx, stroke[c-1].second);
      
      glColor4dv(edg);
      glVertex2d(stroke[c].first+halfx, stroke[c].second);
      
      glColor4dv(mid);
      glVertex2d(stroke[c].first, stroke[c].second);
      glEnd();
      if (pbuffer && (s < strokes_.size()-1)) {
	pbuffer->release();
	pbuffer->swapBuffers();
	pbuffer->bind();
      }
    }
  }
  glShadeModel(GL_FLAT);
  shader->release();
  un_normalize();
  CHECK_OPENGL_ERROR("paintcm2widget rasterize end");
}

void
PaintCM2Widget::splat(Array3<float> &data, double width, int x0, int y0) {
  double range = 1.0;
  if (value_range_.first < value_range_.second)
    range = value_range_.second - value_range_.first;

  const int wid = Round(data.dim2()*(width/range));
  float r = color_.r();
  float g = color_.g();
  float b = color_.b();
  float a = alpha_;
  float oma = 1.0 - a;
  const bool flat = shadeType_ == CM2_SHADE_FLAT;
  for (int y = y0-wid; y <= y0+wid; ++y)
    if (y >= 0 && y < data.dim2())
      if (flat) {
	data(x0, y, 0) = Clamp(oma * data(x0, y, 0) + r, 0.0, 1.0);
	data(x0, y, 1) = Clamp(oma * data(x0, y, 1) + g, 0.0, 1.0);
	data(x0, y, 2) = Clamp(oma * data(x0, y, 2) + b, 0.0, 1.0);
	data(x0, y, 3) = Clamp(oma * data(x0, y, 3) + a, 0.0, 1.0);
      } else {
	a = float(alpha_*(wid-fabs(float(y-y0)))/wid);
	oma = 1.0 - a;
	data(x0, y, 0) = Clamp(oma * data(x0, y, 0) + r*a, 0.0, 1.0);
	data(x0, y, 1) = Clamp(oma * data(x0, y, 1) + g*a, 0.0, 1.0);
	data(x0, y, 2) = Clamp(oma * data(x0, y, 2) + b*a, 0.0, 1.0);
	data(x0, y, 3) = Clamp(oma * data(x0, y, 3) + a, 0.0, 1.0);
      }  
}

// Bressenhams line algorithm modified to only draw when the x pos changes
void 
PaintCM2Widget::line(Array3<float> &data, double width,
		     int x0, int y0, int x1, int y1, bool first)
{
  if (x0 < 0 || x0 >= data.dim1() || 
      x1 < 0 || x1 >= data.dim1() || 
      y0 < 0 || y0 >= data.dim2() || 
      y1 < 0 || y1 >= data.dim2()) return;
  int dy = y1 - y0;
  int dx = x1 - x0;
  int sx = 1;
  int sy = 1;
  int frac = 0;
  bool do_splat = false;
  if (dy < 0) { 
    dy = -dy;
    sy = -1; 
  } 
  if (dx < 0) { 
    dx = -dx;  
    sx = -1;
  } 
  dy <<= 1;
  dx <<= 1;
  if (first) splat(data, width, x0, y0);
  if (dx > dy) {
    frac = dy - (dx >> 1);
    while (x0 != x1) {
      if (frac >= 0) {
	y0 += sy;
	frac -= dx;
      }
      x0 += sx;
      frac += dy;
      splat(data, width, x0, y0);
    }
  } else {
    frac = dx - (dy >> 1);
    while (y0 != y1) {
      if (frac >= 0) {
	x0 += sx;
	frac -= dy;
	do_splat = true;
      }
      y0 += sy;
      frac += dx;
      if (do_splat) {
	splat(data, width, x0, y0);
	do_splat = false;
      }
    }
  }
}

void
PaintCM2Widget::rasterize(Array3<float>& array)
{
  if(!onState_) return;
  const float offset = -value_range_.first;
  const float scale = array.dim2()/(value_range_.second-value_range_.first);

  for (unsigned int s = 0; s < strokes_.size(); ++s)
  {
    double width = strokes_[s].first;
    Stroke &stroke = strokes_[s].second;
    const unsigned int coordinates = stroke.size();
    if (coordinates == 1)
      splat(array, width,
	    Floor(stroke[0].second* array.dim1()), 
	    Floor((stroke[0].first+offset)*scale));
    else
      for (unsigned c = 1; c < coordinates; ++c)
	line(array, width,
	     Floor(stroke[c-1].second * array.dim1()), 
	     Floor((stroke[c-1].first+offset)*scale),
	     Floor(stroke[c].second   * array.dim1()), 
	     Floor((stroke[c].first+offset)*scale),
	     (c == 1));
  }
}

void
PaintCM2Widget::draw()
{
  // no widget controls to draw.
}

void
PaintCM2Widget::add_stroke(double width)
{
  double range = 1.0;
  if (value_range_.first < value_range_.second)
    range = value_range_.second - value_range_.first;

  if (width < 0.0)
    width = range/35.0;
  strokes_.push_back(make_pair(width,Stroke()));
}

void
PaintCM2Widget::add_coordinate(const Coordinate &coordinate)
{
  if (strokes_.empty()) return;

  Stroke &stroke = strokes_.back().second;
  // filter duplicate points
  if (!stroke.empty() && 
      coordinate.first == stroke.back().first &&
      coordinate.second == stroke.back().second) return;
      
  stroke.push_back(coordinate);
}

bool
PaintCM2Widget::pop_stroke()
{
  if (strokes_.empty()) return false;
  strokes_.pop_back();
  return true;
}

void
PaintCM2Widget::normalize()
{
  const float offset = -value_range_.first;
  const float scale = 1.0/(value_range_.second-value_range_.first);
  for (unsigned int s = 0; s < strokes_.size(); ++s)
    for (unsigned int c = 0; c < strokes_[s].second.size(); ++c)
      strokes_[s].second[c].first = 
	(strokes_[s].second[c].first + offset) * scale;
}

void
PaintCM2Widget::un_normalize()
{
  const float offset = -value_range_.first;
  const float scale = 1.0/(value_range_.second-value_range_.first);
  for (unsigned int s = 0; s < strokes_.size(); ++s)
    for (unsigned int c = 0; c < strokes_[s].second.size(); ++c)
      strokes_[s].second[c].first = strokes_[s].second[c].first/scale - offset;
}
