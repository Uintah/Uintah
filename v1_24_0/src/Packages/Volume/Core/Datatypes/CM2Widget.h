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
//    File   : CM2Widget.h
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:12 2004

#ifndef CM2Widget_h
#define CM2Widget_h

#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Color.h>
#include <string>

namespace Volume {

class CM2ShaderFactory;
class Pbuffer;

class CM2Widget
{
public:
  CM2Widget();
  virtual ~CM2Widget();
  CM2Widget(CM2Widget& copy);
  
  // appearance
  virtual void draw() = 0;
  virtual void rasterize(CM2ShaderFactory& factory, bool faux, Pbuffer* pbuffer) = 0;
  virtual void rasterize(SCIRun::Array3<float>& array, bool faux) = 0;
  virtual CM2Widget* clone() = 0;
  
  // behavior
  virtual int pick1 (int x, int y, int w, int h) = 0;
  virtual int pick2 (int x, int y, int w, int h, int m) = 0;
  virtual void move (int obj, int x, int y, int w, int h) = 0;
  virtual void release (int obj, int x, int y, int w, int h) = 0;

  virtual std::string tcl_pickle() = 0;
  virtual void tcl_unpickle(const std::string &p) = 0;

  void select(int obj) { selected_ = obj; }
  void unselect_all() { selected_ = 0; }
  SCIRun::Color color() const { return color_; }
  void set_color(const SCIRun::Color &c) { color_ = c; }
  float alpha() const { return alpha_; }
  void set_alpha(float a);

protected:
  void selectcolor(int obj);

  SCIRun::Color line_color_;
  float line_alpha_;
  SCIRun::Color selected_color_;
  float selected_alpha_;
  float thin_line_width_;
  float thick_line_width_;
  float point_size_;
  SCIRun::Color color_;
  float alpha_;
  int selected_;
  SCIRun::HSVColor last_hsv_;
};

class TriangleCM2Widget : public CM2Widget
{
public:
  TriangleCM2Widget();
  TriangleCM2Widget(float base, float top_x, float top_y,
                    float width, float bottom);
  ~TriangleCM2Widget();
  TriangleCM2Widget(TriangleCM2Widget& copy);

  virtual CM2Widget* clone();
  
  // appearance
  void draw();
  void rasterize(CM2ShaderFactory& factory, bool faux, Pbuffer* pbuffer);
  void rasterize(SCIRun::Array3<float>& array, bool faux);
  
  // behavior
  virtual int pick1 (int x, int y, int w, int h);
  virtual int pick2 (int x, int y, int w, int h, int m);
  virtual void move (int obj, int x, int y, int w, int h);
  virtual void release (int obj, int x, int y, int w, int h);

  virtual std::string tcl_pickle();
  virtual void tcl_unpickle(const std::string &p);

protected:
  float base_;
  float top_x_, top_y_;
  float width_;
  float bottom_;

  // Used by picking.
  float last_x_, last_y_, last_width_;
  int pick_ix_, pick_iy_;
};

enum CM2RectangleType
{
  CM2_RECTANGLE_1D = 0,
  CM2_RECTANGLE_ELLIPSOID = 1
};

class RectangleCM2Widget : public CM2Widget
{
public:
  RectangleCM2Widget();
  RectangleCM2Widget(CM2RectangleType type, float left_x, float left_y,
                     float width, float height, float offset);
  ~RectangleCM2Widget();
  RectangleCM2Widget(RectangleCM2Widget& copy);

  virtual CM2Widget* clone();

  // appearance
  void draw();
  void rasterize(CM2ShaderFactory& factory, bool faux, Pbuffer* pbuffer);
  void rasterize(SCIRun::Array3<float>& array, bool faux);
  
  // behavior
  virtual int pick1 (int x, int y, int w, int h);
  virtual int pick2 (int x, int y, int w, int h, int m);
  virtual void move (int obj, int x, int y, int w, int h);
  virtual void release (int obj, int x, int y, int w, int h);
  
  virtual std::string tcl_pickle();
  virtual void tcl_unpickle(const std::string &p);

protected:
  CM2RectangleType type_;
  float left_x_, left_y_;
  float width_, height_, offset_;

  // Used by picking.
  float last_x_, last_y_;
  int pick_ix_, pick_iy_;
};

} // End namespace Volume

#endif // CM2Widget_h
