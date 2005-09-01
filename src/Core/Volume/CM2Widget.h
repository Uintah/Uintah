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

#include <Core/Volume/CM2Shader.h>

#include <Core/Datatypes/Color.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geometry/BBox.h>

#include <string>
using std::string;

namespace SCIRun {

class CM2ShaderFactory;
class Pbuffer;

typedef pair<float, float> range_t;

class CM2Widget : public Datatype
{
public:
  CM2Widget();
  CM2Widget(CM2Widget& copy);
  virtual ~CM2Widget();  
  virtual CM2Widget*	clone() = 0;
  
  // Draw widget frame
  virtual void		draw() = 0;
  // Draw widget to hardware GL buffer
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer *buf) = 0;
  // Draw widget to software GL buffer
  virtual void		rasterize(Array3<float>& array) = 0;

  // Getters/Setters
  virtual range_t	get_value_range() { return value_range_; }
  virtual void		set_value_range(range_t range);
  virtual string	get_name() { return name_; }
  virtual void		set_name(string name) { name_ = name; }
  virtual int		get_shadeType() { return shadeType_; }
  virtual void		set_shadeType(int type) { shadeType_ = type; }
  virtual int		get_onState() { return onState_; }
  virtual void		set_onState(int state) { onState_ = state; }
  virtual bool		get_faux() { return faux_; }
  virtual void		set_faux(bool faux) { faux_ = faux; }
  virtual Color		get_color() const { return color_; }
  virtual void		set_color(const Color &c) { color_ = c; }
  virtual float		get_alpha() const { return alpha_; }
  virtual void		set_alpha(float alpha) { alpha_ = alpha; }

  // selection
  virtual void		select(int obj) { selected_ = obj; }
  virtual void		unselect_all() { selected_ = 0; }
  virtual string	tk_cursorname(int) { return "left_ptr"; };

  // Pure Virtual Methods below

  // Picking/movement
  virtual int		pick1(int x, int y, int w, int h) = 0;
  virtual int		pick2(int x, int y, int w, int h, int m) = 0;
  virtual void		move(int x, int y, int w, int h) = 0;
  virtual void		release(int x, int y, int w, int h) = 0;


  // State management to/from a string
  virtual string	tcl_pickle() = 0;
  virtual void		tcl_unpickle(const string &p) = 0;

  // State management to/from disk
  virtual void		io(Piostream &stream) = 0;
  static PersistentTypeID type_id;

protected:
  virtual void		normalize() {}
  virtual void		un_normalize() {}
  virtual void		draw_thick_gl_line(double x1, double y1, 
					   double x2, double y2,
					   double r, double g, double b);
  virtual void		draw_thick_gl_point(double x1, double y1,
					    double r, double g, double b);

  string		name_;
  Color			color_;
  float			alpha_;
  int			selected_;
  int			shadeType_;
  int			onState_;
  bool			faux_;
  HSVColor		last_hsv_;
  range_t		value_range_;
};

typedef LockingHandle<CM2Widget> CM2WidgetHandle;


class ClippingCM2Widget : public CM2Widget
{
public:
  ClippingCM2Widget();
  ClippingCM2Widget(ClippingCM2Widget& copy);
  ClippingCM2Widget(vector<BBox> &bboxes);
  ~ClippingCM2Widget();
  vector<BBox> &	bboxes() { return bboxes_; }

  virtual CM2Widget*	clone();
  
  virtual void		draw(){}
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer){}
  virtual void		rasterize(Array3<float>& array){}
  virtual int		pick1(int x, int y, int w, int h) { return 0; }
  virtual int		pick2(int x, int y, int w, int h, int m){ return 0; }
  virtual void		move(int x, int y, int w, int h){}
  virtual void		release(int x, int y, int w, int h){}
  virtual string	tcl_pickle() { return ""; };
  virtual void		tcl_unpickle(const std::string &p) {};

  virtual void		io(Piostream &stream);
  static PersistentTypeID type_id;
protected:
  vector<BBox>		bboxes_;
};
  



class TriangleCM2Widget : public CM2Widget
{
public:
  TriangleCM2Widget();
  TriangleCM2Widget(TriangleCM2Widget& copy);
  TriangleCM2Widget(float base, float top_x, float top_y,
                    float width, float bottom);
  ~TriangleCM2Widget();
  virtual CM2Widget*	clone();
  
  virtual void		draw();
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer);
  virtual void		rasterize(Array3<float>& array);
  virtual int		pick1(int x, int y, int w, int h);
  virtual int		pick2(int x, int y, int w, int h, int m);
  virtual void		move(int x, int y, int w, int h);
  virtual void		release(int x, int y, int w, int h);
  virtual string	tk_cursorname(int obj);
  virtual string	tcl_pickle();
  virtual void		tcl_unpickle(const std::string &p);

  virtual void		io(Piostream &stream);
  static PersistentTypeID type_id;
protected:
  void			normalize();
  void			un_normalize();  

  float			base_;
  float			top_x_;
  float			top_y_;
  float			width_;
  float			bottom_;

  // Used by picking.
  float			last_x_;
  float			last_y_;
  float			last_width_;
  int			pick_ix_;
  int			pick_iy_;
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
  RectangleCM2Widget(RectangleCM2Widget& copy);
  RectangleCM2Widget(CM2RectangleType type, float left_x, float left_y,
                     float width, float height, float offset);
  ~RectangleCM2Widget();
  virtual CM2Widget*	clone();

  virtual void		draw();
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer);
  virtual void		rasterize(Array3<float>& array);
  virtual int		pick1(int x, int y, int w, int h);
  virtual int		pick2(int x, int y, int w, int h, int m);
  virtual void		move(int x, int y, int w, int h);
  virtual void		release(int x, int y, int w, int h);
  virtual string	tk_cursorname(int obj);
  virtual string	tcl_pickle();
  virtual void		tcl_unpickle(const std::string &p);

  virtual void		io(Piostream &stream);
  static PersistentTypeID type_id;
protected:
  void			normalize();
  void			un_normalize();
  
  CM2RectangleType	type_;
  float			left_x_;
  float			left_y_;
  float			width_;
  float			height_;
  float			offset_;

  // Used by picking.
  float			last_x_;
  float			last_y_;
  int			pick_ix_;
  int			pick_iy_;
};



class ColorMapCM2Widget : public RectangleCM2Widget
{
public:
  ColorMapCM2Widget();
  ColorMapCM2Widget(ColorMapCM2Widget& copy);
  ColorMapCM2Widget(CM2RectangleType type, float left_x, float left_y,
                     float width, float height, float offset);
  ~ColorMapCM2Widget();
  virtual CM2Widget*	clone();

  virtual void		draw();
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer);
  virtual void		rasterize(Array3<float>& array);
  virtual string	tcl_pickle();
  virtual void		tcl_unpickle(const std::string &p);

  virtual void		io(Piostream &stream);
  ColorMapHandle	get_colormap();
  void			set_colormap(ColorMapHandle &cmap);
  static PersistentTypeID type_id;
protected:
  ColorMapHandle	colormap_;
};


// The image widget cannot be manipulated, only drawn.
class ImageCM2Widget : public CM2Widget
{
public:
  ImageCM2Widget();
  ImageCM2Widget(ImageCM2Widget& copy);
  ImageCM2Widget(NrrdDataHandle p);
  ~ImageCM2Widget();

  virtual CM2Widget* clone();

  virtual void		draw();
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer);
  virtual void		rasterize(Array3<float>& array);
  virtual int		pick1(int x, int y, int w, int h) { return 0; }
  virtual int		pick2(int x, int y, int w, int h, int m) { return 0; }
  virtual void		move(int x, int y, int w, int h) {}
  virtual void		release(int x, int y, int w, int h) {}
  virtual string	tcl_pickle() { return "i"; }
  virtual void		tcl_unpickle(const std::string &) {}

  virtual void		io(Piostream &stream);
  static PersistentTypeID type_id;
protected:
  Nrrd*			resize(int w, int h);  // nrrdSpatialResample ...
  NrrdDataHandle	pixels_;
};



class PaintCM2Widget : public CM2Widget
{
public:
  typedef pair<double, double>		Coordinate;
  typedef vector<Coordinate>		Stroke;
  typedef vector<pair<double, Stroke> >	Strokes;

  PaintCM2Widget();
  ~PaintCM2Widget();
  PaintCM2Widget(PaintCM2Widget& copy);
  virtual CM2Widget*	clone();
  
  virtual void		draw();
  virtual void		rasterize(CM2ShaderFactory& factory, Pbuffer* pbuffer);
  virtual void		rasterize(Array3<float>& array);
  virtual int		pick1(int x, int y, int w, int h) { return 0; }
  virtual int		pick2(int x, int y, int w, int h, int m) { return 0; }
  virtual void		move(int x, int y, int w, int h) {}
  virtual void		release(int x, int y, int w, int h) {};
  virtual string	tcl_pickle() { return "p"; }
  virtual void		tcl_unpickle(const std::string &) {}

  virtual void		io(Piostream &stream);
  static PersistentTypeID type_id;

  void			add_stroke(double width = -1.0);
  bool			pop_stroke();
  void			add_coordinate(const Coordinate &);
protected:
  void			normalize();
  void			un_normalize();  

  void			line(Array3<float> &, double, 
			     int, int, int, int, bool first);
  void			splat(Array3<float> &, double, int, int);
  Strokes		strokes_;
};





} // End namespace SCIRun

#endif // CM2Widget_h
