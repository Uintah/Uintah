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



/*
 *  Line.h:  Line object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Line_h
#define SCI_Geom_Line_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Thread/Mutex.h>

#include <stdlib.h>	// For size_t

namespace SCIRun {

class GeomLine : public GeomObj {
public:
  Point p1, p2;
  float lineWidth_;

  GeomLine(const Point& p1, const Point& p2);
  GeomLine(const GeomLine&);
  virtual ~GeomLine();
  virtual GeomObj* clone();

  void* operator new(size_t);
  void operator delete(void*, size_t);

  virtual void get_bounds(BBox&);
  void setLineWidth(float val);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

class GeomLines : public GeomObj {
protected:
  double line_width_;
  vector<float> points_;
  vector<unsigned char> colors_;
  vector<float> indices_;

public:
  GeomLines();
  GeomLines(const GeomLines&);

  void add(const Point &p0, const Point &p1);
  void add(const Point &p0, const MaterialHandle &c0,
	   const Point &p1, const MaterialHandle &c1);
  void add(const Point &p0, double cindex0,
	   const Point &p1, double cindex1);
  void add(const Point &p0, const MaterialHandle &c0, double cindex0,
	   const Point &p1, const MaterialHandle &c1, double cindex1);
  void setLineWidth(float val) { line_width_ = val; }

  unsigned int size() { return (int)points_.size() / 6; }

  virtual ~GeomLines();
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomTranspLines : public GeomLines {
protected:
  vector<unsigned int> xindices_;
  vector<unsigned int> yindices_;
  vector<unsigned int> zindices_;
  bool xreverse_;
  bool yreverse_;
  bool zreverse_;

public:
  GeomTranspLines();
  GeomTranspLines(const GeomTranspLines&);

  virtual ~GeomTranspLines();
  virtual GeomObj* clone();

  void sort();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


// can generate "lit" streamlines this way

class TexGeomLines : public GeomObj {
protected:
  Array1<unsigned char>  tmap1d; // 1D texture - should be in Viewer?
  int tmapid;                    // id for this texture map
  int tex_per_seg;               // 0 if batched...

public:
  Mutex mutex;
  Array1<Point>   pts;
  Array1<Vector>  tangents;  // used in 1D illumination model...
  Array1<double>  times;
  Array1<Colorub>   colors;    // only if you happen to have color...
  

  Array1<int>     sorted; // x,y and z sorted list
  double alpha;                  // 2D texture wi alpha grad/mag would be nice

  TexGeomLines();
  TexGeomLines(const TexGeomLines&);

  void add(const Point&, const Point&, double scale); // for one 
  void add(const Point&, const Vector&, const Colorub&); // for one 
  
  void batch_add(Array1<double>& t, Array1<Point>&); // adds a bunch
  // below is for when you also have color...
  void batch_add(Array1<double>& t, Array1<Point>&,Array1<Color>&);


  void CreateTangents(); // creates tangents from curve...

  void SortVecs();  // creates sorted lists...

  virtual ~TexGeomLines();
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomCLineStrips : public GeomObj {
protected:
  double line_width_;
  vector<vector<float> > points_;
  vector<vector<unsigned char> > colors_;

public:
  GeomCLineStrips();
  GeomCLineStrips(const GeomCLineStrips&);
  virtual ~GeomCLineStrips();

  void add(const vector<Point> &p, const vector<MaterialHandle> &c);
  void add(const Point &p, const MaterialHandle c);
  void newline();
  void setLineWidth(float val) { line_width_ = val; }

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};


} // End namespace SCIRun


#endif /* SCI_Geom_Line_h */
