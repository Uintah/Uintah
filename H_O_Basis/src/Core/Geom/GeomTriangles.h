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
 *  GeomTriangles.h: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Triangles_h
#define SCI_Geom_Triangles_h 1

#include <Core/Geom/GeomVertexPrim.h>

namespace SCIRun {

class Color;

class GeomTriangles : public GeomVertexPrim {
private:
    void add(const Point&);
    void add(const Point&, const Vector&);
    void add(const Point&, const MaterialHandle&);
    void add(const Point&, const Vector&, const MaterialHandle&);
    void add(GeomVertex*);
public:
    Array1<Vector> normals;
    GeomTriangles();
    GeomTriangles(const GeomTriangles&);
    virtual ~GeomTriangles();

    int size(void);
    void add(const Point&, const Point&, const Point&);
    void add(const Point&, const Vector&,
	     const Point&, const Vector&,
	     const Point&, const Vector&);
    void add(const Point&, const MaterialHandle&,
	     const Point&, const MaterialHandle&,
	     const Point&, const MaterialHandle&);
    void add(const Point&, const Color&,
	     const Point&, const Color&,
	     const Point&, const Color&);
    void add(const Point&, const Vector&, const MaterialHandle&,
	     const Point&, const Vector&, const MaterialHandle&,
	     const Point&, const Vector&, const MaterialHandle&);
    void add(GeomVertex*, GeomVertex*, GeomVertex*);
    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomFastTriangles : public GeomObj {
protected:
  vector<float> points_;
  vector<unsigned char> colors_;
  vector<float> indices_;
  vector<float> normals_;
  vector<float> face_normals_;
  MaterialHandle material_;

public:
  GeomFastTriangles();
  GeomFastTriangles(const GeomFastTriangles&);
  virtual ~GeomFastTriangles();
  virtual GeomObj* clone();

  int size(void);
  void add(const Point &p0, const Point &p1, const Point &p2);
  void add(const Point &p0, const Vector &n0,
	   const Point &p1, const Vector &n1,
	   const Point &p2, const Vector &n2);
  void add(const Point &p0, const MaterialHandle &m0,
	   const Point &p1, const MaterialHandle &m1,
	   const Point &p2, const MaterialHandle &m2);
  void add(const Point &p0, double cindex0,
	   const Point &p1, double cindex1,
	   const Point &p2, double cindex2);
  void add(const Point &p0, const Vector &n0, const MaterialHandle &m0,
	   const Point &p1, const Vector &n1, const MaterialHandle &m1,
	   const Point &p2, const Vector &n2, const MaterialHandle &m2);
  void add(const Point &p0, const Vector &n0, double i0,
	   const Point &p1, const Vector &n1, double i1,
	   const Point &p2, const Vector &n2, double i2);


  virtual void get_bounds(BBox& bb);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};



class GeomTranspTriangles : public GeomFastTriangles
{
protected:
  vector<unsigned int> xlist_;
  vector<unsigned int> ylist_;
  vector<unsigned int> zlist_;
  bool xreverse_;
  bool yreverse_;
  bool zreverse_;

public:
  GeomTranspTriangles();
  GeomTranspTriangles(const GeomTranspTriangles&);
  virtual ~GeomTranspTriangles();
  virtual GeomObj* clone();

  void SortPolys();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


class GeomTrianglesP: public GeomObj {
protected:
    Array1<float> points;
    Array1<float> normals;

    int has_color;
    double r,g,b;  // actual color values...
public:
    GeomTrianglesP();
    virtual ~GeomTrianglesP();

    void SetColor(double rr, double gg, double bb) {
      has_color =1;
      r = rr; g = gg; b = bb;
    };

    int size(void);

    int add(const Point&, const Point&, const Point&);
    
    // below is a virtual function - makes some things easier...
    virtual int vadd(const Point&, const Point&, const Point&);

    void reserve_clear(int);   // reserves storage... and clears

    virtual GeomObj* clone();

    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void get_triangles( Array1<float> &v);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class GeomTrianglesPT1d : public GeomTrianglesP {
protected:
  Array1<float> scalars;
public:

  GeomTrianglesPT1d();
  virtual ~GeomTrianglesPT1d();

  int add(const Point&, const Point&, const Point&,
	  const float&, const float&, const float&);
  
  unsigned char* cmap;

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomTranspTrianglesP : public GeomTrianglesP {
protected:
  double alpha_;

  bool sorted_p_;

  vector<pair<float, unsigned int> > xlist_;
  vector<pair<float, unsigned int> > ylist_;
  vector<pair<float, unsigned int> > zlist_;

public:
  GeomTranspTrianglesP();
  GeomTranspTrianglesP(double);
  virtual ~GeomTranspTrianglesP();

  // below computes the x/y/z center points as well...
  // should only use below...
  virtual int vadd(const Point&, const Point&, const Point&);

  // function below sorts the polygons...
  void SortPolys();

  // function below merges in another "list" - also clears
  //void MergeStuff(GeomTranspTrianglesP*);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomTrianglesPC: public GeomTrianglesP {
    Array1<float> colors;
public:
    GeomTrianglesPC();
    virtual ~GeomTrianglesPC();

    int add(const Point&, const Color&,
	    const Point&, const Color&,
	    const Point&, const Color&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class GeomTrianglesVP: public GeomObj {
protected:
    Array1<float> points;
    Array1<float> normals;
public:
    GeomTrianglesVP();
    virtual ~GeomTrianglesVP();

    int size(void);

    int add(const Point&, const Vector&, 
	    const Point&, const Vector&, 
	    const Point&, const Vector&);
    
    void reserve_clear(int);   // reserves storage... and clears

    virtual GeomObj* clone();

    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class GeomTrianglesVPC: public GeomTrianglesVP {
    Array1<float> colors;
public:
    GeomTrianglesVPC();
    virtual ~GeomTrianglesVPC();

    int add(const Point&, const Vector&, const Color&,
	    const Point&, const Vector&, const Color&,
	    const Point&, const Vector&, const Color&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Triangles_h */

