
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

#include <Geom/GeomVertexPrim.h>

namespace SCICore {
namespace GeomSpace {

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
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
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
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
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
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};


class GeomTranspTrianglesP : public GeomTrianglesP {
protected:
  Array1<int> xlist;
  Array1<int> ylist;
  Array1<int> zlist;

  Array1<float> xc;
  Array1<float> yc;
  Array1<float> zc;

  double alpha;
  int sorted;

  // also save off some of the other stuff...
  int list_pos; // posistion in the list
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
  void MergeStuff(GeomTranspTrianglesP*);

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
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
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
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
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
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:47  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:10  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:06  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_Triangles_h */

