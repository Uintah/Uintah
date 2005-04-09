
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

#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geom/Color.h>
#include <Multitask/ITC.h>

#include <stdlib.h>	// For size_t

class GeomLine : public GeomObj {
public:
    Point p1, p2;

    GeomLine(const Point& p1, const Point& p2);
    GeomLine(const GeomLine&);
    virtual ~GeomLine();
    virtual GeomObj* clone();

    void* operator new(size_t);
    void operator delete(void*, size_t);

    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void objdraw(DrawInfoX11*, Material*);
    virtual double depth(DrawInfoX11*);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

class GeomLines : public GeomObj {
public:
  Array1<Point> pts;
  GeomLines();
  GeomLines(const GeomLines&);

  void add(const Point&, const Point&);
  virtual ~GeomLines();
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

// can generate "lit" streamlines this way

class TexGeomLines : public GeomObj {
protected:
  Array1<unsigned char>  tmap1d; // 1D texture - should be in Salmon?
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

#endif /* SCI_Geom_Line_h */
