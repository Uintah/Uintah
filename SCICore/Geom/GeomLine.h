
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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Multitask/ITC.h>

#include <stdlib.h>	// For size_t

namespace SCICore {
namespace GeomSpace {

using SCICore::Multitask::Mutex;

class SCICORESHARE GeomLine : public GeomObj {
public:
    Point p1, p2;

    GeomLine(const Point& p1, const Point& p2);
    GeomLine(const GeomLine&);
    virtual ~GeomLine();
    virtual GeomObj* clone();

    void* operator new(size_t);
    void operator delete(void*, size_t);

    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomLines : public GeomObj {
public:
  Array1<Point> pts;
  GeomLines();
  GeomLines(const GeomLines&);

  void add(const Point&, const Point&);
  virtual ~GeomLines();
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

// can generate "lit" streamlines this way

class SCICORESHARE TexGeomLines : public GeomObj {
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
// Revision 1.3  1999/08/17 23:50:22  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:09  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:40  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:05  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:58  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_Line_h */
