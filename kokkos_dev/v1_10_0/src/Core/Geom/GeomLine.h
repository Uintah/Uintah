/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

class SCICORESHARE GeomLine : public GeomObj {
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
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
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
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};


class SCICORESHARE GeomCLines : public GeomObj {
protected:
  double line_width_;
  vector<float> points_;
  vector<unsigned char> colors_;

public:
  GeomCLines();
  GeomCLines(const GeomCLines&);

  void add(const Point &p0, MaterialHandle c0,
	   const Point &p1, MaterialHandle c1);
  void setLineWidth(float val) { line_width_ = val; }

  virtual ~GeomCLines();
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};


class SCICORESHARE GeomTranspLines : public GeomCLines {
protected:
  vector<unsigned int> xindices_;
  vector<unsigned int> yindices_;
  vector<unsigned int> zindices_;

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
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};


// can generate "lit" streamlines this way

class SCICORESHARE TexGeomLines : public GeomObj {
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
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};

} // End namespace SCIRun


#endif /* SCI_Geom_Line_h */
