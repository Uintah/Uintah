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
 *  GeomVertexPrim.h: Base class for primitives that use the Vertex class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_VertexPrim_h
#define SCI_Geom_VertexPrim_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Datatypes/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace SCIRun {

struct GeomVertex : public Persistent {
  Point p;
  GeomVertex(const Point& p);
  GeomVertex(const GeomVertex&);
  virtual ~GeomVertex();
  virtual GeomVertex* clone();
#ifdef SCI_OPENGL
  virtual void emit_all(DrawInfoOpenGL* di);
  void emit_point(DrawInfoOpenGL* di);
  virtual void emit_matl(DrawInfoOpenGL* di);
  virtual void emit_normal(DrawInfoOpenGL* di);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
#if 1
  void* operator new(size_t);
  void operator delete(void*, size_t);
#endif
};

struct GeomNVertex : public GeomVertex {
  Vector normal;
  GeomNVertex(const Point& p, const Vector& normal);
  GeomNVertex(const GeomNVertex&);
  virtual GeomVertex* clone();
  virtual ~GeomNVertex();
#ifdef SCI_OPENGL
  virtual void emit_all(DrawInfoOpenGL* di);
  virtual void emit_normal(DrawInfoOpenGL* di);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
#if 1
  void* operator new(size_t);
  void operator delete(void*, size_t);
#endif
};

struct GeomNMVertex : public GeomNVertex {
  MaterialHandle matl;
  GeomNMVertex(const Point& p, const Vector& normal,
	       const MaterialHandle& matl);
  GeomNMVertex(const GeomNMVertex&);
  virtual GeomVertex* clone();
  virtual ~GeomNMVertex();
#ifdef SCI_OPENGL
  virtual void emit_all(DrawInfoOpenGL* di);
  virtual void emit_matl(DrawInfoOpenGL* di);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
#if 1
  void* operator new(size_t);
  void operator delete(void*, size_t);
#endif
};

struct GeomMVertex : public GeomVertex {
  MaterialHandle matl;
  GeomMVertex(const Point& p, const MaterialHandle& matl);
  GeomMVertex(const GeomMVertex&);
  virtual GeomVertex* clone();
  ~GeomMVertex();
#ifdef SCI_OPENGL
  virtual void emit_all(DrawInfoOpenGL* di);
  virtual void emit_matl(DrawInfoOpenGL* di);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
#if 1
  void* operator new(size_t);
  void operator delete(void*, size_t);
#endif
};

struct GeomCVertex : public GeomVertex {
  Color color;
  GeomCVertex(const Point& p, const Color& clr);
  GeomCVertex(const GeomCVertex&);
  virtual GeomVertex* clone();
  ~GeomCVertex();
#ifdef SCI_OPENGL
  virtual void emit_all(DrawInfoOpenGL* di);
  virtual void emit_matl(DrawInfoOpenGL* di);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
#if 1
  void* operator new(size_t);
  void operator delete(void*, size_t);
#endif
};

class SCICORESHARE GeomVertexPrim : public GeomObj {
public:
  Array1<double> times;
  Array1<GeomVertex*> verts;

  GeomVertexPrim();
  GeomVertexPrim(const GeomVertexPrim&);
  virtual ~GeomVertexPrim();

  virtual void get_bounds(BBox&);
    
  void add(const Point&);
  void add(const Point&, const Vector&);
  void add(const Point&, const MaterialHandle&);
  void add(const Point&, const Color&);
  void add(const Point&, const Vector&, const MaterialHandle&);
  void add(GeomVertex*);
  void add(double time, GeomVertex*);

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

void Pio(Piostream&, GeomVertex*&);

} // End namespace SCIRun

#endif /* SCI_Geom_VertexPrim_h */
