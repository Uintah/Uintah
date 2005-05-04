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

class GeomVertexPrim : public GeomObj {
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
