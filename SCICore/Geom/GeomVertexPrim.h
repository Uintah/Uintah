
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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>
#include <stdlib.h>

namespace SCICore {
namespace GeomSpace {

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

#if 0
    // These are here for emiting to a file.
    virtual void io_emit_all(ostream& out, const clString& format,
    void io_emit_point(ostream& out, const clString& format, 
    virtual void io_emit_matl(ostream& out, const clString& format, 
    virtual void io_emit_normal(ostream& out, const clString& format, 
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
    virtual void get_bounds(BSphere&);
    
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

  void Pio(Piostream&, GeomSpace::GeomVertex*&);
} // End namespace GeomSpace
} // End namespace SCICore


//
// $Log$
// Revision 1.2  1999/08/17 06:39:18  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:48  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:10  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:07  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_VertexPrim_h */
