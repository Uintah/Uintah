
/*
 *  VertexPrim.h: Base class for primitives that use the Vertex class
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

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Color.h>
#include <Classlib/Array1.h>
#include <Geometry/Point.h>

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
#if 0
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
#if 0
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
#if 0
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
#if 0
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
#if 0
    void* operator new(size_t);
    void operator delete(void*, size_t);
#endif
};

void Pio(Piostream&, GeomVertex*&);

class GeomVertexPrim : public GeomObj {
public:
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

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Geom_VertexPrim_h */



