#ifdef BROKEN
/*
 *  GeomGrid.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Grid_h
#define SCI_Geom_Grid_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/Array2.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomGrid : public GeomObj {
    Array1<float> data;
    int have_matls;
    int have_normals;
    Point corner;
    Vector u, v, w;
    int stride;
    int offset;
    int vstride;
    void adjust();
    void resize();
    int nu, nv;
    Vector uu,vv;
public:
  enum Format {
	Regular, WithMaterials, WithNormals, WithNormAndMatl,
    };
    Format format;
    GeomGrid(int, int, const Point&, const Vector&, const Vector&,
	     Format);
    GeomGrid(const GeomGrid&);
    virtual ~GeomGrid();

    virtual GeomObj* clone();

    inline void set(int, int, double);
    inline void set(int, int, double, const MaterialHandle&);
    inline void set(int, int, double, const Vector&);
    inline void setn(int, int, double);
    inline void set(int, int, double, const Vector&, const MaterialHandle&);
    void compute_normals();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

inline void GeomGrid::set(int i, int j, double h)
{
    float* p=&data[j*nu*3+i*3];
    *p++=i;
    *p++=j;
    *p++=h;
}

inline void GeomGrid::set(int i, int j, double h, const Vector& normal)
{
    float* p=&data[j*nu*6+i*6];
    *p++=i;
    *p++=j;
    *p++=h;
    *p++=v.x();
    *p++=v.y();
    *p++=v.z();
}

inline void GeomGrid::setn(int i, int j, double h)
{
    float* p=&data[j*nu*6+i*6+3];
    *p++=i;
    *p++=j;
    *p++=h;
}

inline void GeomGrid::set(int i, int j, double h, const MaterialHandle& matl)
{
    float* p=&data[j*nu*7+i*7];
    matl->diffuse.get_color(p);
    p+=4;
    *p++=i;
    *p++=j;
    *p++=h;
}

inline void GeomGrid::set(int i, int j, double h, const Vector& normal,
		   const MaterialHandle& matl)
{
    float* p=&data[j*nu*10+i*10];
    matl->diffuse.get_color(p);
    p+=4;
    *p++=normal.x();
    *p++=normal.y();
    *p++=normal.z();
    *p++=i;
    *p++=j;
    *p++=h;
}

} // End namespace GeomSpace
} // End namespace SCICore

#endif /* SCI_Geom_Grid_h */
#endif

/*
 *  Grid.h: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Grid_h
#define SCI_Geom_Grid_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/Array2.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::Array2;

class SCICORESHARE GeomGrid : public GeomObj {
    Array2<double> verts;
    Array2<MaterialHandle> matls;
    Array2<Vector> normals;
    int have_matls;
    int have_normals;
    Point corner;
    Vector u, v, w;

    int image;
public:
  GeomGrid(int, int, const Point&, const Vector&, const Vector&, 
	     int image=0);
  GeomGrid(const GeomGrid&);
  virtual ~GeomGrid();

  virtual GeomObj* clone();
  
  void set(int, int, double);
  void set(int, int, double, const MaterialHandle&);
  void set(int, int, double, const Vector&);
  void set(int, int, double, const Vector&, const MaterialHandle&);
  double get(int, int);

  void adjust();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:39  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:04  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:57  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif /* SCI_Geom_Grid_h */
