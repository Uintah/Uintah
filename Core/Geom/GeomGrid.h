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

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array2.h>

namespace SCIRun {

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

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
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
} // End namespace Core

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

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/Array2.h>

namespace SCIRun {


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

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Grid_h */
