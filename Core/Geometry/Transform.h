
/*
 *  Transform.h:  ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Transform_h
#define Geometry_Transform_h 1

#include <SCICore/share/share.h>
#include <SCICore/Geometry/Plane.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
namespace SCICore {
namespace Geometry {

class Vector;
class Point;
class Quaternion;

class SCICORESHARE Transform {
    double mat[4][4];
    double imat[4][4];
    int inverse_valid;
    void install_mat(double[4][4]);
    void build_permute(double m[4][4], int, int, int, int pre);
    void build_rotate(double m[4][4], double, const Vector&);
    void build_shear(double mat[4][4], const Vector&, const Plane&);
    void build_scale(double m[4][4], const Vector&);
    void build_translate(double m[4][4], const Vector&);
    void pre_mulmat(double[4][4]);
    void post_mulmat(double[4][4]);
    void invmat(double[4][4]);
    void switch_rows(double m[4][4], int row1, int row2) const;
    void sub_rows(double m[4][4], int row1, int row2, double mul) const;
    void load_identity(double[4][4]);
    void load_zero(double[4][4]);

    friend class Quaternion;

public:
    Transform();
    Transform(const Transform&);
    Transform& operator=(const Transform&);
    ~Transform();
    Transform(const Point&, const Vector&, const Vector&, const Vector&);

    void load_frame(const Point&,const Vector&, const Vector&, const Vector&);
    void change_basis(Transform&);

    void post_trans(Transform&);
    void pre_trans(Transform&);
    
    void print(void);
    void printi(void);

    void pre_permute(int xmap, int ymap, int zmap);
    void post_permute(int xmap, int ymap, int zmap);
    void pre_scale(const Vector&);
    void post_scale(const Vector&);
    void pre_shear(const Vector&, const Plane&);
    void post_shear(const Vector&, const Plane&);
    void pre_rotate(double, const Vector& axis);
    void post_rotate(double, const Vector& axis);
    void pre_translate(const Vector&);
    void post_translate(const Vector&);
    Point unproject(const Point& p);
    Point project(const Point& p);
    Vector project(const Vector& p);
    void get(double*);
    void get_trans(double*);
    void set(double*);
    void load_identity();
    void perspective(const Point& eyep, const Point& lookat,
		     const Vector& up, double fov,
		     double znear, double zfar,
		     int xres, int yres);
    void compute_imat();
    void invert();
};

} // End namespace SCICore
} // End namespace GeomSpace

//
// $Log$
// Revision 1.5  2000/08/04 19:09:25  dmw
// fixed shear
//
// Revision 1.4  2000/07/27 05:21:17  samsonov
// Added friend class Quaternion
//
// Revision 1.3  2000/03/13 05:05:12  dmw
// Added Transform::permute for swapping axes, and fixed compute_imat
//
// Revision 1.2  1999/08/17 06:39:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:57  mcq
// Initial commit
//
// Revision 1.4  1999/07/09 00:27:40  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.3  1999/05/06 19:56:17  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:19  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif

