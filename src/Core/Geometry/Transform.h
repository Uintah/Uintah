
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

#include <share/share.h>

namespace SCICore {
namespace Geometry {

class Vector;
class Point;

class SHARE Transform {
    double mat[4][4];
    double imat[4][4];
    int inverse_valid;
    void install_mat(double[4][4]);
    void compute_imat();
    void build_rotate(double m[4][4], double, const Vector&);
    void build_scale(double m[4][4], const Vector&);
    void build_translate(double m[4][4], const Vector&);
    void pre_mulmat(double[4][4]);
    void post_mulmat(double[4][4]);
    void invmat(double[4][4]);
    void switch_rows(double m[4][4], int row1, int row2) const;
    void sub_rows(double m[4][4], int row1, int row2, double mul) const;
    void load_identity(double[4][4]);
public:
    Transform();
    Transform(const Transform&);
    Transform& operator=(const Transform&);
    ~Transform();
    Transform(const Point&, const Vector&, const Vector&, const Vector&);

    void load_frame(const Point&,const Vector&, const Vector&, const Vector&);
    void change_basis(Transform&);

    void post_trans(Transform&);
    void print(void);
    void printi(void);

    void pre_scale(const Vector&);
    void post_scale(const Vector&);
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
    void invert();
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
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

