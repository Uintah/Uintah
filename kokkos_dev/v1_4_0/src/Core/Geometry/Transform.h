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

#include <Core/share/share.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

class Vector;
class Point;
class Quaternion;
class Plane;
  
void SCICORESHARE Pio(Piostream&, Transform&);

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
  void rotate(const Vector& from, const Vector& to);
  void pre_translate(const Vector&);
  void post_translate(const Vector&);

  Point unproject(const Point& p);
  Vector unproject(const Vector& p);
  Point project(const Point& p) const;
  Vector project(const Vector& p) const;
  Vector project_normal(const Vector&) const;
  void get(double*) const;
  void get_trans(double*) const;
  void set(double*);
  void load_identity();
  void perspective(const Point& eyep, const Point& lookat,
		   const Vector& up, double fov,
		   double znear, double zfar,
		   int xres, int yres);
  void compute_imat();
  void invert();
  
  //! support dynamic compilation
  static const string& get_h_file_path();

  friend void SCICORESHARE Pio(Piostream&, Transform&);
};

const TypeDescription* get_type_description(Transform*);

} // End namespace SCIRun

#endif

