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

#include <Util/Point.h>

namespace SemotusVisum {

class Transform {

  // member variables
  double mat[4][4];
  double imat[4][4];
  int inverse_valid;

  // member functions
  void install_mat(double[4][4]);
  void build_permute(double m[4][4], int, int, int, int pre);
  void build_rotate(double m[4][4], double, const Vector3d&);
  void build_scale(double m[4][4], const Vector3d&);
  void build_translate(double m[4][4], const Vector3d&);
  void pre_mulmat(double[4][4]);
  void post_mulmat(double[4][4]);
  void invmat(double[4][4]);
  void switch_rows(double m[4][4], int row1, int row2) const;
  void sub_rows(double m[4][4], int row1, int row2, double mul) const;
  void load_identity(double[4][4]);
  void load_zero(double[4][4]);

public:
  Transform();
  Transform(const Transform&);
  Transform& operator=(const Transform&);
  ~Transform();
  Transform(const Point3d&, const Vector3d&, const Vector3d&, const Vector3d&);

  void load_frame(const Point3d&,const Vector3d&, const Vector3d&, const Vector3d&);
  void change_basis(Transform&);

  void post_trans(Transform&);
  void pre_trans(Transform&);
    
  void print(void);
  void printi(void);

  void pre_permute(int xmap, int ymap, int zmap);
  void post_permute(int xmap, int ymap, int zmap);
  void pre_scale(const Vector3d&);
  void post_scale(const Vector3d&);
  void pre_rotate(double, const Vector3d& axis);
  void post_rotate(double, const Vector3d& axis);
  void pre_translate(const Vector3d&);
  void post_translate(const Vector3d&);

  Point3d unproject(const Point3d& p);
  Vector3d unproject(const Vector3d& p);
  Point3d project(const Point3d& p) const;
  Vector3d project(const Vector3d& p) const;
  void get(double*);
  void get_trans(double*);
  void set(double*);
  void load_identity();
  void perspective(const Point3d& eyep, const Point3d& lookat,
		   const Vector3d& up, double fov,
		   double znear, double zfar,
		   int xres, int yres);
  void compute_imat();
  void invert();
  
};

} // End namespace SemotusVisum

#endif










