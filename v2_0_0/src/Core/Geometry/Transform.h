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

#ifndef Geometry_Transform_h
#define Geometry_Transform_h 1

#include <Core/Persistent/Persistent.h>
#include <Core/share/share.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class Vector;
class Point;
class Quaternion;
class Piostream;
class Plane;
class Transform;
class TypeDescription;

void Pio_old(Piostream&, Transform&);
void Pio(Piostream&, Transform*&);

class SCICORESHARE Transform : public Persistent {
  double mat[4][4];
  double imat[4][4];
  int inverse_valid;
  void install_mat(double[4][4]);
  void build_permute(double m[4][4], int, int, int, int pre);
  void build_rotate(double m[4][4], double, const Vector&);
  void build_shear(double mat[4][4], const Vector&, const Plane&);
  void build_scale(double m[4][4], const Vector&);
  void build_translate(double m[4][4], const Vector&);
  void pre_mulmat(const double[4][4]);
  void post_mulmat(const double[4][4]);
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


  //! Persistent I/O.
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  friend void Pio_old(Piostream&, Transform&);
  friend void Pio(Piostream&, Transform*&);

  void load_basis(const Point&,const Vector&, const Vector&, const Vector&);
  void load_frame(const Point&,const Vector&, const Vector&, const Vector&);

  void change_basis(Transform&);
  void post_trans(const Transform&);
  void pre_trans(const Transform&);
    
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
  void unproject(const Point& p, Point& res);
  void unproject_inplace(Point& p);
  Vector unproject(const Vector& p);
  void unproject(const Vector& v, Vector& res);
  void unproject_inplace(Vector& v);
  Point project(const Point& p) const;
  void project(const Point& p, Point& res) const;
  void project_inplace(Point& p) const;
  Vector project(const Vector& p) const;
  void project(const Vector& p, Vector& res) const;
  void project_inplace(Vector& p) const;
  Vector project_normal(const Vector&);
  void project_normal(const Vector&, Vector& res);
  void project_normal_inplace(Vector&);
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
  bool inv_valid()
  {
    return inverse_valid;
  }

  //! support dynamic compilation
  static const std::string& get_h_file_path();

};

const TypeDescription* get_type_description(Transform*);

} // End namespace SCIRun

#endif

