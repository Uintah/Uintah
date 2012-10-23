/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Geometry_Transform_h
#define Geometry_Transform_h 1

#include <Core/Persistent/Persistent.h>
#include <string>

#include <Core/Geometry/share.h>

namespace SCIRun {

class Vector;
class Point;
class Quaternion;
class Piostream;
class Plane;
class Transform;
class TypeDescription;

SCISHARE void Pio_old(Piostream&, Transform&);
SCISHARE void Pio(Piostream&, Transform*&);

class SCISHARE Transform  : public Persistent
{
  double mat[4][4];
  mutable double imat[4][4];
  mutable int inverse_valid;
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
  SCISHARE friend void Pio_old(Piostream&, Transform&);
  SCISHARE friend void Pio(Piostream&, Transform*&);

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
  // Returns true if the rotation happened, false otherwise.
  bool rotate(const Vector& from, const Vector& to);
  void pre_translate(const Vector&);
  void post_translate(const Vector&);

  Point unproject(const Point& p) const;
  void unproject(const Point& p, Point& res) const;
  void unproject_inplace(Point& p) const;
  Vector unproject(const Vector& p) const;
  void unproject(const Vector& v, Vector& res) const;
  void unproject_inplace(Vector& v) const;
  Point project(const Point& p) const;
  void project(const Point& p, Point& res) const;
  void project_inplace(Point& p) const;
  Vector project(const Vector& p) const;
  void project(const Vector& p, Vector& res) const;
  void project_inplace(Vector& p) const;
  Vector project_normal(const Vector&) const;
  void project_normal(const Vector&, Vector& res) const;
  void project_normal_inplace(Vector&) const;
  void get(double*) const;
  void get_trans(double*) const;
  void set(double*);
  void set_trans(double*);
  void load_identity();
  void perspective(const Point& eyep, const Point& lookat,
		   const Vector& up, double fov,
		   double znear, double zfar,
		   int xres, int yres);
  void compute_imat() const;
  void invert();
  bool inv_valid()
  {
    return inverse_valid;
  }

  //! support dynamic compilation
  static const std::string& get_h_file_path();

};


SCISHARE Point operator*(Transform &t, const Point &d);
SCISHARE Vector operator*(Transform &t, const Vector &d);



SCISHARE const TypeDescription* get_type_description(Transform*);

} // End namespace SCIRun

#endif
