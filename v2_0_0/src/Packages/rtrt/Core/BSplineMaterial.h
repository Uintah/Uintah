
#ifndef BSPLINEMATERIAL_H
#define BSPLINEMATERIAL_H 1

#include <iostream>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>

namespace rtrt {
using namespace std;
  
class Knot {
private:
  int _num_pts;
  int _degree;
public:
  Knot(const int num_pts, const int degree):
    _num_pts(num_pts), _degree(degree) {}

  float operator[](int n) const {
    ASSERTRANGE(n, 0, _num_pts + _degree + 1);
    if (n > _degree) {
      if (n < _num_pts) {
	// do the middle case
	return (n - _degree);
      } else {
	// do the end case
	return (_num_pts - _degree);
      }
    } else {
      // do the beginning case
      return 0;
    }
  }

  inline int size() const { return (_num_pts + _degree + 1); }
  inline int degree() const { return _degree; }
  inline float domain_min() const { return 0; }
  inline float domain_max() const { return (_num_pts - _degree); }

  int find_next_index(const float t) const;
};

class Knot_01 {
private:
  int _num_pts;
  int _degree;
  float inc;
public:
  Knot_01(const int num_pts, const int degree):
    _num_pts(num_pts), _degree(degree), inc(1/(float)(num_pts - degree)) {}

  float operator[](int n) const {
    ASSERTRANGE(n, 0, _num_pts + _degree + 1);
    if (n > _degree) {
      if (n < _num_pts) {
	// do the middle case
	return (n - _degree) * inc;
      } else {
	// do the end case
	return 1;
      }
    } else {
      // do the beginning case
      return 0;
    }
  }

  inline int size() const { return (_num_pts + _degree + 1); }
  inline int degree() const { return _degree; }
  inline float domain_min() const { return 0; }
  inline float domain_max() const { return 1; }

  int find_next_index(const float t) const;
};

class BSplineMaterial : public Material {
public:
  enum Mode {
    Tile,
    Clamp,
    None
  };
private:
  Color ambient;
  double Kd;
  Color specular;
  double specpow;
  double refl;
  double transp;
  Array2<Color> image;
  Mode umode, vmode;
  Color outcolor;
  
  // BSpline specific stuff
  //Array1<float> knotV_u;
  //Array1<float> knotV_v;
  //int last_J;
  //int last_Mu;

  int find_next_index(const float t, const Knot& knotV) const;

  //  void change_degree(const int new_degree);
  
  //Array2<Color> p_eval; // should be image.dim1() ^ 2
  //Array2<Color> q_eval; // should be image.dim2() ^ 2
  //Array1<Color> Q;      // should be image.dim2()
  void get_Array2(const int dim1, const int dim2,
		  Color**& objs, char * index, char * data);
  unsigned long offset1, offset2, offset3;
  unsigned long data_size;
  
  Color evaluate_2d_spline(const float u, const float v, const int degree,
			   PerProcessorContext* ppc);
  Color evaluate_1d_spline(const float t, const int J, const int degree,
			   Color**& curve, const Knot_01& knotV) const;
  float basis(const int i, const int k, const Knot_01& knotV,const float t) const;
  void read_image(char* texfile);
public:
  BSplineMaterial(char* filename, Mode umode, Mode vmode,
		  const Color& ambient,
		  double Kd, const Color& specular,
		  double specpow, double refl, 
		  double transp=0);
  BSplineMaterial(char* filename, Mode umode, Mode vmode,
		  const Color& ambient,
		  double Kd, const Color& specular,
		  double specpow, double refl=0);
  virtual ~BSplineMaterial();

  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual int get_scratchsize();
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt
#endif
