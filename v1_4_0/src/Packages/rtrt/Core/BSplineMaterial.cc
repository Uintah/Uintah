
#include <Packages/rtrt/Core/BSplineMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <Packages/rtrt/Core/Context.h>
#include <Core/Malloc/Allocator.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;


bool do_bspline = true;

BSplineMaterial::BSplineMaterial(char* texfile, BSplineMaterial::Mode umode,
				 BSplineMaterial::Mode vmode,
				 const Color& ambient, double Kd,
				 const Color& specular, double specpow,
				 double refl)
  : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
    specpow(specpow), refl(refl),  transp(0)
{
  read_image(texfile);
  outcolor=Color(1,0,0);
}

BSplineMaterial::BSplineMaterial(char* texfile, BSplineMaterial::Mode umode,
				 BSplineMaterial::Mode vmode,
				 const Color& ambient, double Kd,
				 const Color& specular, double specpow,
				 double refl,  double transp)
  : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
    specpow(specpow), refl(refl),  transp(transp)
{
  read_image(texfile);
  outcolor=Color(1,0,0);
}

BSplineMaterial::~BSplineMaterial()
{
}

int BSplineMaterial::get_scratchsize() {
  offset1 = image.dim1() * sizeof(Color*);
  offset1 += (8-offset1%8);
  offset2 = offset1 + image.dim1() * image.dim1() * sizeof(Color);
  offset2 += (8-offset2%8);
  offset3 = offset2 + image.dim2() * sizeof(Color*);
  offset3 += (8-offset3%8);
  data_size = offset3 + image.dim2() * image.dim2() * sizeof(Color);
  data_size += (8-data_size%8);
  return data_size;
}

void BSplineMaterial::shade(Color& result, const Ray& ray,
			    const HitInfo& hit, int /* depth */, 
			    double /* atten */, const Color& /* accumcolor */,
			    Context* cx)
{
  UVMapping* map=hit.hit_obj->get_uvmapping();
  UV uv;
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);
  map->uv(uv, hitpos, hit);
  Color diffuse;
  double u=uv.u();
  double v=uv.v();
  switch(umode){
  case None:
    if(u<0 || u>1){
      diffuse=outcolor;
      goto skip;
    }
    break;
  case Tile:
    {
      int iu=(int)u;
      u-=iu;
      if (u < 0) u += 1;
    }
    break;
  case Clamp:
    if(u>1)
      u=1;
    else if(u<0)
      u=0;
  };
  switch(vmode){
  case None:
    if(v<0 || v>1){
      diffuse=outcolor;
      goto skip;
    }
    break;
  case Tile:
    {
      int iv=(int)v;
      v-=iv;
      if (v < 0) v += 1;
    }
    break;
  case Clamp:
    if(v>1)
      v=1;
    else if(v<0)
      v=0;
  };
  {
    if (do_bspline) {
      diffuse = evaluate_2d_spline(u,v,3, cx->ppc);
      //cerr << "Made it past evaluate_2d_spline\n";
      //flush(cerr);
    } else {
      u*=image.dim1();
      v*=image.dim2();
      int iu=(int)u;
      int iv=(int)v;
      diffuse=image(iu, iv);
    }
  }
 skip:
#if 0
  ambient=diffuse*50;
  
  phongshade(result, ambient, diffuse, specular, specpow, refl,
	     ray, hit, depth,  atten,
	     accumcolor, cx);
#else
  result=diffuse;
#endif
}

#if 0
void BSplineMaterial::change_degree(const int new_degree) {
  // check to make sure the new degree is valid
  if (new_degree < 0) {
    cerr << "BSplineMaterial::change_degree:new_degree is negative!\n";
    return;
  }
  // need to check the upper bounds as well
  if (new_degree < image.dim1() -1 && new_degree < image.dim2() -1)
    degree = new_degree;
  else
    cerr << "BSplineMaterial::change_degree:new_degree is too large!\n";
}
#endif

void BSplineMaterial::get_Array2(const int dim1, const int dim2,
				 Color**& objs, char* index, char * data) {
#if 0
  objs= (Color **)memory;//new T*[dm1];
  memory += dim1 * sizeof(Color*);
  Color* p= (Color*)memory;//new T[dm1*dm2];
#else
  objs = (Color **) index;
  Color* p = (Color *) data;
  //cerr << "objs = " << (unsigned long) objs << endl;
  //cerr << "p = " << (unsigned long) p << endl;
  //flush(cerr);
#endif
  for(int i = 0; i < dim1; i++){
    //cerr << "i=" << i << " ";
    //flush(cerr);
    objs[i] = p;
    p += dim2;
  }
}

int BSplineMaterial::find_next_index(const float t, const Knot& knotV) const {
  //if (knotV[last_J] <= t && t < knotV[last_J+1] )
  //  return last_J;
  int NK = knotV.size() - knotV.degree() - 1;
  for (int i = knotV.degree(); i < NK; i++) {
    if (knotV[i] <= t && t < knotV[i+1]) {
      //    last_J = i;
      return i;
    }
  }
  // should never reach this point
  return knotV.degree();
}

Color BSplineMaterial::evaluate_1d_spline(const float t, const int J,
					  const int degree,
					  Color**& curve,
					  const Knot_01& knotV) const
{
#if 1
  Color result(0,0,0);
  for (int m = J - degree; m <= J; m++) {
    float b =  basis(m, degree, knotV, t);
    //    if (b < 0 || b > 1) {
    if (false) {
      cerr << "\t\t\tbasis is : " << b << endl;
      flush (cerr);
    }
    //    result += curve[0][m] * basis(m, degree, knotV, t);
    result += curve[0][m] * b;
  }
  return result;
#else
  //cerr << "BSplineMaterial::evaluate_1d_spline:start\n";
  //cerr << "curve_ptr: " << curve.get_dataptr() << endl;
  //flush(cerr);
  for (int p = 1; p <= degree; p++) {
    //cerr << "p = " << p << endl;
    //flush(cerr);
    for (int i = J - degree + p; i <= J; i++) {
      //cerr << "i = " << i << endl;
      //flush(cerr);
      float t2 = knotV[i + degree - p + 1];
      //cerr << "t2 = " << t2 << endl;
      //flush(cerr);
      float ti = knotV[i];
      //cerr << "ti = " << ti << endl;
      //flush(cerr);
      //if (t2 == ti) {
      //cerr << "divide by zero: t2 = " << t2 << ", ti = " << ti << endl;
      //flush(cerr);
      //}
      //Color p1 = curve(p-1, i)   * ((t-ti)/(t2-ti));
      //cerr << "p1 = " << p1 << endl;
      //flush(cerr);
      //Color p2 = curve(p-1, i-1) * ((t2-t)/(t2-ti));
      //cerr << "p2 = " << p2 << endl;
      //flush(cerr);
      //curve(p, i) = p1 + p2;
      curve[p][i] = curve[p-1][i]   * ((t-ti)/(t2-ti)) +
	            curve[p-1][i-1] * ((t2-t)/(t2-ti));
    }
  }
  //Color result = curve[degree][J];
  //cerr << "BSplineMaterial::evaluate_1d_spline:end\n";
  //flush(cerr);
  return curve[degree][J];
  //return result;
#endif
}

Color BSplineMaterial::evaluate_2d_spline(const float u, const float v,
					  const int degree,
					  PerProcessorContext* ppc) {
  //cerr << "BSplineMaterial::evaluate_2d_spline:start\n";
  //flush(cerr);
  Knot_01 knotV_u(image.dim2(), degree);
  float a = knotV_u.domain_min();//[degree];
  float b = knotV_u.domain_max();//[knotV_v.size() - 1 - degree];
  float U = v * ( b - a ) + a;
  int J = knotV_u.find_next_index(U);
  //int J = find_next_index(U, knotV_u);
  
  Knot_01 knotV_v(image.dim1(), degree);
  a = knotV_v.domain_min();//[degree];
  b = knotV_v.domain_max();//[knotV_u.size() - 1 - degree];
  float V = u * ( b - a ) + a;
  int Mu = knotV_v.find_next_index(V);
  //  int Mu = find_next_index(V, knotV_v);

  Color** p_eval;
  char* start = ppc->getscratch(data_size);
  //cerr << "image( " << image.dim1() << " , " << image.dim2() << " )\n";
  //cerr << "start = " << (unsigned long) start << endl;
  //cerr << "data_size = " << data_size << endl;
  //cerr << "offset1 = " << offset1 << endl;
  //cerr << "offset2 = " << offset2 << endl;
  //cerr << "offset3 = " << offset3 << endl;
  //flush(cerr);
  get_Array2(degree + 1, image.dim1(), p_eval,
	     start, start + offset1);
  Color** q_eval;
  get_Array2(degree + 1, image.dim2(), q_eval,
	     start + offset2, start + offset3);

  Color **row = &(p_eval[0]);
  //cerr << "row = " << *row << ", *row = " << *row << endl;
  //cerr << "original = " << original << endl;
  //flush(cerr);
  for (int j = Mu - degree; j <= Mu; j++) {
    // now lets asign the first pointer in the array to values that match
    // the row of data
    *row = *(image.get_ptr_to_row(j));
    //cerr << "*row = " << *row << endl;
    //cerr << "image.get_ptr_to_row( " << j << " ) = " << image.get_ptr_to_row(j) << endl;
    //flush(cerr);
    q_eval[0][j] = evaluate_1d_spline(U,J,knotV_u.degree(),p_eval,knotV_u);
  }
  //Color result = evaluate_1d_spline(V,Mu,knotV_v.degree(),q_eval,knotV_v);
  //cerr << "Color = " << result << endl;
  //cerr << "BSplineMaterial::evaluate_2d_spline:end\n";
  //flush(cerr);
  return evaluate_1d_spline(V,Mu,knotV_v.degree(),q_eval,knotV_v);
  //return result;
}

float BSplineMaterial::basis(const int i, const int k,
			     const Knot_01& knotV,const float t) const {
#if 0
  if (k > 1) {
    if (knotV[i] < knotV[i+k]) {
      return ((t - knotV[i])/(knotV[i+k-1] - knotV[i]) *
	      basis(i, k-1, knotV, t) +
	      (knotV[i+k] - t)/(knotV[i+k] - knotV[i+1]) *
	      basis(i+1, k-1, knotV, t));
    } else {
      return 0;
    }
  } else {
    if (knotV[i] <= t && t < knotV[i+1])
      return 1;
    else
      return 0;
  }
#else
  if (k > 0) {
    float b1 = 0;
    float b2 = 0;
    if (knotV[i] < knotV[i+k]) {
      //    if (knotV[i] < knotV[i+k] && knotV[i+1] < knotV[i+k+1]) {
      //cerr << "knotV[i=" << i << "] = " << knotV[i] << ", knotV[i+k=" << k << "+1] = " << knotV[i+k+1] << endl; flush(cerr);
      float b1a = (t - knotV[i]);
      float b1b = (knotV[i+k] - knotV[i]);
      float b1c = basis(i, k-1, knotV, t);
      b1 = b1a/b1b * b1c;
      //cerr << "b1a = " << b1a << ", b1b = " << b1b << ", b1c = " << b1c << endl; flush (cerr);
      //      float b1 = (t - knotV[i])/(knotV[i+k] - knotV[i]) * basis(i, k-1, knotV, t);
    }
    if (knotV[i+1] < knotV[i+k+1]) {
      float b2a = (knotV[i+k+1] - t);
      float b2b = (knotV[i+k+1] - knotV[i+1]);
      float b2c = basis(i+1, k-1, knotV, t);
      //cerr << "b2a = " << b2a << ", b2b = " << b2b << ", b2c = " << b2c << endl; flush (cerr);
      b2 = b2a/b2b * b2c;
      //      float b2 = (knotV[i+k+1] - t)/(knotV[i+k+1] - knotV[i+1]) * basis(i+1, k-1, knotV, t);
      //cerr << "b1 = " << b1 << ", b2 = " << b2 << endl; flush(cerr);
    }
    return b2 + b1;
    //      return ((t - knotV[i])/(knotV[i+k] - knotV[i]) *
    //	      basis(i, k-1, knotV, t) +
    //	      (knotV[i+k+1] - t)/(knotV[i+k+1] - knotV[i+1]) *
    //	      basis(i+1, k-1, knotV, t));
    //    } else {
    //      return 0;
    //    }
  } else {
    if (knotV[i] <= t && t < knotV[i+1])
      return 1;
    else
      return 0;
  }
#endif
}

void BSplineMaterial::read_image(char* filename)
{
    char buf[200];
    sprintf(buf, "%s.hdr", filename);
    ifstream in(buf);
    if(!in){
	cerr << "Error opening header: " << buf << '\n';
	exit(1);
    }
    int nu, nv;
    in >> nu >> nv;
    if(!in){
	cerr << "Error reading header: " << buf << '\n';
	exit(1);
    }
    ifstream indata(filename);
    image.resize(nu, nv);
    for(int i=0;i<nu;i++){
	for(int j=0;j<nv;j++){
	    unsigned char color[3];
	    indata.read((char*)color, 3);
	    double r=color[0]/255.;
	    double g=color[1]/255.;
	    double b=color[2]/255.;
	    image(i,j)=Color(r,g,b);
	}
    }
    if(!indata){
	cerr << "Error reading image!\n";
	exit(1);
    }
    AuditAllocator(DefaultAllocator());
}

int Knot::find_next_index(const float t) const {
  int t_trunk = (int) t + _degree;
  if ((*this)[t_trunk] <= t && t < (*this)[t_trunk+1]) {
    return t_trunk;
  } else {
    for (int i = _degree; i < _num_pts; i++) {
      if ((*this)[i] <= t && t < (*this)[i+1]) {
	//    last_J = i;
	return i;
      }
    }
  }
  // should never reach this point
  return _degree;
}

int Knot_01::find_next_index(const float t) const {
  int t_trunk = (int) (t / inc) + _degree;
  if ((*this)[t_trunk] <= t && t < (*this)[t_trunk+1]) {
    return t_trunk;
  } else {
    for (int i = _degree; i < _num_pts; i++) {
      if ((*this)[i] <= t && t < (*this)[i+1]) {
	//    last_J = i;
	return i;
      }
    }
  }
  // should never reach this point
  return _degree;
}

