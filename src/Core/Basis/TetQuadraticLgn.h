/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//  
//    File   : TetQuadraticLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(TetQuadraticLgn_h)
#define TetQuadraticLgn_h

#include <Core/Basis/TetLinearLgn.h>

#include <Core/Basis/share.h>
namespace SCIRun {

//! Class for describing unit geometry of TetQuadraticLgn 
  class SCISHARE TetQuadraticLgnUnitElement : public TetLinearLgnUnitElement {
public:
  static double unit_vertices[10][3]; //!< Parametric coordinates of vertices of unit edge
  
  TetQuadraticLgnUnitElement() {}
  virtual ~TetQuadraticLgnUnitElement() {}
   
  static int number_of_vertices() { return 10; } //! return number of vertices
  static int dofs() { return 10; } //!< return degrees of freedom
};


//! Class for handling of element of type tetrahedron with 
//! quadratic lagrangian interpolation
template <class T>
class TetQuadraticLgn : public BasisAddNodes<T>, 
                        public TetApprox, 
			public TetGaussian3<double>, 
			public TetQuadraticLgnUnitElement  
{
public:
  typedef T value_type;

  TetQuadraticLgn() {}
  virtual ~TetQuadraticLgn() {}

  static int polynomial_order() { return 2; }

  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];
    w[0] = (1 + 2*x*x + 2*y*y - 3*z + 2*z*z + y*(-3 + 4*z) + x*(-3 + 4*y + 4*z));
    w[1] = +x*(-1 + 2*x);
    w[2] = +y*(-1 + 2*y);
    w[3] = +z*(-1 + 2*z);
    w[4] = -4*x*(-1 + x + y + z);
    w[5] = +4*x*y;
    w[6] = -4*y*(-1 + x + y + z);
    w[7] = -4*z*(-1 + x + y + z);
    w[8] = +4*x*z;
    w[9] = +4*y*z;
  }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[10];
    get_weights(coords, w); 

    return (T)(w[0] * cd.node0() +
	       w[1] * cd.node1() +
	       w[2] * cd.node2() +
	       w[3] * cd.node3() +
	       w[4] * this->nodes_[cd.edge0_index()] +
	       w[5] * this->nodes_[cd.edge1_index()] +
	       w[6] * this->nodes_[cd.edge2_index()] +
	       w[7] * this->nodes_[cd.edge3_index()] +
	       w[8] * this->nodes_[cd.edge4_index()] +
	       w[9] * this->nodes_[cd.edge5_index()]);
  }
 
 //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    w[0] = (-3 + 4*x + 4*y + 4*z);
    w[1] = +(-1 + 4*x);
    w[2] = 0;
    w[3] = 0;
    w[4] = -4*(-1 + 2*x + y + z);
    w[5] = 4*y;
    w[6] = -4*y;
    w[7] = -4*z;
    w[8] = 4*z;
    w[9] = 0;
    w[10] = (-3 + 4*x + 4*y + 4*z);
    w[11] = 0;
    w[12] = (-1 + 4*y);
    w[13] = 0;
    w[14] = -4*x;
    w[15] = +4*x;
    w[16] = -4*(-1 + x + 2*y + z);
    w[17] = -4*z;
    w[18] = 0;
    w[19] = +4*z;
    w[20] = (-3 + 4*x + 4*y + 4*z);
    w[21] = 0;
    w[22] = 0;
    w[23] = +(-1 + 4*z);
    w[24] = -4*x;
    w[25] = 0;
    w[26] = -4*y;
    w[27] = -4*(-1 + x + y + 2*z);
    w[28] = +4*x;
    w[29] = +4*y;
 }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];
    
    derivs.resize(3);

    derivs[0]=T((-3 + 4*x + 4*y + 4*z)*cd.node0()
		+(-1 + 4*x)*cd.node1()
		-4*(-1 + 2*x + y + z)*this->nodes_[cd.edge0_index()]
		+4*y*this->nodes_[cd.edge1_index()]
		-4*y*this->nodes_[cd.edge2_index()]
		-4*z*this->nodes_[cd.edge3_index()]
		+4*z*this->nodes_[cd.edge4_index()]);
      
    derivs[1]=T((-3 + 4*x + 4*y + 4*z)*cd.node0()
		+(-1 + 4*y)*cd.node2()
		-4*x*this->nodes_[cd.edge0_index()]
		+4*x*this->nodes_[cd.edge1_index()]
		-4*(-1 + x + 2*y + z)*this->nodes_[cd.edge2_index()]
		-4*z*this->nodes_[cd.edge3_index()]
		+4*z*this->nodes_[cd.edge5_index()]);
      
    derivs[2]=T((-3 + 4*x + 4*y + 4*z)*cd.node0()
		+(-1 + 4*z)*cd.node3()
		-4*x*this->nodes_[cd.edge0_index()]
		-4*y*this->nodes_[cd.edge2_index()]
		-4*(-1 + x + y + 2*z)*this->nodes_[cd.edge3_index()]
		+4*x*this->nodes_[cd.edge4_index()]
		+4*y*this->nodes_[cd.edge5_index()]);
  }
  
  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    TetLocate< TetQuadraticLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }
 
  //! get arc length for edge
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    return get_arc3d_length<CrvGaussian2<double> >(this, edge, cd);
  }
 
  //! get area
  template <class ElemData>
    double get_area(const unsigned face, const ElemData &cd) const  
  {
    return get_area3<TriGaussian3<double> >(this, face, cd);
  }
 
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & cd) const  
  {
    return get_volume3(this, cd);
  }
  
  static  const std::string type_name(int n = -1);

  virtual void io (Piostream& str);
};


template <class T>
const TypeDescription* get_type_description(TetQuadraticLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("TetQuadraticLgn", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
TetQuadraticLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("TetQuadraticLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int TETQUADRATICLGN_VERSION = 1;
template <class T>
void
TetQuadraticLgn<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     TETQUADRATICLGN_VERSION);
  Pio(stream, this->nodes_);
  stream.end_class();
}

} //namespace SCIRun

#endif // TetQuadraticLgn_h
