/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
//  
//    File   : TetCubicHmt.h
//    Author : Frank B. Sachse
//    Date   : Nov 30 2004

#if !defined(TetCubicHmt_h)
#define TetCubicHmt_h

#include <Core/Basis/TetLinearLgn.h>

namespace SCIRun {

//! Class for describing unit geometry of TetCubicHmt
class TetCubicHmtUnitElement : public TetLinearLgnUnitElement {
public:
  TetCubicHmtUnitElement() {}
  virtual ~TetCubicHmtUnitElement() {}

  static int dofs() { return 16; } //!< return degrees of freedom
};

//! Class for handling of element of type tetrahedron with 
//! cubic hermitian interpolation
template <class T>
class TetCubicHmt : public BasisAddDerivatives<T>, 
                    public TetApprox, 
		    public TetGaussian3<double>, 
		    public TetCubicHmtUnitElement
{
public:
  typedef T value_type;

  TetCubicHmt() {}
  virtual ~TetCubicHmt() {}

  static int polynomial_order() { return 3; }

  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  { 
    const double x=coords[0], y=coords[1], z=coords[2];  
    w[0]  = (-3*x*x + 2*x*x*x - 3*y*y + 2*y*y*y + (z-1)*(z-1)*(1 + 2*z));
    w[1]  = +x*(1 + x*x + x*(-2 + y) - y - z*z);
    w[2]  = +y*(-x*x + (-1 + y)*(-1 + y + z));
    w[3]  = +(-y*y + x*(-1 + z) + (z-1)*(z-1))*z;
    w[4]  = +(3 - 2*x)*x*x;
    w[5]  = +(-1 + x)*x*x;
    w[6]  = +x*x*y;
    w[7]  = -(x*(-1 + z)*z);
    w[8]  = +(3 - 2*y)*y*y;
    w[9]  = -((-1 + x)*x*y);
    w[10] = +(-1 + y)*y*y;
    w[11] = +y*y*z;
    w[12] = +(3 - 2*z)*z*z;
    w[13] = +x*z*z;
    w[14] = -((-1 + y)*y*z);
    w[15] = +(-1 + z)*z*z;
  }

  //! get value at parametric coordinate  
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[16];
    get_weights(coords, w); 

    return (T)(w[0]  * cd.node0()                   +
	       w[1]  * this->derivs_[cd.node0_index()][0] +
	       w[2]  * this->derivs_[cd.node0_index()][1] +
	       w[3]  * this->derivs_[cd.node0_index()][2] +
	       w[4]  * cd.node1()		    +
	       w[5]  * this->derivs_[cd.node1_index()][0] +
	       w[6]  * this->derivs_[cd.node1_index()][1] +
	       w[7]  * this->derivs_[cd.node1_index()][2] +
	       w[8]  * cd.node2()		    +
	       w[9]  * this->derivs_[cd.node2_index()][0] +
	       w[10] * this->derivs_[cd.node2_index()][1] +
	       w[11] * this->derivs_[cd.node2_index()][2] +
	       w[12] * cd.node3()		    +
	       w[13] * this->derivs_[cd.node3_index()][0] +
	       w[14] * this->derivs_[cd.node3_index()][1] +
	       w[15] * this->derivs_[cd.node3_index()][2]);
  }
  
 //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    w[0] = 6*(-1 + x)*x;
    w[1] = +(1 + 3*x*x + 2*x*(-2 + y) - y - z*z);
    w[2] = -2*x*y;
    w[3] = +(-1 + z)*z;
    w[4] = -6*(-1 + x)*x;
    w[5] = +x*(-2 + 3*x);
    w[6] = +2*x*y;
    w[7] = -((-1 + z)*z);
    w[8] = (y - 2*x*y);
    w[9] = 0;
    w[10] = 0;
    w[11] = 0;
    w[12] = z*z;
    w[13] = 0;
    w[14] = 0;
    w[15] = 0;
    w[16] = 6*(-1 + y)*y;
    w[17] = (-1 + x)*x;
    w[18] = (1 - x*x + 3*y*y + 2*y*(-2 + z) - z);
    w[19] = -2*y*z;
    w[20] = 0;
    w[21] = 0;
    w[22] = x*x;
    w[23] = 0;
    w[24] = -6*(-1 + y)*y;
    w[25] = -((-1 + x)*x);
    w[26] = +y*(-2 + 3*y);
    w[27] = +2*y*z;
    w[28] = 0;
    w[29] = 0;
    w[30] = (z - 2*y*z);
    w[31] = 0;
    w[32] = 6*(-1 + z)*z;
    w[33] = -2*x*z;
    w[34] = (-1 + y)*y;
    w[35] = (1 - x - y*y - 4*z + 2*x*z + 3*z*z);
    w[36] = 0;
    w[37] = 0;
    w[38] = 0;
    w[39] = (x - 2*x*z);
    w[40] = 0;
    w[41] = 0;
    w[42] = 0;
    w[43] = y*y;
    w[44] = -6*(-1 + z)*z;
    w[45] = 2*x*z;
    w[46] = -((-1 + y)*y);
    w[47] = z*(-2 + 3*z);
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2]; 
 
    derivs.resize(3);

 
    derivs[0]=
      T(6*(-1 + x)*x*cd.node0()
	+(1 + 3*x*x + 2*x*(-2 + y) - y - z*z)*this->derivs_[cd.node0_index()][0]
	-2*x*y*this->derivs_[cd.node0_index()][1]
	+(-1 + z)*z*this->derivs_[cd.node0_index()][2]
	-6*(-1 + x)*x*cd.node1()
	+x*(-2 + 3*x)*this->derivs_[cd.node1_index()][0]
	+2*x*y*this->derivs_[cd.node1_index()][1]
	-((-1 + z)*z)*this->derivs_[cd.node1_index()][2]
	+(y - 2*x*y)*this->derivs_[cd.node2_index()][0]
	+z*z*this->derivs_[cd.node3_index()][0]);

    derivs[1]=
      T(6*(-1 + y)*y*cd.node0()
	+(-1 + x)*x*this->derivs_[cd.node0_index()][0]
	+(1 - x*x + 3*y*y + 2*y*(-2 + z) - z)*this->derivs_[cd.node0_index()][1]
	-2*y*z*this->derivs_[cd.node0_index()][2]
	+x*x*this->derivs_[cd.node1_index()][1]
	-6*(-1 + y)*y*cd.node2()
	-((-1 + x)*x)*this->derivs_[cd.node2_index()][0]
	+y*(-2 + 3*y)*this->derivs_[cd.node2_index()][1]
	+2*y*z*this->derivs_[cd.node2_index()][2]
	+(z - 2*y*z)*this->derivs_[cd.node3_index()][1]);

    derivs[2]=
      T(6*(-1 + z)*z*cd.node0()
	-2*x*z*this->derivs_[cd.node0_index()][0]
	+(-1 + y)*y*this->derivs_[cd.node0_index()][1]
	+(1 - x - y*y - 4*z + 2*x*z + 3*z*z)*this->derivs_[cd.node0_index()][2]
	+(x - 2*x*z)*this->derivs_[cd.node1_index()][2]
	+y*y*this->derivs_[cd.node2_index()][2]
	-6*(-1 + z)*z*cd.node3()
	+2*x*z*this->derivs_[cd.node3_index()][0]
	-((-1 + y)*y)*this->derivs_[cd.node3_index()][1]
	+z*(-2 + 3*z)*this->derivs_[cd.node3_index()][2]);
  }
  
  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    TetLocate< TetCubicHmt<T> > CL;
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
const TypeDescription* get_type_description(TetCubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("TetCubicHmt", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
TetCubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("TetCubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int TETCUBICHMT_VERSION = 1;
template <class T>
void
TetCubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     TETCUBICHMT_VERSION);
  Pio(stream, this->derivs_);
  stream.end_class();
}

} //namespace SCIRun


#endif // TetCubicHmt_h
