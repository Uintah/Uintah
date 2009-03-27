//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : PrismCubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 1 2004

#if !defined(PrismCubicHmt_h)
#define PrismCubicHmt_h

#include <Core/Basis/PrismLinearLgn.h>


namespace SCIRun {

//! Class for describing unit geometry of PrismCubicHmt
class PrismCubicHmtUnitElement : public PrismLinearLgnUnitElement {
public:
  PrismCubicHmtUnitElement() {}
  virtual ~PrismCubicHmtUnitElement() {}

   static int dofs() { return 24; } //!< return degrees of freedom
};


//! Class for handling of element of type prism with 
//! cubic hermitian interpolation
template <class T>
class PrismCubicHmt : public BasisAddDerivatives<T>, 
                      public PrismApprox, 
		      public PrismGaussian2<double>, 
		      public PrismCubicHmtUnitElement 
{
public:
  typedef T value_type;

  PrismCubicHmt() {}
  virtual ~PrismCubicHmt() {}

  static int polynomial_order() { return 3; }

  //! get weight factors at parametric coordinate 
  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);
      
    w[0]  = -((-1 + x + y)*(-1 + z)*(-1 + 2*x2 - y + 2*y2 - x*(1 + 2*y) - z + 2*z2));
    w[1]  = -(x*(1 - 2*x + x2 - y2)*(-1 + z));
    w[2]  = +(x2 - y12)*y*(-1 + z);
    w[3]  = -((-1 + x + y)*z12*z);
    w[4]  = +x*(-1 + z)*(-3*x + 2*x2 + z*(-1 + 2*z));
    w[5]  = +x2*(-1 + x + z - x*z);
    w[6]  = -(x2*y*(-1 + z));
    w[7]  = +x*z12*z;
    w[8]  = +y*(-1 + z)*(-3*y + 2*y2 + z*(-1 + 2*z));
    w[9]  = -(x*y2*(-1 + z));
    w[10] = +y2*(-1 + y + z - y*z);
    w[11] = +y*z12*z;
    w[12] = +(-1 + x + y)*z*(2*x2 - y + 2*y2 - x*(1 + 2*y) + z*(-3 + 2*z));
    w[13] = +x*(1 - 2*x + x2 - y2)*z;
    w[14] = +(-x2 + y12)*y*z;
    w[15] = -((-1 + x + y)*(-1 + z)*z2);
    w[16] = +x*z*(-1 + 3*x - 2*x2 + 3*z - 2*z2);
    w[17] = +(-1 + x)*x2*z;
    w[18] = +x2*y*z;
    w[19] = x*(-1 + z)*z2;
    w[20] = +y*z*(-1 + 3*y - 2*y2 + 3*z - 2*z2);
    w[21] = +x*y2*z;
    w[22] = +(-1 + y)*y2*z;
    w[23] = +y*(-1 + z)*z2;
  }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[24];
    get_weights(coords, w); 
    
    return (T)(w[0]  * cd.node0() +
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
	       w[15] * this->derivs_[cd.node3_index()][2] +
	       w[16] * cd.node4()		    +
	       w[17] * this->derivs_[cd.node4_index()][0] +
	       w[18] * this->derivs_[cd.node4_index()][1] +
	       w[19] * this->derivs_[cd.node4_index()][2] +
	       w[20] * cd.node5()		    +
	       w[21] * this->derivs_[cd.node5_index()][0] +
	       w[22] * this->derivs_[cd.node5_index()][1] +
	       w[23] * this->derivs_[cd.node5_index()][2]);
  }
  
 //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);
    w[0]=-((-1 + z)*(-6*x + 6*x2 + z*(-1 + 2*z)));
    w[1]=-((1 - 4*x + 3*x2 - y2)*(-1 + z));
    w[2]=+2*x*y*(-1 + z);
    w[3]=-(z12*z);
    w[4]=+(-1 + z)*(-6*x + 6*x2 + z*(-1 + 2*z));
    w[5]=-(x*(-2 + 3*x)*(-1 + z));
    w[6]=-2*x*y*(-1 + z);
    w[7]=+z12*z;
    w[8]=0;
    w[9]=-(y2*(-1 + z));
    w[10]=0;
    w[11]=0;
    w[12]=z*(1- 6*x + 6*x2 - 3*z + 2*z2);
    w[13]=+(1 - 4*x + 3*x2 - y2)*z;
    w[14]=-2*x*y*z;
    w[15]=-((-1 + z)*z2);
    w[16]=+z*(-1 + 6*x - 6*x2 + 3*z - 2*z2);
    w[17]=+x*(-2 + 3*x)*z;
    w[18]=+2*x*y*z;
    w[19]=+(-1 + z)*z2;
    w[20]=0;
    w[21]=+y2*z;
    w[22]=0;
    w[23]=0;
    w[24]=-((-1 + z)*(-6*y + 6*y2 + z*(-1 + 2*z)));
    w[25]=+2*x*y*(-1 + z);
    w[26]=+(-1 + x2 + 4*y - 3*y2)*(-1 + z);
    w[27]=-(z12*z);
    w[28]=0;
    w[29]=0;
    w[30]=-(x2*(-1 + z));
    w[31]=0;
    w[32]=+(-1 + z)*(-6*y + 6*y2 + z*(-1 + 2*z));
    w[33]=-2*x*y*(-1 + z);
    w[34]=-(y*(-2 + 3*y)*(-1 + z));
    w[35]=+z12*z;
    w[36]=+z*(1 - 6*y + 6*y2 - 3*z + 2*z2);
    w[37]=-2*x*y*z;
    w[38]=-((-1 + x2 + 4*y - 3*y2)*z);
    w[39]=-((-1 + z)*z2);
    w[40]=0;
    w[41]=0;
    w[42]=+x2*z;
    w[43]=0;
    w[44]=z*(-1 + 6*y - 6*y2 + 3*z - 2*z2);
    w[45]=2*x*y*z;
    w[46]=y*(-2 + 3*y)*z;
    w[47]=(-1 + z)*z2;
    w[48]=-((-1 + x + y)*(2*x2 - y + 2*y2 - x*(1 + 2*y) + 6*(-1 + z)*z));
    w[49]=x*(-1 + 2*x - x2 + y2);
    w[50]=+(x2 - y12)*y;
    w[51]=-((-1 + x + y)*(1 - 4*z + 3*z2));
    w[52]=+x*(1 - 3*x + 2*x2 - 6*z + 6*z2);
    w[53]=-((-1 + x)*x2);
    w[54]=-(x2*y);
    w[55]=+x*(1 - 4*z + 3*z2);
    w[56]=+y*(1 - 3*y + 2*y2 - 6*z + 6*z2);
    w[57]=-(x*y2);
    w[58]=-((-1 + y)*y2);
    w[59]=+y*(1 - 4*z + 3*z2);
    w[60]=+(-1 + x + y)*(2*x2 - y + 2*y2 - x*(1 + 2*y) + 6*(-1 + z)*z);
    w[61]=+x*(1 - 2*x + x2 - y2);
    w[62]=+(-x2 + y12)*y;
    w[63]=-((-1 + x + y)*z*(-2 + 3*z));
    w[64]=+x*(-1 + 3*x - 2*x2 + 6*z - 6*z2);
    w[65]=+(-1 + x)*x2;
    w[66]=+x2*y;
    w[67]=+x*z*(-2 + 3*z);
    w[68]=+y*(-1 + 3*y - 2*y2 + 6*z - 6*z2);
    w[69]=+x*y2;
    w[70]=+(-1 + y)*y2;
    w[71]=+y*z*(-2 + 3*z);
  }


  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];  
    const double x2=x*x;
    const double y2=y*y;
    const double z2=z*z;
    const double y12=(y-1)*(y-1);
    const double z12=(z-1)*(z-1);

    derivs.resize(3);

    derivs[0]=
      T(-((-1 + z)*(-6*x + 6*x2 + z*(-1 + 2*z)))*cd.node0()
	-((1 - 4*x + 3*x2 - y2)*(-1 + z))*this->derivs_[cd.node0_index()][0]
	+2*x*y*(-1 + z)*this->derivs_[cd.node0_index()][1]
	-(z12*z)*this->derivs_[cd.node0_index()][2]
	+(-1 + z)*(-6*x + 6*x2 + z*(-1 + 2*z))*cd.node1()
	-(x*(-2 + 3*x)*(-1 + z))*this->derivs_[cd.node1_index()][0]
	-2*x*y*(-1 + z)*this->derivs_[cd.node1_index()][1]
	+z12*z*this->derivs_[cd.node1_index()][2]
	-(y2*(-1 + z))*this->derivs_[cd.node2_index()][0]
	+z*(1- 6*x + 6*x2 - 3*z + 2*z2)*cd.node3()
	+(1 - 4*x + 3*x2 - y2)*z*this->derivs_[cd.node3_index()][0]
	-2*x*y*z*this->derivs_[cd.node3_index()][1]
	-((-1 + z)*z2)*this->derivs_[cd.node3_index()][2]
	+z*(-1 + 6*x - 6*x2 + 3*z - 2*z2)*cd.node4()
	+x*(-2 + 3*x)*z*this->derivs_[cd.node4_index()][0]
	+2*x*y*z*this->derivs_[cd.node4_index()][1]
	+(-1 + z)*z2*this->derivs_[cd.node4_index()][2]
	+y2*z*this->derivs_[cd.node5_index()][0]);

    derivs[1]=
      T(-((-1 + z)*(-6*y + 6*y2 + z*(-1 + 2*z)))*cd.node0()
	+2*x*y*(-1 + z)*this->derivs_[cd.node0_index()][0]
	+(-1 + x2 + 4*y - 3*y2)*(-1 + z)*this->derivs_[cd.node0_index()][1]
	-(z12*z)*this->derivs_[cd.node0_index()][2]
	-(x2*(-1 + z))*this->derivs_[cd.node1_index()][1]
	+(-1 + z)*(-6*y + 6*y2 + z*(-1 + 2*z))*cd.node2()
	-2*x*y*(-1 + z)*this->derivs_[cd.node2_index()][0]
	-(y*(-2 + 3*y)*(-1 + z))*this->derivs_[cd.node2_index()][1]
	+z12*z*this->derivs_[cd.node2_index()][2]
	+z*(1 - 6*y + 6*y2 - 3*z + 2*z2)*cd.node3()
	-2*x*y*z*this->derivs_[cd.node3_index()][0]
	-((-1 + x2 + 4*y - 3*y2)*z)*this->derivs_[cd.node3_index()][1]
	-((-1 + z)*z2)*this->derivs_[cd.node3_index()][2]
	+x2*z*this->derivs_[cd.node4_index()][1]
	+z*(-1 + 6*y - 6*y2 + 3*z - 2*z2)*cd.node5()
	+2*x*y*z*this->derivs_[cd.node5_index()][0]
	+y*(-2 + 3*y)*z*this->derivs_[cd.node5_index()][1]
	+(-1 + z)*z2*this->derivs_[cd.node5_index()][2]);

    derivs[2]=
      T(-((-1 + x + y)*(2*x2 - y + 2*y2 - x*(1 + 2*y) + 6*(-1 + z)*z))*cd.node0()
	+x*(-1 + 2*x - x2 + y2)*this->derivs_[cd.node0_index()][0]
	+(x2 - y12)*y*this->derivs_[cd.node0_index()][1]
	-((-1 + x + y)*(1 - 4*z + 3*z2))*this->derivs_[cd.node0_index()][2]
	+x*(1 - 3*x + 2*x2 - 6*z + 6*z2)*cd.node1()
	-((-1 + x)*x2)*this->derivs_[cd.node1_index()][0]
	-(x2*y)*this->derivs_[cd.node1_index()][0]
	+x*(1 - 4*z + 3*z2)*this->derivs_[cd.node1_index()][0]
	+y*(1 - 3*y + 2*y2 - 6*z + 6*z2)*cd.node2()
	-(x*y2)*this->derivs_[cd.node2_index()][0]
	-((-1 + y)*y2)*this->derivs_[cd.node2_index()][1]
	+y*(1 - 4*z + 3*z2)*this->derivs_[cd.node2_index()][2]
	+(-1 + x + y)*(2*x2 - y + 2*y2 - x*(1 + 2*y) + 6*(-1 + z)*z)*cd.node3()
	+x*(1 - 2*x + x2 - y2)*this->derivs_[cd.node3_index()][0]
	+(-x2 + y12)*y*this->derivs_[cd.node3_index()][1]
	-((-1 + x + y)*z*(-2 + 3*z))*this->derivs_[cd.node3_index()][2]
	+x*(-1 + 3*x - 2*x2 + 6*z - 6*z2)*cd.node4()
	+(-1 + x)*x2*this->derivs_[cd.node4_index()][0]
	+x2*y*this->derivs_[cd.node4_index()][1]
	+x*z*(-2 + 3*z)*this->derivs_[cd.node4_index()][2]
	+y*(-1 + 3*y - 2*y2 + 6*z - 6*z2)*cd.node5()
	+x*y2*this->derivs_[cd.node5_index()][0]
	+(-1 + y)*y2*this->derivs_[cd.node5_index()][1]
	+y*z*(-2 + 3*z)*this->derivs_[cd.node5_index()][2]);
  }  

  //! get parametric coordinate for value within the element
  //! iterative solution...
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  

  {
    PrismLocate< PrismCubicHmt<T> > CL;
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
    if (unit_faces[face][3]==-1) 
      return get_area3<TriGaussian3<double> >(this, face, cd);
    else
      return get_area3<QuadGaussian3<double> >(this, face, cd);
  }
 
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & cd) const  
  {
    return get_volume3(this, cd);
  }
  
  static  const std::string type_name(int n = -1);

  virtual void io (Piostream& str);

protected:
  //! support data (node data is elsewhere)

  std::vector<std::vector<T> >          derivs_; 

};


template <class T>
const TypeDescription* get_type_description(PrismCubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("PrismCubicHmt", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
PrismCubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("PrismCubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int PRISMCUBICHMT_VERSION = 1;
template <class T>
void
PrismCubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     PRISMCUBICHMT_VERSION);
  Pio(stream, this->derivs_);
  stream.end_class();
}

} //namespace SCIRun


#endif // PrismCubicHmt_h
