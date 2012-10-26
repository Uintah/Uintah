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
//  
//    File   : TriCubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Oct 24 2005

#if !defined(TriCubicHmt_h)
#define TriCubicHmt_h

#include <Core/Basis/TriLinearLgn.h>

#include <Core/Basis/share.h>
namespace SCIRun {

//! Class for describing unit geometry of TriCubicHmt
  class SCISHARE TriCubicHmtUnitElement : public TriLinearLgnUnitElement {
  public:
  //! Parametric coordinates of vertices of unit edge
  static double unit_vertices[4][2];

  TriCubicHmtUnitElement() {}
  virtual ~TriCubicHmtUnitElement() {}

  //! return number of vertices
  static int number_of_vertices() { return 4; }
  //! return degrees of freedom
  static int dofs() { return 10; } 
};


//! Class for handling of element of type triangle with 
//! cubic hermitian interpolation
template <class T>
  class TriCubicHmt : public BasisAddDerivatives<T>, 
                      public TriApprox, 
		      public TriGaussian3<double>, 
		      public TriCubicHmtUnitElement 
{
public:
  typedef T value_type;

  TriCubicHmt() {}
  virtual ~TriCubicHmt() {}
    
  static int polynomial_order() { return 3; }

  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1];  

    w[0] = (-1 + x + y)*(-1 - x + 2*x*x - y + 11*x*y + 2*y*y);
    w[1] = +x*(-1+ x + y)*(-1 + x + 2*y);
    w[2] = +y*(-1 + x + y)*(-1 + 2*x + y);
    w[3] = -(x*(-3*x +2*x*x + 7*y - 7*x*y - 7*y*y));
    w[4] = +x*(-x + x*x + 2*y - 2*x*y- 2*y*y);
    w[5] = +x*y*(-1 + 2*x + y);
    w[6] = -(y*(7*x - 7*x*x - 3*y - 7*x*y+ 2*y*y));
    w[7] = +x*y*(-1 + x + 2*y);
    w[8] = +y*(2*x - 2*x*x - y - 2*x*y + y*y);
    w[9] = -27*x*y*(-1 + x + y);
  }

  //! get value at parametric coordinate
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[10];
    get_weights(coords, w); 

    return (T)(w[0] * cd.node0()                   +
	       w[1] * this->derivs_[cd.node0_index()][0] +
	       w[2] * this->derivs_[cd.node0_index()][1] +
	       w[3] * cd.node1()		   +
	       w[4] * this->derivs_[cd.node1_index()][0] +
	       w[5] * this->derivs_[cd.node1_index()][1] +
	       w[6] * cd.node2()		   +
	       w[7] * this->derivs_[cd.node2_index()][0] +
	       w[8] * this->derivs_[cd.node2_index()][1] +
	       w[9] * this->nodes_[cd.elem_index()]);    
  }
  
  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
   const double x=coords[0], y=coords[1];  
    w[0] = (6*x*x + 13*(-1 + y)*y + x*(-6 + 26*y));
    w[1] = (1 + 3*x*x - 3*y + 2*y*y + x*(-4 + 6*y));
    w[2] = y*(-3 + 4*x + 3*y);
    w[3] = (-6*x*x + 7*(-1 + y)*y+ 2*x*(3 + 7*y));
    w[4] = (3*x*x - 2*(-1 + y)*y - 2*x*(1 + 2*y));
    w[5] = y*(-1 + 4*x + y);
    w[6] = 7*y*(-1 + 2*x + y);
    w[7] = y*(-1 + 2*x + 2*y);
    w[8] = -2*y*(-1 + 2*x + y);
    w[9] = -27*y*(-1 + 2*x + y);
    w[10] = (13*x*x + 6*(-1+ y)*y + 13*x*(-1 + 2*y));
    w[11] = x*(-3 + 3*x + 4*y);
    w[12] = (1 + 2*x*x - 4*y + 3*y*y + x*(-3 + 6*y));
    w[13] = 7*x*(-1 + x + 2*y);
    w[14] = -2*x*(-1 + x + 2*y);
    w[15] = x*(-1 + 2*x + 2*y);
    w[16] = (7*x*x - 6*(-1 + y)*y + 7*x*(-1 + 2*y));
    w[17] = x*(-1 + x + 4*y);
    w[18] = (-2*x*x + x*(2 - 4*y) + y*(-2 + 3*y));
    w[19] = -27*x*(-1 + x + 2*y);
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1];  

    derivs.resize(2);

    derivs[0]=
      T((6*x*x + 13*(-1 + y)*y + x*(-6 + 26*y))*cd.node0()
	+(1 + 3*x*x - 3*y + 2*y*y + x*(-4 + 6*y))*this->derivs_[cd.node0_index()][0]
	+y*(-3 + 4*x + 3*y)*this->derivs_[cd.node0_index()][1]
	+(-6*x*x + 7*(-1 + y)*y + 2*x*(3 + 7*y))*cd.node1()
	+(3*x*x - 2*(-1 + y)*y - 2*x*(1 + 2*y))*this->derivs_[cd.node1_index()][0]
	+y*(-1 + 4*x + y)*this->derivs_[cd.node1_index()][1]
	+7*y*(-1 + 2*x + y)*cd.node2()
	+y*(-1 + 2*x + 2*y)*this->derivs_[cd.node2_index()][0]
	-2*y*(-1 + 2*x + y)*this->derivs_[cd.node2_index()][1]
	-27*y*(-1 + 2*x + y)*this->nodes_[cd.elem_index()]);
    
    derivs[1]=
      T((13*x*x + 6*(-1+ y)*y + 13*x*(-1 + 2*y))*cd.node0()
	+x*(-3 + 3*x + 4*y)*this->derivs_[cd.node0_index()][0]
	+(1 + 2*x*x - 4*y + 3*y*y + x*(-3 + 6*y))*this->derivs_[cd.node0_index()][1]
	+7*x*(-1 + x + 2*y)*cd.node1()
	-2*x*(-1 + x + 2*y)*this->derivs_[cd.node1_index()][0]
	+x*(-1 + 2*x + 2*y)*this->derivs_[cd.node1_index()][1]
	+(7*x*x - 6*(-1 + y)*y + 7*x*(-1 + 2*y))*cd.node2()
	+x*(-1 + x + 4*y)*this->derivs_[cd.node2_index()][0]
	+(-2*x*x + x*(2 - 4*y) + y*(-2 + 3*y))*this->derivs_[cd.node2_index()][1]
	-27*x*(-1 + x + 2*y)*this->nodes_[cd.elem_index()]);
  }
  
  //! get the parametric coordinate for value within the element.
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    TriLocate< TriCubicHmt<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }
 
  //! get arc length for edge
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    return get_arc2d_length<CrvGaussian2<double> >(this, edge, cd);
  }
 
  //! get area
  template <class ElemData>
    double get_area(const unsigned face, const ElemData &cd) const  
  {
    return get_area2<TriGaussian3<double> >(this, face, cd);
  }
 
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & /* cd */) const  
  {
    return 0.;
  }
  
  //! add a node value corresponding to center of triangle
  void add_node_value(const T &p) { this->nodes_.push_back(p); }

  //! return number of additional nodes
  int size_node_values() { return nodes_.size(); }

  static const std::string type_name(int n = -1);

  virtual void io (Piostream& str);

protected:
  //! Cubic Hermitian needs additional nodes stored for each edge
  //! in the topology.
  std::vector<T> nodes_; 
};


template <class T>
const TypeDescription* get_type_description(TriCubicHmt<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("TriCubicHmt", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const std::string
TriCubicHmt<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("TriCubicHmt");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int TRICUBICHMT_VERSION = 1;
template <class T>
void
TriCubicHmt<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     TRICUBICHMT_VERSION);
  Pio(stream, this->derivs_);
  Pio(stream, nodes_);
  stream.end_class();
}

} //namespace SCIRun

#endif // TriCubicHmt_h
