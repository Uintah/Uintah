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
//    File   : Basis.h
//    Author : Frank B. Sachse
//    Date   : Mon Oct 21 

#if !defined(Basis_h)
#define Basis_h

#include <vector>
#include <cfloat>
#include <Core/Util/Assert.h>

namespace SCIRun {

//! Class for describing interfaces to basis elements
template <class T>
class BasisSimple 
{
public:
  BasisSimple() {}
  virtual ~BasisSimple() {}
  
  static int polynomial_order() { return 0; }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    ASSERTFAIL("interpolate not supported by basis");
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
  {
    ASSERTFAIL("derivate not supported by basis");
  }

  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const
  {
    ASSERTFAIL("get_coords not supported by basis");
  }

  //! get arc length for edge of element
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    ASSERTFAIL("get_arc_length not supported by basis");
  }
 
  //! get area for face of element
  template <class ElemData>
  double get_area(const unsigned face, const ElemData &cd) const  
  {
    ASSERTFAIL("get_area not supported by basis");
  }
 
  //! get volume for element
  template <class ElemData>
  double get_volume(const ElemData &cd) const  
  {
    ASSERTFAIL("get_volume not supported by basis");
  }
 
  //! add a node value 
  void add_node_value(const T &p) 
  {
    ASSERTFAIL("add_node_value not supported by basis");
  }

  //! add a derivative value 
  void add_derivatives(const std::vector<T> &p) 
  {
    ASSERTFAIL("add_derivative not supported by basis");
  }

  //! add scale factors 
  void add_scalefactors(const std::vector<T> &p) 
  {
    ASSERTFAIL("add_scalefactors not supported by basis");
  }

  //! return number of additional nodes
  inline int size_node_values()
  {
    return 0;
  }

  //! return number of additional derivatives
  inline int size_derivatives()
  {
    return 0;
  }
};


//! Class for describing interfaces to basis elements with additional nodes
template <class T>
class BasisAddNodes : public BasisSimple<T> 
{
public:
  BasisAddNodes() {}
  virtual ~BasisAddNodes() {}

  //! add a node value corresponding to edge
  inline void add_node_value(const T &p) { nodes_.push_back(p); }

  //! return node value
  inline T& node_values(int i) { return &nodes_[i]; }

  //! return number of additional nodes
  inline int size_node_values() { return nodes_.size(); }

protected:
  std::vector<T>          nodes_;  
};


//! Class for describing interfaces to basis elements with 
//! additional derivatives
template <class T>
class BasisAddDerivatives : public BasisSimple<T> 
{
public:
  BasisAddDerivatives() {}
  virtual ~BasisAddDerivatives() {}

  //! add derivative values (dx, dy, dxy) for nodes.
  inline void add_derivatives(const std::vector<T> &p) 
  {
    derivs_.push_back(p); 
  }

  //! return number of additional derivatives
  inline int size_derivatives() { return derivs_.size(); }

protected:
  std::vector<std::vector<T> > derivs_; 
};


//! Class for describing interfaces to basis elements with 
//! additional derivatives and scale factors at nodes.
template <class T>
class BasisAddDerivativesScaleFactors : public BasisAddDerivatives<T> 
{
public:
  BasisAddDerivativesScaleFactors() {}
  virtual ~BasisAddDerivativesScaleFactors() {}

  //! add scale factors (sdx, sdy) for nodes.
  inline void add_scalefactors(const std::vector<double> &p) 
  {
    scalefactors_.push_back(p); 
  }

  //! return number of additional derivatives
  inline int size_scalefactors() { return scalefactors_.size(); }

protected:
  std::vector<std::vector<double> > scalefactors_; 
};

//! Class for describing interfaces to basis elements with 
//! additional derivatives and scale factors at edges
template <class T>
class BasisAddDerivativesScaleFactorsEdges : public BasisAddDerivatives<T> 
{
public:
  BasisAddDerivativesScaleFactorsEdges() {}
  virtual ~BasisAddDerivativesScaleFactorsEdges() {}

  //! add scale factors (sdx, sdy) for nodes.
  inline void add_scalefactors(const std::vector<double> &p) 
  { scalefactors_.push_back(p[0]); }

  //! return number of additional derivatives
  inline int size_scalefactors() { return scalefactors_.size(); }

protected:
  std::vector<double> scalefactors_; 
};

}
#endif // Basis_h
