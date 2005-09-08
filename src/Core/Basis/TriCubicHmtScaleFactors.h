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
//    File   : TriCubicHmtScaleFactors.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mar 01 2005

#if !defined(TriCubicHmtScaleFactors_h)
#define TriCubicHmtScaleFactors_h

#include <TriLinearLgn.h>

namespace SCIRun {

  //! Class for handling of element of type triangle with cubic hermitian interpolation with scale factors
  template <class T>
    class TriCubicHmtScaleFactors : public TriApprox<T> {
  public:
    static int GaussianNum;
    double GaussianPoints[7][2];
    double GaussianWeights[7];

    TriCubicHmtScaleFactors() {}
    virtual ~TriCubicHmScaleFactorst() {}

    //! get value at parametric coordinate 
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x=coords[0], y=coords[1]; 
	const double x2=x*x, y2=y*y;

	const double sdx0=derivs_[cd.node0_index()][0]*scalefactors_[cd.elem][0];
	const double sdx1=derivs_[cd.node1_index()][0]*scalefactors_[cd.elem][0];
	const double sdx2=derivs_[cd.node2_index()][0]*scalefactors_[cd.elem][0];

	const double sdy0=derivs_[cd.node0_index()][1]*scalefactors_[cd.elem][1];
	const double sdy1=derivs_[cd.node1_index()][1]*scalefactors_[cd.elem][1];
	const double sdy2=derivs_[cd.node2_index()][1]*scalefactors_[cd.elem][1];

	const double sdxy0=derivs_[cd.node0_index()][2]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
	const double sdxy1=derivs_[cd.node1_index()][2]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
	const double sdxy2=derivs_[cd.node2_index()][2]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];

	return (-1 + x + y)*(-1 - x + 2*x2 - y - 2*x*y + 2*y2)*cd.node0()
	  +x*(1 - 2*x + x2 - 3*y2 + 2*y3)*sdx0
	  +y*(1 - 3*x2 + 2*x3 - 2*y + y2)*sdy0
	  +x*y*(1 - 2*x + x2 - 2*y + y2)*sdxy0
	  -(x2*(-3 + 2*x))*cd.node1()
	  +(-1 + x)*x2*sdx1
	  -(x2*(-3 + 2*x)*y)*sdy1
	  +(-1 + x)*x2*y*sdxy1
	  -y2*(-3 + 2*y)*cd.node2()
	  -(x*y2*(-3 + 2*y))*sdx2
	  (-1 + y)*y2*sdy2
	  +x*(-1 + y)*y2*sdxy2;
      }
  
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	const double x=coords[0], y=coords[1];  
	const double x2=x*x, y2=y*y;
	const double y12=(y-1)*(y-1);

	derivs.size(2);

	const double sdx0=derivs_[cd.node0_index()][0]*scalefactors_[cd.elem][0];
	const double sdx1=derivs_[cd.node1_index()][0]*scalefactors_[cd.elem][0];
	const double sdx2=derivs_[cd.node2_index()][0]*scalefactors_[cd.elem][0];

	const double sdy0=derivs_[cd.node0_index()][1]*scalefactors_[cd.elem][1];
	const double sdy1=derivs_[cd.node1_index()][1]*scalefactors_[cd.elem][1];
	const double sdy2=derivs_[cd.node2_index()][1]*scalefactors_[cd.elem][1];

	const double sdxy0=derivs_[cd.node0_index()][2]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
	const double sdxy1=derivs_[cd.node1_index()][2]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];
	const double sdxy2=derivs_[cd.node2_index()][2]*scalefactors_[cd.elem][0]*scalefactors_[cd.elem][1];

	derivs[0]=6*(-1 + x)*x*cd.node0()
	  +(-4*x + 3*x2 + y12*(1 + 2*y))*sdx0
	  +6*(-1 + x)*x*y*sdy0
	  +(-4*x + 3*x2 + y12)*y*sdxy0
	  -6*(-1 + x)*x*cd.node1()
	  +x*(-2 + 3*x)*sdx1
	  -6*(-1 + x)*x*y*sdy1
	  +x*(-2 + 3*x)*y*sdxy1
	  +(3 - 2*y)*y2*sdx2
	  +(-1 + y)*y2*sdxy2;

	derivs[1]=6*(-1 + y)*y*cd.node0()
	  +6*x*(-1 + y)*y*sdx0
	  +(1 - 3*x2 + 2*x3 - 4*y + 3*y2)*sdy0
	  +x*(1 - 2*x + x2 - 4*y + 3*y2)*sdxy0
	  +(3 - 2*x)*x2*sdy1
	  +(-1 + x)*x2*sdxy1
	  -6*(-1 + y)*y*cd.node2()
	  -6*x*(-1 + y)*y*sdx2
	  +y*(-2 + 3*y)*sdy2 
	  +x*y*(-2 + 3*y)*sdxy2;
      }
  
    //! get the parametric coordinate for value within the element.
    //! iterative solution...
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const  
      {
      }

    //! add derivative values (dx, dy, dxy) for nodes.
    void add_derivatives(const vector<T> &p) { derivs_.push_back(p); }

    //! add scale factors (sdx, sdy) for nodes.
    void add_scalefactors(const vector<T> &p) { scalefactors_.push_back(p); }

    static const string type_name(int n = -1);

    virtual void io (Piostream& str);

  protected:
    double distance(const T&, const T&) const;

    //! support data 
    vector<T[2]>          derivs_; 
    vector<T[3]>          scalefactors_; 
  };


  template <class T>
    const TypeDescription* get_type_description(TriCubicHmt<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(TriCubicHmt<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    TriCubicHmt<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("TriCubicHmt");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  const int TRICUBICHMTSCALEFACTORS_VERSION = 1;
  template <class T>
    void
    TriCubicHmt<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), TRICUBICHMTSCALEFACTORS_VERSION);
      Pio(stream, derivs_);
      Pio(stream, scalefactors_);
      stream.end_class();
    }

  template <class T>
    int TriCubicHmtScaleFactors<T>::GaussianNum = 7;

  template <class T>
    double TriCubicHmtScaleFactors<T>::GaussianPoints[7][2] = {
    {0.1012865073, 0.1012865073}, {0.7974269853, 0.1012865073}, {0.1012865073, 0.7974269853},
    {0.4701420641, 0.0597158717}, {0.4701420641, 0.4701420641}, {0.0597158717, 0.4701420641},
    {0.3333333333, 0.3333333333}};

  template <class T>
    double TriCubicHmtScaleFactors<T>::GaussianWeights[7] = 
    {0.1259391805, 0.1259391805, 0.1259391805, 0.1323941527, 0.1323941527, 0.1323941527, 0.0225};

} //namespace SCIRun

#endif // TriCubicHmtScaleFactors_h
