/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  FEM.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef CCA_Components_FEM_h
#define CCA_Components_FEM_h

#include <Core/CCA/spec/sci_sidl.h>

namespace SCIRun {
 
class myFEMmatrixPort: public virtual sci::cca::ports::FEMmatrixPort{
 public:
  virtual ~myFEMmatrixPort(){}
  virtual int makeFEMmatrices(const SSIDL::array1<int> &tmesh, const SSIDL::array1<double> &nodes, 
			      const SSIDL::array1<int> &dirichletNodes, const SSIDL::array1<double> &dirichletValues,
			      SSIDL::array2<double> &Ag, SSIDL::array1<double> &fg, int &size);
};


class FEM : public sci::cca::Component{
                
  public:
    FEM();
    virtual ~FEM();
    virtual void setServices(const sci::cca::Services::pointer& svc);
  private:
    FEM(const FEM&);
    FEM& operator=(const FEM&);
    sci::cca::Services::pointer services;
  };


 class FEMgenerator{
                
  public:
    FEMgenerator(const SSIDL::array1<int> &dirichletNodes, const SSIDL::array1<double> &dirichletValues);

    void diffTriangle(double b[3], double c[3], double &area,
			   const double x[3], const double y[3]);
    void localMatrices(double A[3][3], double f[3], 
		       const double x[3], const double y[3]);
    void globalMatrices(const SSIDL::array1<double> &node1d,
			const SSIDL::array1<int> &tmesh1d);
    double source(int index);
    double boundary(int index);
    bool isConst(int index);
    SSIDL::array2<double> Ag;
    SSIDL::array1<double> fg;
 private:
    //TODO: need use pointers instead for better performance
    SSIDL::array1<int> dirichletNodes;
    SSIDL::array1<double> dirichletValues;
  };
}

#endif


