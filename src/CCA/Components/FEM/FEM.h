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

#ifndef SCIRun_Framework_FEM_h
#define SCIRun_Framework_FEM_h

#include <Core/CCA/spec/cca_sidl.h>

#define myUIPort FEMUIPort
#define myGoPort FEMGoPort

//namespace SCIRun {
 
class FEM; 
class myUIPort : public virtual sci::cca::ports::UIPort {
public:
  virtual ~myUIPort(){}
  virtual int ui();
  void setParent(FEM *com){this->com=com;}
  FEM *com;
};

class myGoPort : public virtual sci::cca::ports::GoPort {
public:
  virtual ~myGoPort(){}
  virtual int go();
  void setParent(FEM *com){this->com=com;}
  FEM *com;
};


class myPDEMatrixPort: public virtual sci::cca::ports::PDEMatrixPort{
 public:
  virtual ~myPDEMatrixPort(){}
  virtual void getMatrix(SSIDL::array2<double>&);
  virtual SSIDL::array1<double> getVector();
  int getSize();
  void setParent(FEM *com){this->com=com;}
  FEM *com;  
};


class FEM : public sci::cca::Component{
                
  public:
    FEM();
    virtual ~FEM();

    virtual void setServices(const sci::cca::Services::pointer& svc);
    virtual sci::cca::Services::pointer getServices(){return services;}
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
     //Matrix::pointer Ag;
    SSIDL::array1<double> fg;
    SSIDL::array1<int> dirichletNodes;
    SSIDL::array1<double> dirichletValues;
    myPDEMatrixPort *matrixPort;
    myGoPort *goPort;

  private:

    FEM(const FEM&);
    FEM& operator=(const FEM&);
    sci::cca::Services::pointer services;
  };
//}




#endif


