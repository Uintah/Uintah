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
 *  Viwer.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef SCIRun_LinSolver_h
#define SCIRun_LinSolver_h


#include <Core/CCA/spec/cca_sidl.h>

//namespace SCIRun {
  

#define myGoPort LinSolverGoPort

class LinSolver;

class myField2DPort : public virtual sci::cca::ports::Field2DPort {
public:
   virtual ~myField2DPort(){}
   void setParent(LinSolver *com){this->com=com;}
   SSIDL::array1<double>  getField();
 private:
   LinSolver *com;
};

class myGoPort : public virtual sci::cca::ports::GoPort {
public:
  virtual ~myGoPort(){}
  virtual int go();
  void setParent(LinSolver *com){this->com=com;}
  LinSolver *com;
};


class LinSolver: public sci::cca::Component{
                
  public:
    LinSolver();
    virtual ~LinSolver();
    sci::cca::Services::pointer getServices(){return services;}
    virtual void setServices(const sci::cca::Services::pointer& svc);
    bool jacobi(const SSIDL::array2<double> &A, 
		const SSIDL::array1<double> &b);
    SSIDL::array1<double> solution;
 private:

    LinSolver(const LinSolver&);
    LinSolver& operator=(const LinSolver&);
    myGoPort goPort;
    myField2DPort fieldPort;
    sci::cca::Services::pointer services;
  };
//}




#endif
