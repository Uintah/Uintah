/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include "Matrix.h"

#define myUIPort FEMUIPort
#define myGoPort FEMGoPort

//namespace SCIRun {
 
class FEM; 
class myUIPort : public virtual gov::cca::ports::UIPort {
public:
  virtual ~myUIPort(){}
  virtual int ui();
  void setParent(FEM *com){this->com=com;}
  FEM *com;
};

class myGoPort : public virtual gov::cca::ports::GoPort {
public:
  virtual ~myGoPort(){}
  virtual int go();
  void setParent(FEM *com){this->com=com;}
  FEM *com;
};


class myPDEMatrixPort: public virtual gov::cca::ports::PDEMatrixPort{
 public:
  virtual ~myPDEMatrixPort(){}
  virtual gov::cca::Matrix::pointer getMatrix();
  virtual SIDL::array1<double> getVector();
  void setParent(FEM *com){this->com=com;}
  FEM *com;  
};


class FEM : public gov::cca::Component{
                
  public:
    FEM();
    virtual ~FEM();

    virtual void setServices(const gov::cca::Services::pointer& svc);
    virtual gov::cca::Services::pointer getServices(){return services;}
    void diffTriangle(double b[3], double c[3], double &area,
			   const double x[3], const double y[3]);
    void localMatrices(double A[3][3], double f[3], 
		       const double x[3], const double y[3]);
    void globalMatrices(const SIDL::array1<double> &node1d,
			const SIDL::array1<int> &tmesh1d);
    double source(int index);
    double boundary(int index);
    bool isConst(int index);

    Matrix::pointer Ag;
    SIDL::array1<double> fg;
    SIDL::array1<int> dirichletNodes;
    SIDL::array1<double> dirichletValues;
  private:

    FEM(const FEM&);
    FEM& operator=(const FEM&);
    myUIPort uiPort;
    myGoPort goPort;
    myPDEMatrixPort matrixPort;


    gov::cca::Services::pointer services;
  };
//}




#endif


