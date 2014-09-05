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
 *  Viwer.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2003
 *
 */

#ifndef SCIRun_LinSolver_h
#define SCIRun_LinSolver_h


#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  
#define myGoPort LinSolverGoPort

class PLinSolver;

class myField2DPort : public virtual sci::cca::ports::Field2DPort {
public:
   virtual ~myField2DPort(){}
   void setParent(PLinSolver *com){this->com=com;}
   SSIDL::array1<double>  getField();
 private:
   PLinSolver *com;
};

class myGoPort : public virtual sci::cca::ports::GoPort {
public:
  virtual ~myGoPort(){}
  virtual int go();
  void setParent(PLinSolver *com){this->com=com;}
  PLinSolver *com;
};


class PLinSolver: public sci::cca::Component{
                
  public:
    PLinSolver();
    virtual ~PLinSolver();
    sci::cca::Services::pointer getServices(){return services;}
    virtual void setServices(const sci::cca::Services::pointer& svc);
    bool jacobi(const SSIDL::array2<double> &A, 
		const SSIDL::array1<double> &b,
		int sta, int fin );
    SSIDL::array1<double> solution;
    myField2DPort *fieldPort;
    myField2DPort::pointer fdp;
    int mpi_size;
    int mpi_rank;
 private:

    PLinSolver(const PLinSolver&);
    PLinSolver& operator=(const PLinSolver&);
    sci::cca::Services::pointer services;
  };
} 

#endif
