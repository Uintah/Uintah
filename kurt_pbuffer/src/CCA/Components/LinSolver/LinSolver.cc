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
 *  LinSolver.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/LinSolver/LinSolver.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <qmessagebox.h>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_LinSolver()
{
  return sci::cca::Component::pointer(new LinSolver);
}

void LinSolver::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  LSComponentIcon::pointer cip(&ciPort);
  myLinSolverPort::pointer lsp(new myLinSolverPort);

  svc->addProvidesPort(lsp, "linsolver", "sci.cca.ports.LinSolverPort",
    sci::cca::TypeMap::pointer(0));
  svc->addProvidesPort(cip, "icon", "sci.cca.ports.ComponentIcon",
    sci::cca::TypeMap::pointer(0));

}

int 
myLinSolverPort::jacobi(const SSIDL::array2<double> &A, 
			const SSIDL::array1<double> &b,::SSIDL::array1< double>& x)
{
  //we might set the accurracy by UI
  double eps=1e-6;
  int maxiter=1000;
  
  int N=b.size();
  x.clear();
  while(x.size()<b.size()) x.push_back(1.0);

  int iter;
  
  for(iter=0; iter<maxiter; iter++){
    double norm=0;
    for(int i=0; i<N; i++){
      double res_i=0;
      for(int k=0; k<N; k++){
	res_i+=A[i][k]*x[k];
      }
      res_i-=b[i];
      norm+=res_i*res_i;
    }
    if(norm<eps*eps) break;
    //cerr<<"iter="<<iter<<"  norm2="<<norm<<endl;
    
    SSIDL::array1<double> tempx=x;
    for(int i=0; i<N; i++){
      tempx[i]=b[i];
      for(int k=0; k<N; k++){
	if(i==k) continue;
	tempx[i]-=A[i][k]*x[k];
      }
      tempx[i]/=A[i][i];
    }
    x=tempx;
  }

  if (iter != maxiter) {
      return 0;
  } else {
      return 1;
  }
}  


std::string LSComponentIcon::getDisplayName()
{
    return "Jacobi Linear Solver";
}

std::string LSComponentIcon::getDescription()
{
    return "Jacobi Linear Solver Component";
}

int LSComponentIcon::getProgressBar()
{
    return 0;
}
 
std::string LSComponentIcon::getIconShape()
{
    return "RECT";
}
