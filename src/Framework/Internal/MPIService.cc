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

#include <Framework/Internal/MPIService.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/CCA/CCAException.h>

namespace SCIRun {

MPIService::MPIService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:MPIService")
{
  prototype = MPI_COMM_WORLD;
}

MPIService::~MPIService() {
  size_t i, cap;
  cap = rlist.size();
  for (i = 0; i < cap; i++) {
    CommHolder *c = rlist[i];
    c->destroy();	
    delete c;
    rlist[i] = 0;
  }
  rlist.clear();
  cap = ulist.size();
  for (i = 0; i < cap; i++) {
    // this shouldn't occur if user is correct.
    CommHolder *c2 = ulist[i];
    delete c2;
    ulist[i] = 0;
  }
  ulist.clear();
}

InternalFrameworkServiceInstance*
MPIService::create(SCIRunFramework* framework)
{
  return new MPIService(framework);
}
  
long
MPIService::getComm() {
  // check for one recyclable
  if (rlist.size() > 0) {
    CommHolder *c = rlist[(rlist.size()-1)];
    rlist.erase(rlist.end()-1);
    ulist.push_back(c);
    MPI_Comm result = c->getComm();
    return result;
  }
  // make a new duplicate. save it and return it.
  MPI_Comm tmp;
  MPI_Comm_dup(prototype, &tmp);
  CommHolder *c2 = new CommHolder(tmp);
  ulist.push_back(c2);

  // convert to long and return
  long fcomm = MPI_Comm_c2f(tmp);
  return fcomm;
}

void 
MPIService::releaseComm(long comm) {
  MPI_Comm m = MPI_Comm_f2c(comm);
  if (m == MPI_COMM_NULL) {
    return;
  }
  // if we don't find it, ignore it quietly.
  size_t i, cap;
  cap = ulist.size();
  for (i = 0; i < cap; i++) {
    int result;
    MPI_Comm tmp = ulist[i]->getComm();
    MPI_Comm_compare(m,tmp,&result);
    if (result == MPI_IDENT) {
      CommHolder *c = ulist[i];
      rlist.push_back(c);
      ulist.erase(ulist.begin()+i);
      return;
    }
  }
}


}
