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

#ifndef Framework_Internal_MPIService_h
#define Framework_Internal_MPIService_h

#include <sci_metacomponents.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <Framework/Internal/InternalComponentModel.h>
#include <Framework/Internal/InternalFrameworkServiceInstance.h>

#include <map>
#include <mpi.h>


namespace SCIRun {

class SCIRunFramework;

class CommHolder {
private:
	MPI_Comm com;
public:
	CommHolder(MPI_Comm c) : com(c) {}
	~CommHolder(){}
	MPI_Comm getComm() { return com; }
	void destroy() {
		MPI_Comm_free(&com);
	}
}; // end class commholder


/**
 * \class MPIService
 *
 * The MPIService class is used to get an MPICommunicator
 * from the framework. Adapted from Ben Allan implementation
 * in CCAFFEINE
 *
 * \sa SCIRunFramework
 */

class MPIService : public sci::cca::ports::MPIService,
                   public InternalFrameworkServiceInstance {

public:
  virtual ~MPIService();

  /**
   * Factory method for creating an instance of a MPIService class.
   * Returns a reference counted pointer to a newly-allocated MPIService port.
   * The \em framework parameter is a pointer to the relevent framework
   * and the \em name parameter will become the unique name for the new port.
   */
  static InternalFrameworkServiceInstance *create(SCIRunFramework* framework);

  virtual sci::cca::Port::pointer getService(const std::string &) 
                        { return sci::cca::Port::pointer(this); }

  // long .sci.cca.ports.MPIService.getComm() throws .sci.cca.CCAException
  virtual long getComm();

  // void .sci.cca.ports.MPIService.releaseComm(in long comm) throws .sci.cca.CCAException
  virtual void releaseComm(long comm);

private:

  MPIService(SCIRunFramework* fwk);

  /** coms for recycle */
  std::vector< CommHolder *> rlist;
  /** coms in use */
  std::vector< CommHolder *> ulist;

  MPI_Comm prototype;
};

}

#endif
