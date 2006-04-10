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
 *  PHello.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#ifndef SCIRun_CCA_Components_PHello_h
#define SCIRun_CCA_Components_PHello_h

#include <sci_defs/mpi_defs.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <mpi.h>
namespace SCIRun {
  
  class myUIPort : public virtual sci::cca::ports::UIPort {
  public:
    int ui();
  };
  
  class myGoPort : public virtual sci::cca::ports::GoPort {
  public:
    myGoPort(const sci::cca::Services::pointer& svc);
    int go();
  private:
    sci::cca::Services::pointer services;
  };
  
  
  class PHello : public sci::cca::Component{
    
  public:
    PHello();
    ~PHello();
    void setServices(const sci::cca::Services::pointer& svc);
    void setCommunicator(int comm);
    //    MPI_Comm MPI_COMM_COM;
  private:
    PHello(const PHello&);
    PHello& operator=(const PHello&);
    sci::cca::Services::pointer services;
  };
  
  
} //namespace SCIRun


#endif
