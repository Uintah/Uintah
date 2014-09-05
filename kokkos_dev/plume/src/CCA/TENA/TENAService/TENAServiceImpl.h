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
 *  TENAServiceImpl.h: Implementation of the CCA TENAService interface for SCIRun
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   October 2005
 *
 */

#ifndef CCA_TENA_TENAServiceImpl_h
#define CCA_TENA_TENAServiceImpl_h

#include <Core/Thread/Mutex.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;
  using namespace sci::cca::tena;

  struct TENAServiceInfo;

  /**
   * \class TENAServiceImpl
   *
   */
  
  class TENAServiceImpl : public TENAService
  {
  public:
    TENAServiceImpl(const Services::pointer &services);
    virtual ~TENAServiceImpl();
  
    virtual bool setConfiguration();

    virtual TENAExecution::pointer joinExecution(const std::string &name);

  protected:
    typedef std::map<std::string, TENAExecution::pointer> ExecutionsMap;

    Services::pointer services;
    ExecutionsMap executions;

    bool initialized;
    TENAServiceInfo *info;
    Mutex lock;

    TENAServiceImpl(const TENAServiceImpl &);
  };
}

#endif
