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
 *  Builder.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_Builder_h
#define SCIRun_Framework_Builder_h

#include <Core/CCA/spec/sci_sidl.h>

#include <string>


namespace SCIRun{

class BuilderWindow;
class Builder;

/**
 * \class myBuilderPort
 *
 * An extension and implementation of sci::cca::ports::BuilderPort. 
 */
class myBuilderPort : public virtual sci::cca::ports::BuilderPort
{
public:
  virtual ~myBuilderPort();
  /** Obtain Services handle, through which the component communicates with the
      framework. */
  virtual void setServices(const sci::cca::Services::pointer& svc);
  
    /** ? */
    virtual void buildRemotePackageMenus(
    const sci::cca::ports::ComponentRepository::pointer &reg,
    const std::string &frameworkURL);

protected:
  friend class Builder;
  sci::cca::Services::pointer services;
  BuilderWindow* builder;
};


/**
 * \class Builder
 *
 * ?
 */
class Builder : public sci::cca::Component
{
public:
  Builder();
  virtual ~Builder();
  
  /** Obtain Services handle, through which the component communicates with the
      framework. */
  virtual void setServices(const sci::cca::Services::pointer& svc);

  virtual void setCommunicator(int comm) {  };

private:
  Builder(const Builder&);
  Builder& operator=(const Builder&);
  myBuilderPort builderPort;
};

} //namespace SCIRun

#endif
