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

#include <Core/CCA/spec/cca_sidl.h>

#include <string>

namespace SCIRun {
  class BuilderWindow;
  class myBuilderPort : public virtual sci::cca::ports::BuilderPort {
  public:
    virtual ~myBuilderPort(){}
    virtual void setServices(const sci::cca::Services::pointer& svc);
    virtual void buildRemotePackageMenus(const  sci::cca::ports::ComponentRepository::pointer &reg,
				    const std::string &frameworkURL);
  protected:
    sci::cca::Services::pointer services;
    BuilderWindow* builder;
  };

  class Builder : public sci::cca::Component {
  public:
    Builder();
    virtual ~Builder();
    virtual void setServices(const sci::cca::Services::pointer& svc);
  private:
    Builder(const Builder&);
    Builder& operator=(const Builder&);
    myBuilderPort builderPort;

  };
} //namespace SCIRun

#endif
