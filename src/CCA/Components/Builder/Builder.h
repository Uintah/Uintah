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

#include <Core/CCA/ccaspec/cca_sidl.h>

namespace SCIRun {
  class BuilderWindow;
  class Builder : public gov::cca::Component_interface {
  public:
    Builder();
    virtual ~Builder();

    virtual void setServices(const gov::cca::Services& svc);
  private:

    BuilderWindow* builder;
    Builder(const Builder&);
    Builder& operator=(const Builder&);

    gov::cca::Services services;
  };
}

#endif
