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
 *  InternalComponentInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Internal_InternalComponentInstance_h
#define SCIRun_Internal_InternalComponentInstance_h

#include <SCIRun/ComponentInstance.h>

namespace SCIRun {
  class InternalComponentInstance : public ComponentInstance {
  public:
    InternalComponentInstance(SCIRunFramework* framework,
			      const std::string& intanceName,
			      const std::string& className);

    virtual PortInstance* getPortInstance(const std::string& name);
    virtual ~InternalComponentInstance();

    virtual sci::cca::Port::pointer getService(const std::string& name) = 0;
    virtual PortInstanceIterator* getPorts();

    void incrementUseCount();
    bool decrementUseCount();
  private:
    int useCount;

    InternalComponentInstance(const InternalComponentInstance&);
    InternalComponentInstance& operator=(const InternalComponentInstance&);
  };
}

#endif
