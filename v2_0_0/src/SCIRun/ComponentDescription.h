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
 *  ComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ComponentDescription_h
#define SCIRun_ComponentDescription_h

#include <string>

namespace SCIRun {
  class ComponentModel;
  class SCIRunFramework;
  class ComponentDescription {
  public:
    ComponentDescription();
    virtual ~ComponentDescription();

    virtual std::string getType() const = 0;
    virtual const ComponentModel* getModel() const = 0;
    virtual std::string getLoaderName() const;
  private:
    ComponentDescription(const ComponentDescription&);
    ComponentDescription& operator=(const ComponentDescription&);
  };
}

#endif
