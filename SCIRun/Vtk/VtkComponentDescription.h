/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  VtkComponentDescription.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_VtkComponentDescription_h
#define SCIRun_Vtk_VtkComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun{
  class ComponentModel;
  class VtkComponentModel;
  class VtkComponentDescription : public ComponentDescription {
  public:
    VtkComponentDescription(VtkComponentModel* model, const std::string& type);
    virtual ~VtkComponentDescription();
    virtual std::string getType() const;
    virtual const ComponentModel* getModel() const;
    std::string type;

  private:
    VtkComponentModel* model;
    VtkComponentDescription(const VtkComponentDescription&);
    VtkComponentDescription& operator=(const VtkComponentDescription&);
  };
}

#endif
