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
 *  VtkComponentModel.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_VtkComponentModel_h
#define SCIRun_Vtk_VtkComponentModel_h

#include <SCIRun/ComponentModel.h>

namespace SCIRun{
  class VtkComponentDescription;
  class SCIRunFramework;

  class VtkComponentModel : public ComponentModel {
  public:
    VtkComponentModel(SCIRunFramework* framework);
    virtual ~VtkComponentModel();
    
    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);
    virtual bool destroyInstance(ComponentInstance * ic);
    virtual std::string getName() const;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
                                       bool);
    /**
     * Get/Set the directory path to the XML files describing Babel
     * components. By default, sidlXMLPath is initialized to the
     * environment variable SIDL_XML_PATH. This path is expected to
     * contain all .scl and .cca files for Babel components.
     */
    std::string getSidlXMLPath() const
    { return sidlXMLPath; }
    void setSidlXMLPath( const std::string& s)
    { sidlXMLPath = s; }
    
  private:
    SCIRunFramework* framework;
    typedef std::map<std::string, VtkComponentDescription*> componentDB_type;
    componentDB_type components;
    void destroyComponentList();
    void buildComponentList();
    void readComponentDescription(const std::string& file);
    std::string sidlXMLPath;
    

    VtkComponentModel(const VtkComponentModel&);
    VtkComponentModel& operator=(const VtkComponentModel&);
  };
}

#endif
