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
#include <vector>
#include <string>

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

    /** Get/set the directory path to component DLLs.  By default,
     * the sidlDLLPath is initialized to the environment variable
     * SIDL_DLL_PATH. */
    std::string getSidlDLLPath() const
    { return sidlDLLPath; }
    void setSidlDLLPath( const std::string& s)
    { sidlDLLPath = s; }
    
    /** Get/Set the filename for the DTD describing valid xml files for this
        component model. */
    //    std::string getGrammarFileName() const
    //    { return grammarFileName; }
    //    void setGrammarFileName( const std::string& s )
    //    { grammarFileName = s; }

    /** Breaks a concatenated list of paths into a vector of paths. Splits on
     * the ';' character. */
    std::vector<std::string> static splitPathString(const std::string &);
    
  private:
    SCIRunFramework* framework;
    typedef std::map<std::string, VtkComponentDescription*> componentDB_type;
    componentDB_type components;
    void destroyComponentList();
    void buildComponentList();
    void readComponentDescription(const std::string& file);
    std::string sidlXMLPath;
    std::string sidlDLLPath;
    //    std::string grammarFileName;
    

    VtkComponentModel(const VtkComponentModel&);
    VtkComponentModel& operator=(const VtkComponentModel&);
  };
}

#endif
