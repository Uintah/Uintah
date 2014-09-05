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
