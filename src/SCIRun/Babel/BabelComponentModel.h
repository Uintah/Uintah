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
 *  BabelComponentModel.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#ifndef SCIRun_Babel_BabelComponentModel_h
#define SCIRun_Babel_BabelComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/Babel/gov_cca.hh>
#include <string>
#include <map>

namespace SCIRun {
  class SCIRunFramework;
  class BabelComponentDescription;
  class BabelComponentInstance;

  class BabelComponentModel : public ComponentModel {
  public:
    BabelComponentModel(SCIRunFramework* framework);
    virtual ~BabelComponentModel();

    gov::cca::Services createServices(const std::string& instanceName,
					       const std::string& className,
					       const gov::cca::TypeMap& properties);
    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);

    virtual std::string createComponent(const std::string& name,
					 const std::string& type);
						     
    virtual bool destroyInstance(ComponentInstance *ci);
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
    typedef std::map<std::string, BabelComponentDescription*> componentDB_type;
    componentDB_type components;
    std::string sidlXMLPath;
    
    void destroyComponentList();
    void buildComponentList();
    void readComponentDescription(const std::string& file);

    BabelComponentModel(const BabelComponentModel&);
    BabelComponentModel& operator=(const BabelComponentModel&);
  };
} //namespace SCIRun

#endif
