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
 *  CCAComponentModel.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCA_CCAComponentModel_h
#define SCIRun_CCA_CCAComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <SCIRun/ComponentInstance.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <vector>
#include <string>
#include <map>

namespace SCIRun {

class SCIRunFramework;
class CCAComponentDescription;
class CCAComponentInstance;

class CCAComponentModel : public ComponentModel
{
public:
  CCAComponentModel(SCIRunFramework* framework);
  virtual ~CCAComponentModel();
  
  sci::cca::Services::pointer createServices(const std::string& instanceName,
                                             const std::string& className,
                                             const sci::cca::TypeMap::pointer& properties);
  virtual bool haveComponent(const std::string& type);
  virtual ComponentInstance* createInstance(const std::string& name,
                                            const std::string& type,
                                            const sci::cca::TypeMap::pointer& properties);
  
  virtual bool destroyInstance(ComponentInstance *ci);
  virtual std::string getName() const;
  virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
                                     bool);
  int addLoader(resourceReference *rr);
  int removeLoader(const std::string &loaderName);
  
  /**
   * Get/Set the directory path to the XML files describing CCA
   * components. By default, sidlXMLPath is initialized to the
   * environment variable SIDL_XML_PATH.
   */
  std::string getSidlXMLPath() const
  { return sidlXMLPath; }
  void setSidlXMLPath( const std::string& s)
  { sidlXMLPath = s; }
  
private:
  SCIRunFramework* framework;
  typedef std::map<std::string, CCAComponentDescription*> componentDB_type;
  componentDB_type components;
  std::string sidlXMLPath;
    
  void destroyComponentList();
  void buildComponentList();
  void readComponentDescription(const std::string& file);
  resourceReference *getLoader(std::string loaderName);
  CCAComponentModel(const CCAComponentModel&);
  CCAComponentModel& operator=(const CCAComponentModel&);
  
  std::vector<resourceReference* > loaderList;
};
}

#endif
