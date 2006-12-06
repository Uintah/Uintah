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
 *  BridgeComponentModel.h:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September, 2003
 *
 */

#ifndef Framework_CCA_BridgeComponentModel_h
#define Framework_CCA_BridgeComponentModel_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <Framework/ComponentModel.h>
#include <Framework/ComponentInstance.h>
#include <Framework/Bridge/BridgeServices.h>
#include <Framework/Bridge/BridgeComponent.h>
#include <string>
#include <map>

namespace SCIRun {
class SCIRunFramework;
class BridgeComponentDescription;
class BridgeComponentInstance;

class BridgeComponentModel : public ComponentModel {
public:
  BridgeComponentModel(SCIRunFramework* framework);
  virtual ~BridgeComponentModel();

  BridgeServices* createServices(const std::string& instanceName,
				 const std::string& className,
				 const sci::cca::TypeMap::pointer &tm);

  virtual bool haveComponent(const std::string& type);

  virtual ComponentInstance*
  createInstance(const std::string& name,
		 const std::string& type,
		 const sci::cca::TypeMap::pointer &tm);

  virtual bool destroyInstance(ComponentInstance *ci);

  virtual const std::string getName() const { return "Bridge"; }

  virtual void
  listAllComponentTypes(std::vector<ComponentDescription*>&, bool);

  static const std::string DEFAULT_XML_PATH;

private:
  typedef std::map<std::string, BridgeComponentDescription*> componentDB_type;
  componentDB_type components;
  SCIRun::Mutex lock_components;

  virtual void destroyComponentList();
  virtual void buildComponentList(const StringVector& files=StringVector());
  virtual void setComponentDescription(const std::string& type, const std::string& library="");

  void readComponentDescriptions(const std::string& file);

  BridgeComponentModel(const BridgeComponentModel&);
  BridgeComponentModel& operator=(const BridgeComponentModel&);

  std::vector<resourceReference* > loaderList;

};
}

#endif
