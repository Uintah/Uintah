/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


// GUIService.h

#include <Core/CCA/spec/cca_sidl.h>
#include <Framework/Internal/InternalComponentModel.h>
#include <Framework/Internal/InternalFrameworkServiceInstance.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {

class SCIRunFramework;

typedef std::map<std::string, sci::cca::GUIBuilder::pointer> GUIBuilderMap;

class GUIService : public sci::cca::ports::GUIService,
                   public InternalFrameworkServiceInstance {
public:
  virtual ~GUIService() {}
  static InternalFrameworkServiceInstance* create(SCIRunFramework* framework);

  // InternalFrameworkServiceInstance:
  virtual sci::cca::ComponentID::pointer
  createInstance(const std::string& instanceName,
                 const std::string& className,
                 const sci::cca::TypeMap::pointer &properties);

  virtual sci::cca::Port::pointer getService(const std::string &);

  // sci::cca::ports::GUIService:
  virtual void addBuilder(const std::string& builderName, const sci::cca::GUIBuilder::pointer& builder);
  virtual void removeBuilder(const std::string& builderName);

  //   // refresh menus???
  //   void buildComponentMenus();

  // tell guis to update progress for???
  virtual void updateProgress(const sci::cca::ComponentID::pointer& cid, int progressPercent);
  virtual void updateComponentModels();

private:
  GUIService(SCIRunFramework* fwk);
  GUIBuilderMap builders;
  Mutex lock;
};

}
