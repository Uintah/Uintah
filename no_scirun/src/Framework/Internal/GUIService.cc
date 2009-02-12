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


// GUIService.cc

#include <Framework/Internal/GUIService.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/CCA/CCAException.h>
#include <Core/Thread/Guard.h>
#include <Core/Thread/Thread.h>

namespace SCIRun {

GUIService::GUIService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:GUIService"), lock("GUIService lock")
{
}

sci::cca::ComponentID::pointer
GUIService::createInstance(const std::string& instanceName, const std::string& className, const sci::cca::TypeMap::pointer& properties)
{
  if (instanceName.size()) {
    if (framework->lookupComponent(instanceName) != 0) {
      throw sci::cca::CCAException::pointer(new CCAException("Component instance name " + instanceName + " is not unique"));
    }
    return framework->createComponentInstance(instanceName, className, properties);
  }
  return framework->createComponentInstance(framework->getUniqueName(className), className, properties);
}


InternalFrameworkServiceInstance* GUIService::create(SCIRunFramework* framework)
{
  GUIService* n = new GUIService(framework);
  return n;
}

sci::cca::Port::pointer
GUIService::getService(const std::string&)
{
  return sci::cca::Port::pointer(this);
}

void GUIService::addBuilder(const std::string& builderName, const sci::cca::GUIBuilder::pointer& builder)
{
  Guard g(&lock);
#if FWK_DEBUG
  std::cerr << "GUIService::AddBuilder(..): from thread " << Thread::self()->getThreadName() << std::endl;
#endif
  GUIBuilderMap::iterator iter = builders.find(builderName);
  if (iter != builders.end()) {
    // TODO: need exception!
    return;
  }
  builders[builderName] = builder;
}

void GUIService::removeBuilder(const std::string& builderName)
{
  Guard g(&lock);
  GUIBuilderMap::iterator iter = builders.find(builderName);
  if (iter != builders.end()) {
    builders.erase(iter);
    iter->second->deleteReference();
  }
}

// TODO: update SIDL with throws clause
void GUIService::updateProgress(const sci::cca::ComponentID::pointer& cid, int progressPercent)
{
  Guard g(&lock);
  if (progressPercent > 100) {
    throw sci::cca::CCAException::pointer(new CCAException("Progress percent cannot be > 100"));
  } else if (progressPercent < 0) {
    throw sci::cca::CCAException::pointer(new CCAException("Progress percent cannot be < 0"));
  }
  for (GUIBuilderMap::iterator iter = builders.begin(); iter != builders.end(); iter++) {
    iter->second->updateProgress(cid, progressPercent);
  }
}

void GUIService::updateComponentModels()
{
  Guard g(&lock);
  //std::cerr << "GUIService::updateComponentModels() from thread " << Thread::self()->getThreadName() << std::endl;
  for (GUIBuilderMap::iterator iter = builders.begin(); iter != builders.end(); iter++) {
    iter->second->updateComponentModels();
  }
}

}
