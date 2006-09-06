// GUIService.cc

#include <SCIRun/Internal/GUIService.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <Core/Thread/Guard.h>

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
#if DEBUG
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

}
