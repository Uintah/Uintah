// GUIService.cc

#include <SCIRun/Internal/GUIService.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>

namespace SCIRun {


GUIService::GUIService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:GUIService")
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
  GUIBuilderMap::iterator iter = builders.find(builderName);
  if (iter != builders.end()) {
    // TODO: need exception!
    return;
  }
  builders[builderName] = builder;
}

void GUIService::removeBuilder(const std::string& builderName)
{
  GUIBuilderMap::iterator iter = builders.find(builderName);
  if (iter != builders.end()) {
    builders.erase(iter);
    iter->second->deleteReference();
  }
}

void GUIService::updateProgress(const sci::cca::ComponentID::pointer& cid, int progressPercent)
{
  for (GUIBuilderMap::iterator iter = builders.begin(); iter != builders.end(); iter++) {
    iter->second->updateProgress(cid, progressPercent);
  }
}

}
