// GUIService.h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>

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
  void updateProgress();

private:
    GUIService(SCIRunFramework* fwk);
    GUIBuilderMap builders;
};

}
