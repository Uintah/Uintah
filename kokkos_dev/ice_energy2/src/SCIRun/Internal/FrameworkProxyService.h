#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>

namespace SCIRun {

class SCIRunFramework;

/**
 * \class FrameworkProxyService
 *
 */
class FrameworkProxyService : public sci::cca::ports::FrameworkProxyService,
                              public InternalFrameworkServiceInstance {
public:
    virtual ~FrameworkProxyService();

    /** Factory method for creating an instance of a FrameworkProxyService class.
        Returns a reference counted pointer to a newly-allocated BuilderService
        port.  The \em framework parameter is a pointer to the relevent framework
        and the \em name parameter will become the unique name for the new port.*/
    static InternalFrameworkServiceInstance* create(SCIRunFramework* framework);

    virtual sci::cca::ComponentID::pointer
    createInstance(const std::string& instanceName,
                   const std::string& className,
                   const sci::cca::TypeMap::pointer &properties);

    /** */
    int addComponentClasses(const std::string &loaderName);

    /** */
    int removeComponentClasses(const std::string &loaderName);

    /** */
    virtual int
    addLoader(const std::string &loaderName, const std::string &user,
                const std::string &domain, const std::string &loaderPath);

    /** */
    virtual int removeLoader(const std::string &name);
  
    /** */
    virtual sci::cca::Port::pointer
    getService(const std::string &);

  //virtual void registerFramework(const std::string &frameworkURL); 
  //virtual void registerServices(const sci::cca::Services::pointer &svc);

private:
    FrameworkProxyService(SCIRunFramework* fwk);

    // used by registerFramework & registerServices methods
    //std::vector<sci::cca::Services::pointer> servicesList;  
};

}
