
#ifndef RegistrySericesImpl_h
#define RegistrySericesImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/CIA/array.h>

#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;

using std::string;
using CIA::array1;

class Registry;

class RegistryServicesImpl : public RegistryServices {

public:
  RegistryServicesImpl();

  ~RegistryServicesImpl();

  // Framework internals
  void init( const Framework::pointer & );

  // export services
  virtual void getActiveComponentList( array1<ComponentID::pointer> & componentIds );

  // Tells the framework to shut itself down... here for lack of better
  // place to put it currently...
  virtual void shutdown();
  
protected:
  Framework::pointer framework_;
  Registry *registry_;
};

} // end namespace sci_cca

#endif 

