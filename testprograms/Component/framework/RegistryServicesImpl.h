
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

class RegistryServicesImpl : public RegistryServices_interface {

public:
  RegistryServicesImpl();

  ~RegistryServicesImpl();

  // Framework internals
  void init( const Framework & );

  // export services
  virtual void getActiveComponentList( array1<string> & components );
  
protected:
  Framework framework_;
  Registry *registry_;
};

} // end namespace sci_cca

#endif 

