
#ifndef BuilderSericesImpl_h
#define BuilderSericesImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/SSIDL/array.h>

#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;

using std::string;
using SSIDL::array1;

class Registry;

class BuilderServicesImpl : public BuilderServices {

public:
  BuilderServicesImpl();

  ~BuilderServicesImpl();

  // Framework internals
  void init( const Framework::pointer & );

  // export services
  virtual bool connect( const ComponentID::pointer & user,
			const string      & use_port, 
			const ComponentID::pointer & provider, 
			const string      & provide_port);

  virtual bool disconnect( const ComponentID::pointer &, const string &, 
			   const ComponentID::pointer &, const string &);
  virtual bool exportAs( const ComponentID::pointer &, const string &, const string &);
  virtual bool provideTo( const ComponentID::pointer &, const string&, const string &);
  
protected:
  Framework::pointer framework_;
  Registry *registry_;
};

} // end namespace sci_cca

#endif 

