
#ifndef ServicesImpl_h
#define ServicesImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/SSIDL/array.h>

#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;

using std::string;
using SSIDL::array1;

class ServicesImpl : public Services_interface {

public:
  ServicesImpl();

  ~ServicesImpl();

  // From the "Services" Interface:
  
  virtual Port getPort( const string & name );
  virtual PortInfo createPortInfo( const string & name,
			           const string & type,
				   const array1<string> & properties );
  virtual void registerUsesPort( const PortInfo & nameAndType );
  virtual void unregisterUsesPort( const string & name );
  virtual void addProvidesPort( const Port & inPort,
				const PortInfo & name );
  virtual void removeProvidesPort( const string & name );
  virtual void releasePort( const string & name );
  virtual ComponentID getComponentID();

  void init( const Framework &f, const ComponentID & id) { framework_ = f; id_ = id; }
  void shutdown() { framework_->unregisterComponent( id_); id_ = 0;}
protected:

  Framework framework_;
  ComponentID id_;
};

} // end namespace sci_cca

#endif 

