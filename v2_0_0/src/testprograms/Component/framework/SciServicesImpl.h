
#ifndef SciSericesImpl_h
#define SciSericesImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/SSIDL/array.h>

#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;

using std::string;
using SSIDL::array1;

class SciServicesImpl : public SciServices {

public:
  SciServicesImpl();

  ~SciServicesImpl();

  // From the "Services" Interface:
  
  virtual Port::pointer getPort( const string & name );
  virtual PortInfo::pointer createPortInfo( const string & name,
					    const string & type,
					    const array1<string> & properties );
  virtual void registerUsesPort( const PortInfo::pointer & nameAndType );
  virtual void unregisterUsesPort( const string & name );
  virtual void addProvidesPort( const Port::pointer & inPort,
				const PortInfo::pointer & name );
  virtual void removeProvidesPort( const string & name );
  virtual void releasePort( const string & name );
  virtual ComponentID::pointer getComponentID();

  void init( const Framework::pointer &f, const ComponentID::pointer & id);
  void done();

  void shutdown();

protected:

  Framework::pointer framework_;
  ComponentID::pointer id_;
};

} // end namespace sci_cca

#endif 

