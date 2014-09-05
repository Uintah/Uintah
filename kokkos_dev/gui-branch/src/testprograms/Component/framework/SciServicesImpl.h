
#ifndef SciSericesImpl_h
#define SciSericesImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/CIA/array.h>

#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;

using std::string;
using CIA::array1;

class SciServicesImpl : public SciServices_interface {

public:
  SciServicesImpl();

  ~SciServicesImpl();

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

  void init( const Framework &f, const ComponentID & id);
  void done();

  void shutdown();

protected:

  Framework framework_;
  ComponentID id_;
};

} // end namespace sci_cca

#endif 

