
#ifndef ConnectionSericesImpl_h
#define ConnectionSericesImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/CIA/array.h>

#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;

using std::string;
using CIA::array1;

class ConnectionServicesImpl : public ConnectionServices_interface {

public:
  ConnectionServicesImpl();

  ~ConnectionServicesImpl();

  // Framework internals
  void init( const Framework & );

  // export services
  virtual bool connect( const ComponentID &, const string &, 
			const ComponentID &, const string &);
  virtual bool disconnect( const ComponentID &, const string &, 
			   const ComponentID &, const string &);
  virtual bool exportAs( const ComponentID &, const string &, const string &);
  virtual bool provideTo( const ComponentID &, const string&, const string &);
  
protected:
  Framework framework_;
};

} // end namespace sci_cca

#endif 

