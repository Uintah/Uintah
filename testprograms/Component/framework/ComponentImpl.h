
#ifndef ComponentImpl_h
#define ComponentImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <Core/Thread/Mutex.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/Exceptions/InternalError.h>

namespace sci_cca {

using SCIRun::InternalError;
using std::string;

class ComponentImpl : virtual public Component_interface {
protected:
  SciServices services_;

public:
  ComponentImpl();
  ~ComponentImpl();

  virtual void setServices( const Services &s );
};

} // namespace sci_cca

#endif 

