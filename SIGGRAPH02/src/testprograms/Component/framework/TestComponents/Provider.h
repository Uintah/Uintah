
#ifndef Provider_h
#define Provider_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <testprograms/Component/framework/TestPortImpl.h>

namespace sci_cca {

class Provider : virtual public ComponentImpl 
{
private:
  TestPort::pointer test_port_;

public:
  Provider();
  ~Provider();

  virtual void setServices( const Services::pointer &);

  ComponentID::pointer getComponentID() { return services_->getComponentID(); }

};

} // namespace sci_cca

#endif 

