
#ifndef ProviderImpl_h
#define ProviderImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <testprograms/Component/framework/TestPortImpl.h>

namespace sci_cca {

class ProviderImpl : virtual public Provider_interface, 
		     virtual public ComponentImpl 
{
private:
  TestPort test_port_;

public:
  ProviderImpl();
  ~ProviderImpl();

  virtual void setServices( const Services &);
  virtual ComponentID getComponentID() { return services_->getComponentID(); }

};

} // namespace sci_cca

#endif 

