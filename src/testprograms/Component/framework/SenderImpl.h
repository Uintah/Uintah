
#ifndef SenderImpl_h
#define SenderImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>

namespace sci_cca {

class SenderImpl : virtual public Sender_interface, virtual public ComponentImpl {

public:
  SenderImpl();
  ~SenderImpl();

  virtual void setServices( const Services &);
  virtual ComponentID getComponentID() { return services_->getComponentID(); }
  virtual void go();

};

} // namespace sci_cca

#endif 

