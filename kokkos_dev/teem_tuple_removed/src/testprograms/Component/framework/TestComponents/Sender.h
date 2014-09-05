
#ifndef Sender_h
#define Sender_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>

namespace sci_cca {

class Sender : virtual public ComponentImpl {

public:
  Sender();
  ~Sender();

  virtual void setServices( const Services::pointer &);

  ComponentID::pointer getComponentID() { return services_->getComponentID(); }
  void        go();

};

} // namespace sci_cca

#endif 

