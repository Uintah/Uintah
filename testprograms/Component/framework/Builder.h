
#ifndef BuilderImpl_h
#define BuilderImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>

namespace sci_cca {

class Builder : virtual public ComponentImpl {

public:
  Builder();
  ~Builder();

  virtual void setServices( const Services &);

};

} // namespace sci_cca

#endif 

