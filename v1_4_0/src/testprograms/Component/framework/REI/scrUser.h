
#ifndef SCRUSER_H
#define SCRUSER_H

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <testprograms/Component/framework/REI/scrInterfaceImpl.h>

namespace sci_cca {

class scrUser : virtual public ComponentImpl 
{
private:

  scrInterfaceImpl * scr_port_;

public:
  scrUser() {}
  ~scrUser() {}

  virtual void setServices( const Services::pointer & srv );

  void go();

};

} // namespace sci_cca

#endif 

