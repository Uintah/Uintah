
#ifndef SCR_H
#define SCR_H

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <testprograms/Component/framework/REI/scrInterfaceImpl.h>

namespace sci_cca {

class scr : virtual public ComponentImpl 
{
private:

  scrInterfaceImpl * scr_port_;

public:
  scr() {}
  ~scr() {}

  virtual void setServices( const Services::pointer & srv );

};

} // namespace sci_cca

#endif 

