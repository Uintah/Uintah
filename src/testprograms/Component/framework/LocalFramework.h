

#ifndef LocalFramework_h
#define LocalFramework_h

#include <map>
#include <testprograms/Component/framework/cca_sidl.h>

namespace sci_cca {

class LocalFramework : public Component_interface {
private:
  Services services_;

public:
  LocalFramework();
  virtual ~LocalFramework();

  // from Component
  virtual void setServices( const Services &service );
};

} // namespace sci_cca

#endif // LocalFramework_h
