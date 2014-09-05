
#ifndef TestPortImpl_h
#define TestPortImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/PortImpl.h>

namespace sci_cca {

using std::string;
using CIA::array1;

class TestPortImpl : public TestPort_interface {
public:
  TestPortImpl();
  ~TestPortImpl();

  void test();
};

} // namespace sci_cca

#endif 

