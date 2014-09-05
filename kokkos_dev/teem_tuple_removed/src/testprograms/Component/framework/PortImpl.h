
#ifndef PortImpl_h
#define PortImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/SSIDL/array.h>

#include <Core/Exceptions/InternalError.h>

#include <string>
#include <map>

namespace sci_cca {

using std::string;
using SSIDL::array1;

class PortImpl : public Port {
public:
  PortImpl();
  ~PortImpl();

};

} // namespace sci_cca

#endif 

