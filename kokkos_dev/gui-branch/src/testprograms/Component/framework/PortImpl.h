
#ifndef PortImpl_h
#define PortImpl_h

#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/CIA/array.h>

#include <Core/Exceptions/InternalError.h>

#include <string>
#include <map>

namespace sci_cca {

using std::string;
using CIA::array1;

class PortImpl : public Port_interface {
public:
  PortImpl();
  ~PortImpl();

};

} // namespace sci_cca

#endif 

