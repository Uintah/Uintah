
#ifndef PortInfoImpl_h
#define PortInfoImpl_h

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

class PortInfoImpl : public PortInfo {
public:
  PortInfoImpl( const string & name,
		const string & type,
		const array1<string> & properties);
  ~PortInfoImpl();

  virtual  string  getType();
  virtual  string  getName();
  virtual  string  getProperty( const string & name );

private:
  string name_;
  string type_;
  array1<string> properties_;
};

} // namespace sci_cca

#endif 

