
#ifndef ComponentIdImpl_h
#define ComponentIdImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <Core/Thread/Mutex.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/Exceptions/InternalError.h>

#include <string>

namespace sci_cca {

using SCIRun::InternalError;
using std::string;

class ComponentIdImpl : public ComponentID {
public:
  ComponentIdImpl();
  ~ComponentIdImpl();

  virtual string toString();

  virtual void init( const string &, const string & );

  string fullString();

private:
  string host_;
  string program_;
  int number_;
  string id_;
};

} // namespace sci_cca

#endif 

