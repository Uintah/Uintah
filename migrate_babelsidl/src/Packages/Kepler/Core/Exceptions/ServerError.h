// license

#ifndef Kepler_Core_Exceptions_ServerError_h
#define Kepler_Core_Exceptions_ServerError_h

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Exceptions/share.h>

namespace Kepler {

class ServerError : public SCIRun::Exception {
public:
  ServerError(const std::string&, const char* file, int line);
  ServerError(const ServerError&);
  virtual ~ServerError();

  virtual const char* message() const;
  virtual const char* type() const;

private:
  std::string message_;
  ServerError& operator=(const ServerError&);
};


}


#endif
