#ifndef Services_h
#define Services_h

#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;

class Module;
class IPort;
class OPort;

class Resources;
class ServicesDB;

typedef void* LibHandle;

class Services {
public:
  Services();

  Module *make_module( const string &type, const string &name="" );
  IPort  *make_iport( const string &type, const string &name, Module * );
  OPort  *make_oport( const string &type, const string &name, Module * );

private:
  Resources *resources_;
  ServicesDB *db_;
};

extern Services services;

} // namespace SCIRun

#endif // Services
