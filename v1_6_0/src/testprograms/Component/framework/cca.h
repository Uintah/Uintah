

#ifndef CCA_h
#define CCA_h

#include <Core/Thread/Thread.h>
#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

#include <string>

namespace SCIRun {
class Thread;
}

namespace sci_cca {

using namespace SCIRun;

using std::string;

class Server;

class CCA {

public:
  static bool init( int &argc, char *argv[] );
  static bool init ( Component::pointer& component, 
		     const string& component_name = "" );
  static void done();

private:
  static bool       initialized_;
  static Framework::pointer  framework_;
  static string     framework_url_;
  static Thread   * framework_thread_;
  static bool       is_server_;
  static Component::pointer  local_framework_;
  static string     hostname_;
  static string     program_;
  static Semaphore  semaphore_;

  CCA(); // you can not allocate a CCA

  friend class Server;
};



} // namespace sci_cca

#endif // CCA_h
