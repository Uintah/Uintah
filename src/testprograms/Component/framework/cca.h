

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

  static bool initialized_;
  static Framework framework_;
  static string framework_url_;
  static Thread *framework_thread_;
  static bool is_server_;
  static Component local_framework_;
  static string hostname_;
  static string program_;
  static Semaphore semaphore_;

private:
  CCA(); // you can not allocate a CCA

public:
  static bool init( int &argc, char *argv[] );
  static bool init ( Component &);
  static void done();

  friend class Server;
};



} // namespace sci_cca

#endif // CCA_h
