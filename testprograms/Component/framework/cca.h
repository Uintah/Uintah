

#ifndef CCA_h
#define CCA_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

namespace SCIRun {
class Thread;
}

namespace sci_cca {

using namespace SCIRun;

class CCA {

  static bool initialized_;
  static Framework framework_;
  static string framework_url_;
  static Thread *framework_thread_;
  static bool is_server_;
  static Component local_framework_;
  static string hostname_;
  static string program_;

private:
  CCA(); // you can not allocate a CCA

public:
  static bool init( int &argc, char *argv[]);
  static bool init ( Component &);
};



} // namespace sci_cca

#endif // CCA_h
