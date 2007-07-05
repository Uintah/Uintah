#ifndef __user_interface_h_
#define __user_interface_h_

#include <tcl.h>
#include <Thread/Mutex.h>

using namespace SCIRun;

namespace SemotusVisum {

class uiHelper;
class Renderer;

class UserInterface {
public:
  static void initialize( int argc, char ** argv );
  //static void renderer( Renderer * renderer, string type );
  
  static inline Tcl_Interp * getInterp() { return the_interp; }
  static inline uiHelper * getHelper() { return the_helper; }
  static inline Renderer * renderer() { return current_renderer; }
  static inline void renderer( Renderer * r ) { current_renderer = r; }

  static inline void lock() { TclLock.lock(); }
  static inline void unlock() { TclLock.unlock(); }
  
protected:
  UserInterface() {}
  ~UserInterface() {}

  static Tcl_Interp *the_interp;
  static uiHelper *the_helper;
  static Renderer *current_renderer;
  
  static int Tk_AppInit( Tcl_Interp * );
  static Mutex TclLock;
};

}
#endif




