


#include <sys/utsname.h> 
#include <iostream>
#include <string>

#include <Core/Containers/StringUtil.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>

#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/FrameworkImpl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <testprograms/Component/framework/LocalFramework.h>

namespace sci_cca {

using namespace SCIRun;
using namespace std;

class Server : public Runnable {
public:
  Server() {}

  void run() { PIDL::serveObjects(); CCA::semaphore_.up(); }
};


bool CCA::initialized_ = false;
Framework::pointer CCA::framework_;
Thread *CCA::framework_thread_;
bool CCA::is_server_;
Component::pointer CCA::local_framework_;

string CCA::framework_url_;
string CCA::hostname_;
string CCA::program_;
Semaphore CCA::semaphore_("CCA", 0 );

FrameworkImpl *f;

CCA::CCA() 
{
}

bool
CCA::init( int &argc, char *argv[] )
{
  if ( initialized_ ) return false; // error: init can be called only once.

  is_server_ = true;

  // get general info
  
  struct utsname rec;
  uname( &rec );
  hostname_ = rec.nodename;

  program_ = basename( string(argv[0]) );

  // parse args
  int last = 1;
  int num = argc;
 
  for (int i=1; i<num; i++ ) {
    string arg(argv[i]);

    if ( arg == "-server" ) {
      if ( ++i >= argc ) {
	cerr << "missing server url" << endl;
	return 0;
      }
      framework_url_ = string(argv[i]);
      argc -= 2;
      is_server_ = false;
    } else { // not a cca option
      argv[last] = argv[i];
    }
  }

  // init framework

  try {

    PIDL::initialize();
    
    if ( is_server_ ) {
      // start server

      framework_ = Framework::pointer(new FrameworkImpl);

      cerr << "server = " << framework_->getURL().getString() << endl;
      framework_thread_ = new Thread( new Server,"cca server thread" );
      framework_thread_->setDaemon();
      //      framework_thread_->detach();
    }
    else {
      // connect to server
      framework_ = pidl_cast<Framework::pointer>(PIDL::objectFrom(framework_url_));
    }
  } catch (const Exception &e ) {
    cerr << "cca_init caught exception `" << e.message() << "'" << endl;
    return false;
  } catch (...) { 
    cerr << "cca_init caught unknown exception " << endl;
    return false;
  }

  // framework's local agent 
//   local_framework_ = new LocalFramework;
//   if ( !framework_->registerComponent( hostname_, program_, local_framework_ ))
//       return false;


  initialized_ = true;
  return true;
}


bool
CCA::init( Component::pointer & component, const string & component_name /* = "" */ )
{
  if ( !initialized_ )
    return false;

  string name = component_name;
  if( name == "" ) name = program_;

  // user's component
  return framework_->registerComponent( hostname_, name, component );
}

void
CCA::done()
{
  if ( is_server_ )
    {
      local_framework_ = Component::pointer(0);
      framework_ = Framework::pointer(0);
      semaphore_.down();
    }
}

} // namespace sci_cca
