/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  newmain.cc: CCA-ified version of SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/SCIRunFramework.h>
#include <iostream>
using namespace std;
using namespace SCIRun;
using namespace gov::cca;
#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml
#include <sys/stat.h>

void
usage()
{
  cout << "Usage: scirun [args] [net_file]\n";
  cout << "       [-]-v[ersion] : prints out version information\n";
  cout << "       [-]-h[elp]    : prints usage information\n";
  cout << "       net_file      : SCIRun Network Input File\n";
  exit( 0 );
}

// Apparently some args are passed through to TCL where they are parsed...
// Probably need to check to make sure they are at least valid here???
void
parse_args( int argc, char *argv[] )
{
  for( int cnt = 0; cnt < argc; cnt++ )
    {
      string arg( argv[ cnt ] );
      if( ( arg == "--version" ) || ( arg == "-version" )
	  || ( arg == "-v" ) || ( arg == "--v" ) ){
	cout << "Version: " << VERSION << "\n";
	exit( 0 );
      } else if ( ( arg == "--help" ) || ( arg == "-help" ) ||
		  ( arg == "-h" ) ||  ( arg == "--h" ) ) {
	usage();
      } else {
	  struct stat buf;
	  if (stat(arg.c_str(),&buf) < 0) {
	      cerr << "Couldn't find net file " << arg
		   << ".\nNo such file or directory.  Exiting." << endl;
	      exit(0);
	  }
      }
    }
}

int
main(int argc, char *argv[] )
{
  bool gui=true;
  bool framework=true;
  parse_args( argc, argv );

  try {
    // TODO: Move this out of here???
    PIDL::PIDL::initialize(argc, argv);
  } catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }

  // Create a new framework
  try {
    Framework sr;
    if(framework){
      sr = new SCIRunFramework();
      cerr << "URL to framework:\n" << sr->getURL().getString() << '\n';
    } else {
      cerr << "Not finished: pass url to existing framework\n";
    }

    gov::cca::Services main_services = sr->createServices("SCIRun main");
    gov::cca::BuilderService builder = pidl_cast<gov::cca::BuilderService>(main_services->getPort("cca.builderService"));
    if(!builder){
      cerr << "Fatal Error: Cannot find builder service\n";
      Thread::exitAll(1);
    }

    if(gui){
      ComponentID gui_id=builder->createComponentInstance("gui", "cca:SCIRun.Builder");
      if(!gui_id){
	cerr << "Cannot create GUI component\n";
	Thread::exitAll(1);
      }
    }
    main_services->releasePort("cca.builderService");
    cerr << "SCIRun " << VERSION << " started...\n";
    PIDL::PIDL::serveObjects();
    cerr << "serveObjects done!\n";
  } catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }
  return 0;
}
