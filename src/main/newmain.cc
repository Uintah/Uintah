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

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/SCIRunFramework.h>
#include <iostream>
#include <mpi.h>
using namespace std;
using namespace SCIRun;
using namespace sci::cca;
#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml
#include <sys/stat.h>


string defaultBuilder("qt"); 

void
usage()
{
  cout << "Usage: scirun [args] [net_file]\n";
  cout << "       [-]-v[ersion]          : prints out version information\n";
  cout << "       [-]-h[elp]             : prints usage information\n";
  cout << "       [-]-b[uilder] qt/txt   : selects QT or Textual builder\n";
  cout << "       net_file               : SCIRun Network Input File\n";
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
      } else if ( ( arg == "--builder" ) || ( arg == "-builder" ) ||
		  ( arg == "-b" ) ||  ( arg == "--b" ) ) {
	if(++cnt<argc) defaultBuilder=argv[cnt];
	else{
	  cerr << "Unkown builder."<<endl;
	  usage();
	}
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
  bool framework=true;
  MPI_Init(&argc,&argv);

  parse_args( argc, argv );

  try {
    // TODO: Move this out of here???
    PIDL::initialize();
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
    AbstractFramework::pointer sr;
    if(framework){
      sr = AbstractFramework::pointer(new SCIRunFramework());
      cerr << "URL to framework:\n" << sr->getURL().getString() << '\n';
      //ofstream f("framework.url");
      //std::string s;
      //f<<sr->getURL().getString();
      //f.close();


    } else {
      cerr << "Not finished: pass url to existing framework\n";
    }



    sci::cca::Services::pointer main_services = sr->getServices("SCIRun main", "main", sci::cca::TypeMap::pointer(0));
    sci::cca::ports::BuilderService::pointer builder = pidl_cast<sci::cca::ports::BuilderService::pointer>(main_services->getPort("cca.BuilderService"));
    if(builder.isNull()){
      cerr << "Fatal Error: Cannot find builder service\n";
      Thread::exitAll(1);
    }

    
#   if !defined(HAVE_QT)
      defaultBuilder="txt";
#   endif

    if(defaultBuilder=="qt"){
      ComponentID::pointer gui_id=builder->createInstance("QtBuilder", "cca:SCIRun.Builder", sci::cca::TypeMap::pointer(0));
      if(gui_id.isNull()){
	cerr << "Cannot create component: cca:SCIRun.Builder\n";
	Thread::exitAll(1);
      }
    }
    else{
      ComponentID::pointer gui_id=builder->createInstance("TxtBuilder", "cca:SCIRun.TxtBuilder", sci::cca::TypeMap::pointer(0));
      if(gui_id.isNull()){
	cerr << "Cannot create component: cca:SCIRun.TxtBuilder\n";
	Thread::exitAll(1);
      }
    }
    main_services->releasePort("cca.BuilderService");
    cout << "SCIRun " << VERSION << " started...\n";

    //broadcast, listen to URL periodically
    //sr->share(main_services);

    PIDL::serveObjects();
    cout << "serveObjects done!\n";
    PIDL::finalize();




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
