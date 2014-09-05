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
 *  stopserver.cc
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2004
 *
 *  Copyright (C) 1999 U of U
 */

/////////////////////////////////////////////////
// This program is used to stop a running server
// test program. For example, if "pingpong -server"
// is running and gives URL "socket://buzz.sci.utah.edu:34042/134585728",
// then "stopserver socket://buzz.sci.utah.edu:34042/134585728"
// is supposed to make pingpong server quit.

#include <iostream>
#include <string>
#include <vector>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <Core/Thread/Time.h>
#include <Core/CCA/SSIDL/sidl_sidl.h>

using namespace std;

using namespace SCIRun;

void usage(char* progname)
{
    cerr << "usage: " << progname << "server_URL1 server_URL2 ... \n";
    exit(1);
}

int main(int argc, char* argv[])
{
  
  vector <URL> urls;
  try {
    PIDL::initialize();
      
    //if URLs are not provided, show usage and quit
    if(argc==1) usage(argv[0]);

    for(int i=1;i<argc;i++){		
      string url_arg(argv[i]);
      urls.push_back(url_arg);
    }
      
    for(unsigned int i=0; i<urls.size();i++){
      Object::pointer obj=PIDL::objectFrom(urls[i]);
      SSIDL::BaseInterface::pointer bi=  pidl_cast<SSIDL::BaseInterface::pointer>(obj);
      bi->deleteReference();
    }

  } catch(const MalformedURL& e) {
    cerr << "stopserver.cc: Caught MalformedURL exception:\n";
    cerr << e.message() << '\n';
  } catch(const Exception& e) {
    cerr << "stopserver.cc: Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }
  PIDL::serveObjects();
  PIDL::finalize();
  return 0;
}

