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


#include <iostream>
#include <fstream>
#include <vector>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Core/CCA/tools/strauss/strauss.h>

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using namespace SCIRun;


bool fileExists(const char* mFile)
{
  struct stat s;

  if( stat( mFile, &s ) != -1 ) {
    return true;
  }
  return false;
}


int main(int argc, char* argv[])
{
 
  string pluginSpec;
  string portSpec;

  if(argc < 3) {
    cerr << "Wrong parameters to strauss.\n";
    cerr << "USAGE: strauss -p plugin.xml portspec.xml\n"; 
    exit(1);
  }


  std::string arg(argv[1]);
  if(arg == "-p") {
    pluginSpec=argv[2];
    if(!fileExists(pluginSpec.c_str())) {
      cerr << "ERROR: " << pluginSpec << "... doesn't seem to exist... bye.\n";
      exit(1);
    }
  } else {
    cerr << "Unknown option: " << argv[1] << endl;
    exit(1);
  }

  portSpec = argv[3];  
  if(!fileExists(portSpec.c_str())) {
    cerr << "ERROR: " << portSpec << "... doesn't seem to exist... bye.\n";
    exit(1);
  }

  //Currently there is no option to specify output files in commandline, maybe add that later
  string header("Bridge.h");
  string impl("Bridge.cc");

  try {
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    cout << "Error during initialization of XERXES! :\n";
    return 1;
  }

  
  Strauss* strauss = new Strauss(pluginSpec,portSpec,header,impl);
  strauss->emitHeader();
  strauss->emitImpl();
  delete strauss;

  return 0;
}

