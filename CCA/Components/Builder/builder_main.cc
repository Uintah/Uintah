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
 *  builder.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/Builder/Builder.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

int main(int argc, char* argv[])
{
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

  // Get the framework
  try {
    string client_url = argv[1];
    cerr << "got url: " << client_url << '\n';
    Object::pointer obj=PIDL::objectFrom(client_url);
    cerr << "got obj\n";
    sci::cca::AbstractFramework::pointer sr=pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
    cerr << "got sr\n";
    sci::cca::Services::pointer bs = sr->getServices("external builder", "builder main", sci::cca::TypeMap::pointer(0));
    cerr << "got bs\n";
    Builder* builder = new Builder();
    cerr << "created builder\n";
    builder->setServices(bs);
    cerr << "called setservices\n";
    PIDL::serveObjects();
  } catch(const MalformedURL& e) {
    cerr << "builder.cc: Caught MalformedURL exception:\n";
    cerr << e.message() << '\n';
  } catch(const Exception& e) {
    cerr << "builder.cc: Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }
}
