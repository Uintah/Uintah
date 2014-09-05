/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

//using namespace std;
using namespace SCIRun;

int main(int argc, char* argv[])
{
  try {
    // TODO: Move this out of here???
    PIDL::initialize();
  } catch(const Exception& e) {
    std::cerr << "Caught exception:" << std::endl;
    std::cerr << e.message() << std::endl;
    abort();
  } catch(...) {
    std::cerr << "Caught unexpected exception!" << std::endl;
    abort();
  }

  // Get the framework
  try {
    std::string client_url = argv[1];
    std::cerr << "got url: " << client_url << std::endl;
    Object::pointer obj=PIDL::objectFrom(client_url);
    std::cerr << "got obj" << std::endl;
    sci::cca::AbstractFramework::pointer sr =
	pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
    std::cerr << "got sr" << std::endl;
    // test framwork URL availability
    std::cerr << "got sr url: " << sr->getURL().getString() << std::endl;
    // test framwork URL availability
    sci::cca::Services::pointer bs =
	sr->getServices("external builder", "builder main", sci::cca::TypeMap::pointer(0));
    std::cerr << "got bs" << std::endl;
    Builder* builder = new Builder();
    std::cerr << "created builder" << std::endl;
    builder->setServices(bs);
    std::cerr << "called setservices" << std::endl;
    PIDL::serveObjects();
  } catch(const MalformedURL& e) {
    std::cerr << "builder.cc: Caught MalformedURL exception:" << std::endl;
    std::cerr << e.message() << std::endl;
  } catch(const Exception& e) {
    std::cerr << "builder.cc: Caught exception:\n";
    std::cerr << e.message() << std::endl;
    abort();
  } catch(...) {
    std::cerr << "Caught unexpected exception!" << std::endl;
    abort();
  }
}
