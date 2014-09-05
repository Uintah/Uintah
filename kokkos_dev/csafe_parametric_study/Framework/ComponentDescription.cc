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
 *  ComponentDescription.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <sci_defs/framework_defs.h>
#include <Framework/ComponentDescription.h>
#include <iostream> // test

namespace SCIRun {

ComponentDescription::ComponentDescription(ComponentModel* model, const std::string& type, const std::string& libName, const std::string& loader)
  : model(model), type(type), library(libName), loaderName(loader)
{
  if (! library.empty()) {
    library = "lib" + library + "." + SO_OR_ARCHIVE_EXTENSION;
  }
}

ComponentDescription::~ComponentDescription()
{
}

std::string
ComponentDescription::getLoaderName() const
{
  //for the local loaders, the loader name is "";
  return "";
}

std::string
ComponentDescription::getLibrary() const
{
  return library;
}

} // end namespace SCIRun
