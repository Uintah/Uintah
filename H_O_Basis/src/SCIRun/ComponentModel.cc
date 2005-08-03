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
 *  ComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/ComponentModel.h>

#include <iostream>
#include <vector>

namespace SCIRun {

ComponentModel::ComponentModel(const std::string& prefixName)
  : prefixName(prefixName)
{
}

ComponentModel::~ComponentModel()
{
}

bool ComponentModel::haveComponent(const std::string& type)
{
  std::cerr << "Error: this component model does not implement haveComponent, name="
            << type << std::endl;
  return false;
}

ComponentInstance*
ComponentModel::createInstance(const std::string& name,
                               const std::string& type,
                               const sci::cca::TypeMap::pointer &tm)
{
  std::cerr << "Error: this component model does not implement createInstance"
            << std::endl;
  return 0;
}

bool ComponentModel::destroyInstance(ComponentInstance* ic)
{
  std::cerr << "Error: this component model does not implement destroyInstance"
            << std::endl;
  return false;
}

std::vector<std::string>
ComponentModel::splitPathString(const std::string &path)
{
    std::vector<std::string> ans;
    if (path == "" ) {
	return ans;
    }

    // Split the PATH string into a list of paths.  Key on ';' token.
    std::string::size_type start = 0;
    std::string::size_type end = path.find(';', start);
    while (end != path.npos) {
	std::string substring = path.substr(start, end - start);
	ans.push_back(substring);
	start = end + 1;
	end = path.find(';', start);
    }
    // grab the remaining path
    std::string substring = path.substr(start, end - start);
    ans.push_back(substring);

    return ans;  
}

} // end namespace SCIRun
