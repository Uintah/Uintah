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
 *  Object: Implementation of SSIDL.Object for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/SSIDL/sidl_sidl.h>

//using SSIDL::Class;
using SSIDL::BaseInterface;
using SSIDL::Object;

/*
 * These are all implemented in SSIDL.interface, so these are just
 * up calls, since they will get generated from cia spec.
 */

void Object::addReference()
{
  BaseInterface::addReference();
}

void Object::deleteReference()
{
  BaseInterface::deleteReference();
}

/*Class::pointer Object::getClass()
{
  return  BaseInterface::getClass();
}*/

bool Object::isSame(const BaseInterface::pointer& i)
{
  return BaseInterface::isSame(i);
}
/*
bool Object::isInstanceOf(const Class::pointer& c)
{
  return BaseInterface::isInstanceOf(c);
}

bool Object::supportsInterface(const Class::pointer& c)
{
  return BaseInterface::supportsInterface(c);
}*/
/*
BaseInterface::pointer Object::queryInterface(const Class::pointer& c)
{
  return BaseInterface::queryInterface(c);
}
*/

BaseInterface::pointer Object::queryInterface(const std::string& c)
{
  return BaseInterface::queryInterface(c);
}

