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
 *  BaseInterface: Implementation of SSIDL.BaseInterface for PIDL
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
#include <Core/Util/NotFinished.h>

//using SSIDL::Class;
//using SSIDL::Object;
using SSIDL::BaseInterface;
using SSIDL::BaseClass;
using SSIDL::ClassInfo;

// void BaseInterface::addReference()
// {
//   SCIRun::Object::addReference();
// }

void BaseInterface::addRef()
{
    SCIRun::Object::addReference();
}

// void BaseInterface::deleteReference()
// {
//   SCIRun::Object::deleteReference();
// }

void BaseInterface::deleteRef()
{
    SCIRun::Object::deleteReference();
}

/*
Class::pointer BaseInterface::getClass()
{
  NOT_FINISHED(".SSIDL.Class .SSIDL.BaseInterface.getClass()");
  return Class::pointer(0);
}
*/
bool BaseInterface::isSame(const BaseInterface::pointer& /*object*/)
{
    NOT_FINISHED("bool .SSIDL.BaseInterface.isSame(in .SSIDL.BaseInterface object)");
    return false;
}

bool BaseInterface::isType(const std::string& /*name*/)
{
    NOT_FINISHED("bool .SSIDL.BaseInterface.isType(in .SSIDL.string name)");
    return false;
}

BaseInterface::pointer BaseInterface::queryInt(const std::string& /*name*/)
{
    NOT_FINISHED(".SSIDL.BaseInterface .SSIDL.BaseInterface.queryInt(in .SSIDL.string name)");
    return BaseInterface::pointer(0);
}

SSIDL::ClassInfo BaseInterface::getClassInfo()
{
    NOT_FINISHED("ClassInfo .SSIDL.getClassInfo()");
    return ClassInfo::pointer(0);
}

/*
bool BaseInterface::isInstanceOf(const Class::pointer& type)
{
  NOT_FINISHED("bool .SSIDL.BaseInterface.isInstanceOf(in .SSIDL.Class type)");
  return false;
}

bool BaseInterface::supportsInterface(const Class::pointer& type)
{
  NOT_FINISHED("bool .SSIDL.BaseInterface.supportsInterface(in .SSIDL.Class type)");
  return false;
}
*/
// *** replaced by BaseInterface::queryInt
//
// BaseInterface::pointer BaseInterface::queryInterface(const std::string& /*type*/)
// {
//   NOT_FINISHED(".SSIDL.BaseInterface .SSIDL.BaseInterface.queryInterface(in .SSIDL.Class type)");
//   return BaseInterface::pointer(0);
// }

