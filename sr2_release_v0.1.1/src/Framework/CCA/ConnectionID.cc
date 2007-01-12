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
 *  ComponentID.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */
#include <Framework/CCA/ConnectionID.h>
#include <Framework/CCA/ComponentID.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/TypeMap.h>
#include <Core/CCA/PIDL/URL.h>

#include <iostream>

namespace SCIRun {

const std::string ConnectionID::IS_IN_USE("cca.isInUse");

ConnectionID::ConnectionID(const sci::cca::ComponentID::pointer& user,
                           const std::string& uPortName,
                           const sci::cca::ComponentID::pointer& provider,
                           const std::string& pPortName)
  : userPortName(uPortName), providerPortName(pPortName), user(user), provider(provider), properties(new TypeMap)
{
  properties->putBool(IS_IN_USE, false);
}

ConnectionID::~ConnectionID()
{
}

sci::cca::ComponentID::pointer ConnectionID::getProvider()
{
  return provider;
}

sci::cca::ComponentID::pointer ConnectionID::getUser()
{
  return user;
}

std::string ConnectionID::getProviderPortName()
{
  return providerPortName;
}

std::string ConnectionID::getUserPortName()
{
  return userPortName;
}

void ConnectionID::setProperties(const sci::cca::TypeMap::pointer &tm)
{
  bool inUse = properties->getBool(IS_IN_USE, false);
  properties = tm;
  properties->putBool(IS_IN_USE, inUse);
}

} // end namespace SCIRun
