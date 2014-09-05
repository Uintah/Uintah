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
 *  ComponentID.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/SCIRunFramework.h>
#include <Core/CCA/PIDL/URL.h>
#include <SCIRun/CCA/ConnectionID.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

ConnectionID::ConnectionID(const sci::cca::ComponentID::pointer& user,
			   const string& userPortName,
			   const sci::cca::ComponentID::pointer& provider,
			   const string& providerPortName)
{
  this->user=user;
  this->userPortName=userPortName;
  this->provider=provider;
  this->providerPortName=providerPortName;
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



