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
 *  PingThrow_impl.cc: Test client for PIDL
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   August 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <testprograms/Component/exceptiontest/PingThrow_impl.h>
#include <Core/Util/NotFinished.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <mpi.h>

using namespace PingThrow_ns;
using namespace std; 

PingThrow_impl::PingThrow_impl()
{
}

PingThrow_impl::~PingThrow_impl()
{
}

int PingThrow_impl::pingthrow(int arg)
{
  if(arg==1) {
    PPException* ppe = new PPException();
    throw (ppe);
  }
  return arg;
}

int PingThrow_impl::pingthrow2(int arg)
{
  if(arg==1) {
    PP2Exception* ppe2 = new PP2Exception();
    throw (ppe2);
  }
  return arg;
}

void PingThrow_impl::getOX(OtherThrow::pointer& otptr)
{
  otptr = new OtherThrow_impl();
  otptr->addReference();
}

void PingThrow_impl::donone()
{
}

OtherThrow_impl::OtherThrow_impl()
{
}

OtherThrow_impl::~OtherThrow_impl()
{
}

int OtherThrow_impl::otherthrow()
{
  OtherX* ox = new OtherX();
  throw (ox);
  return 0;
}
