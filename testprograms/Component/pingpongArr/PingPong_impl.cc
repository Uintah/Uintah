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
 *  PingPong_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <testprograms/Component/pingpongArr/PingPong_impl.h>
#include <Core/Util/NotFinished.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

using namespace PingPong_ns;
using namespace std;

PingPong_impl::PingPong_impl()
{
}

PingPong_impl::~PingPong_impl()
{
}

int PingPong_impl::pingpong(const SSIDL::array1<int>& arg)
{
  int sum = 0;
  for(unsigned int i = 0; i < arg.size() ; i++)
    sum += arg[i];
  return sum;
}
