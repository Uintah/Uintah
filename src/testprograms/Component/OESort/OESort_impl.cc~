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
 *  OESort_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <testprograms/Component/OESort/OESort_impl.h>
#include <Core/Util/NotFinished.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

using namespace OESort_ns;
using namespace std;

OESort_impl::OESort_impl()
{
}

OESort_impl::~OESort_impl()
{
}

int OESort_impl::sort(const CIA::array1<int>& arr1, const CIA::array1<int>& arr2, CIA::array1<int>& aft)
{

  unsigned int l = 0;
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int total_sz = arr1.size() + arr2.size() - 1;
  aft.resize(total_sz);

  /*Merge the array:*/
  while (i < arr1.size()) {
    if (j < arr2.size()) {
      if (arr1[i] < arr2[j]) {
	aft[l] = arr1[i];
	i++;
      }
      else {
	aft[l] = arr2[j];
	j++;
      }
    }
    else {
      aft[l] = arr1[i];
      i++;
    }
    l++;
  }
  while (j < arr2.size()) {
    aft[l] = arr2[j];
    l++;
    j++;
  }

  return 0;
}
