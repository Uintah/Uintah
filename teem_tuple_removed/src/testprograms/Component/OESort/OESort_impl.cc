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
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <iostream>

using namespace OESort_ns;
using namespace std;

OESort_impl::OESort_impl()
{
}

OESort_impl::~OESort_impl()
{
}

int OESort_impl::sort(const SSIDL::array1<int>& arr, SSIDL::array1<int>& odds, SSIDL::array1<int>& evens)
{
  odds = arr;
  std::sort(odds.begin(), odds.end());
  evens = odds;
  return 0;
}

OESplit_impl::OESplit_impl()
{
}

OESplit_impl::~OESplit_impl()
{
}

int OESplit_impl::split(const SSIDL::array1<int>& arr, SSIDL::array1<int>& result_arr)
{
  SSIDL::array1<int> odds;
  SSIDL::array1<int> evens;
std::cerr << "BEFORE THE CALL\n";
  ss->sort(arr,odds,evens);
std::cerr << "I JUST CALLED TO SAY...\n";

  int totalsize = arr.size();
  int halfsize = odds.size();
  int arri = 0;
  int oddi = 0;
  int eveni = 0;

  result_arr.resize(totalsize);
 
  if(evens.size() != (unsigned int)halfsize)
    cerr << "ERROR: OESplit_impl::split -- ODDS and EVENS are not of the same size\n";
  
  while (oddi < halfsize) {
    if (eveni < halfsize) {
      if (odds[oddi] < evens[eveni]) {
	result_arr[arri] = odds[oddi];
	oddi++;
      }
      else {
	result_arr[arri] = evens[eveni];
	eveni++;
      }
    }
    else {
      result_arr[arri] = odds[oddi];
      oddi++;
    }
    arri++;
  }
  while (eveni < halfsize) {
    result_arr[arri] = evens[eveni];
    arri++;
    eveni++;
  }

  return 0;
  
}







