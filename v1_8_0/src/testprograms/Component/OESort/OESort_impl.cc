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
#include <algo.h>
#include <iostream>

using namespace OESort_ns;
using namespace std;

OESort_impl::OESort_impl()
{
}

OESort_impl::~OESort_impl()
{
}

int OESort_impl::sort(CIA::array1<int>& arr)
{
  sort(arr);
  return 0;
}

OESplit_impl::OESplit_impl()
{
}

OESplit_impl::~OESplit_impl()
{
}

int OESplit_impl::split(CIA::array1<int>& arr)
{
  
  //pp->sort(arr);

  CIA::array1<int> Tarr(arr);
  int totalsize = arr.size();
  int halfsize = totalsize / 2;
  int arri;
  int Tarri;
  int Tarrj;

  /*Odd-Even -> Even*/
  arri = 0;
  Tarri = 1;
  Tarrj = halfsize;
  while (Tarri < halfsize) {
    if (Tarrj < totalsize) {
      if (Tarr[Tarri] < Tarr[Tarrj]) {
	arr[arri] = Tarr[Tarri];
	Tarri+=2;
      }
      else {
	arr[arri] = Tarr[Tarrj];
	Tarrj+=2;
      }
    }
    else {
      arr[arri] = Tarr[Tarri];
      Tarri+=2;
    }
    arri+=2;
  }
  while (Tarrj < totalsize) {
    arr[arri] = Tarr[Tarrj];
    arri+=2;
    Tarrj+=2;
  }

  /*Even-Odd -> Odd*/
  arri = 1;
  Tarri = 0;
  Tarrj = halfsize+1;
  while (Tarri < halfsize) {
    if (Tarrj < totalsize) {
      if (Tarr[Tarri] < Tarr[Tarrj]) {
	arr[arri] = Tarr[Tarri];
	Tarri+=2;
      }
      else {
	arr[arri] = Tarr[Tarrj];
	Tarrj+=2;
      }
    }
    else {
      arr[arri] = Tarr[Tarri];
      Tarri+=2;
    }
    arri+=2;
  }
  while (Tarrj < totalsize) {
    arr[arri] = Tarr[Tarrj];
    arri+=2;
    Tarrj+=2;
  }
  
  /*Pairwise check of merged array:*/
  for(arri = 0; arri+1 < totalsize; arri+=2)
    if (arr[arri] > arr[arri+1]) {
      int t = arr[arri];
      arr[arri] = arr[arri+1];
      arr[arri+1] = t;
    }
  
  
  return 0;
}






