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

#ifndef OESort_OESort_impl_h
#define OESort_OESort_impl_h

#include <testprograms/Component/OESort/OESort_sidl.h>

namespace OESort_ns {

    class OESort_impl : public OESort {
    public:
	OESort_impl();
	virtual ~OESort_impl();
	virtual int sort(const SSIDL::array1<int>&, SSIDL::array1<int>&, SSIDL::array1<int>&);
    };

    class OESplit_impl : public OESplit {
    public:
	OESplit_impl();
	virtual ~OESplit_impl();
	virtual int split(const SSIDL::array1<int>&, SSIDL::array1<int>&);
	OESort::pointer ss;
    };

} // End namespace OESort

#endif

