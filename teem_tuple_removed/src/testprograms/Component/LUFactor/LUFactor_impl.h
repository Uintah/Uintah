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
 *  LUFactor_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef LUFactor_LUFactor_impl_h
#define LUFactor_LUFactor_impl_h

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <testprograms/Component/LUFactor/LUFactor_sidl.h>

namespace LUFactor_ns {

    class LUFactor_impl : public LUFactor {
    public:
	LUFactor_impl();
	virtual ~LUFactor_impl();
	virtual int LUFactorize(const SSIDL::array2<double>& );
    };
} // End namespace LUFactor

#endif

