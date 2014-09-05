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
 *  Jacobi_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef Jacobi_Jacobi_impl_h
#define Jacobi_Jacobi_impl_h

#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <testprograms/Component/Jacobi/Jacobi_sidl.h>

namespace Jacobi_ns {

    class Jacobi_impl : public Jacobi {
    public:
	Jacobi_impl();
	virtual ~Jacobi_impl();
	virtual int solveHeatEquation(SSIDL::array2<double>& ,double top,double bottom, 
				      double left,double right);
    };
} // End namespace Jacobi

#endif

