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
 *  PathPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Novemeber 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Dataflow_Datatypes_PathPort_h
#define SCI_Dataflow_Datatypes_PathPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Geom/Path.h>

namespace SCIRun {


typedef SimpleIPort<PathHandle> PathIPort;
typedef SimpleOPort<PathHandle> PathOPort;

} // End namespace SCIRun


#endif
