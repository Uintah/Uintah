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
 *  NrrdPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Nrrd_NrrdPort_h
#define SCI_Nrrd_NrrdPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Nrrd/Core/Datatypes/NrrdData.h>

namespace SCINrrd {

typedef SCIRun::SimpleIPort<NrrdDataHandle> NrrdIPort;
typedef SCIRun::SimpleOPort<NrrdDataHandle> NrrdOPort;

} // End namespace SCINrrd


#endif
