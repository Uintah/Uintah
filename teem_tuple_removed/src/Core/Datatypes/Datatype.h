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
 *  Datatype.h: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Datatype_h
#define SCI_project_Datatype_h 1

#include <Core/share/share.h>

#include <Core/Persistent/Persistent.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {


class SCICORESHARE Datatype : public Persistent {
public:
  //! needed for our smart pointers -- LockingHandle<T>
  int ref_cnt;
  Mutex lock;

  //! unique id for each instance
  int generation;
  Datatype();
  Datatype(const Datatype&);
  Datatype& operator=(const Datatype&);
  virtual ~Datatype();
};

} // End namespace SCIRun


#endif /* SCI_project_Datatype_h */
