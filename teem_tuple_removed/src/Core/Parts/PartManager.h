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
 *  PartManager.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_PartManager_h
#define SCI_PartManager_h 

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Signals.h>
#include <Core/Parts/Part.h>
#include <Core/Parts/PartInterface.h>

namespace SCIRun {
  
class SCICORESHARE PartManager : public Part, public PartInterface  {
private:
  vector< Part *> parts_;
  Part *current_;

public:
  PartManager( PartInterface *parent = 0, const string &name="PartManager",
	       bool=true);
  virtual ~PartManager();

  virtual int add_part( Part * );

  Part *current() { return current_;}
  Part *operator ->() { return current_; }

//   virtual void report_children( SlotBase1<PartInterface*> &slot );

  // Slot
  void select_part( int );

  // Signal
  Signal1< const string & > has_part;
  Signal1< int > part_selected;
};


template<class T>
class PartManagerOf : public PartManager {
public:
  PartManagerOf( PartInterface *parent = 0, 
		 const string &name="PartManagerOf",
		 bool init=true) 
    : PartManager( parent, name, init )
  {
  }

  virtual ~PartManagerOf() {}

  virtual int add_part( T *part ) { return PartManager::add_part(part); }

  T *operator ->() { return static_cast<T*>(current()); }
};

} // namespace SCIRun

#endif // SCI_PartManager_h

