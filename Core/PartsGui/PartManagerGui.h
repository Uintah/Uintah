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
 *  PartManagerGui.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Oct 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_PartManagerGui_h
#define SCI_PartManagerGui_h 

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/PartsGui/PartGui.h>
#include <Core/Util/Signals.h>

namespace SCIRun {

class PartManager;

class PartManagerGui : public PartGui {
public:
  PartManagerGui( const string &name, const string &script="PartManager"); 
  virtual ~PartManagerGui();

  virtual void attach( PartInterface * );
//   virtual void set_window( const string & );

  // Slot
  void add_part( const string & );
  void select_part( int );

  // Signal
  Signal1<int> part_selected;
};

} // namespace SCIRun

#endif // SCI_PartManagerGui_h
