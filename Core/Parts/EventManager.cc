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
 *  EventManager.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Persistent/Pstreams.h>
#include <Core/Parts/SciEvent.h>
#include <Core/Parts/EventManager.h>

namespace SCIRun {
  

EventManager::EventManager()
{
  log_ = new TextPiostream( "events.log", Piostream::Write );
}
 
EventManager::~EventManager()
{
  // close files
  delete log_;
}

void 
EventManager::record( SciEvent *event)
{
  // Pio(log_, timestamp);
  event->io( *log_);
};

} // namespace SCIRun

