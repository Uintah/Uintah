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
 *  GraphGui.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_GraphGui_h
#define SCI_GraphGui_h 

#include <vector>

#include <Core/PartsGui/PartGui.h>
#include <Core/Util/Signals.h>

namespace SCIRun {

class Diagram;
class Graph;
class CrowdMonitor;
class LockedPolyline;

class GraphGui : public PartGui {
public:
  GraphGui( const string &name, const string &script = "GraphGui"); 
  virtual ~GraphGui();

  void reset( int =0 );
  void add_values( vector<double> & );
  void attach( PartInterface * );

  virtual void set_window( const string & );

private:
  vector<LockedPolyline *> poly_;
  Diagram *diagram_;
  Graph *graph_;
  CrowdMonitor *monitor_;
};

} // namespace SCIRun

#endif // SCI_GraphGui_h


