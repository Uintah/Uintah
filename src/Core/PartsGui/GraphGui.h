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

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/PartsGui/PartGui.h>
#include <Core/Util/Signals.h>

namespace SCIRun {

class Diagram;
class Graph;
class CrowdMonitor;
class LockedPolyline;
class DrawObj;

using std::vector;

class GraphGui : public PartGui {
public:
  GraphGui( const string &name, const string &script = "GraphGui"); 
  virtual ~GraphGui();

#ifdef CHRIS
  void reset( const vector<DrawObj*>& );
  void add_values( unsigned, const vector<double> & );
#else
  void reset( int );
  void add_values( const vector<double> & );
#endif

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


