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
 *  PartPort.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef PartPort_h
#define PartPort_h 

#include <string>
#include <map>
#include <vector>
#include <stdio.h>

#include <Core/Parts/GuiVar.h>
#include <Core/Parts/Part.h>
#include <Core/Util/Signals.h>

namespace SCIRun {

using std::string;
using std::vector;
using std::map;
  
class PartPort {
private:
  Part *part_;

public:
  // Signals
  Signal1 <GuiVar *> var_set_signal;
  Signal1 <const string & > command_signal;
  Signal2 <const string &, string &> eval_signal; 

public:
  PartPort( Part *part ) : part_(part) {}
  virtual ~PartPort() {}

  const string &type();

  // value set by the a GUI 
  template<class T> void set_var( const string& name, T &v ) 
  { 
    GuiTypedVar<T> *var;
    var = part_->find_var(name);
    if ( ! var ) {
      cerr << "Bug: asking for the wrong type of variable\n";
      return;
    }
    var->value_ = v;
  }

  void command( TCLArgs & );
};

} // namespace SCIRun

#endif // PartPort_h


