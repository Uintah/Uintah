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
 *  Part.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Part_h
#define SCI_Part_h 

#include <string>
#include <map>
#include <vector>
#include <stdio.h>

#include <Core/GuiInterface/TCLArgs.h>
#include <Core/Util/Signals.h>

namespace SCIRun {

using std::string;
using std::vector;
using std::map;
  
class GuiVar;
class PartPort;

class Part  {
protected:
  string name_;
  string type_;
  Part *parent_;
  PartPort *port_;
  vector<Part*> children_;
  vector<GuiVar *> vars_;

public:
  Part( Part *parent=0, const string &name="", const string &type="" );
  virtual ~Part();

  const string &name() { return name_; }
  const string &type() { return type_; }

  virtual void init();

  void add_child( Part *child );
  void rem_child( Part *child );

  vector<GuiVar*> &get_gui_vars() { return vars_; }

  virtual void add_gui_var( GuiVar * );
  virtual void rem_gui_var( GuiVar * );
  const string &get_var( const string &, const string &);
  void set_var( const string &, const string &, const string & );
  
  virtual void emit_vars( ostream& out, string &midx );

  // tcl compatibility
  virtual void tcl_command( TCLArgs &, void *) {}
  void reset_vars();
  
  static Signal1<const string &> tcl_execute;
  static Signal2<const string &, string &> tcl_eval;
  static Signal3<const string &, Part *, void *> tcl_add_command;
  static Signal1<const string &> tcl_delete_command;

  static int get_gui_stringvar(const string &, const string &, string & );
  static int get_gui_doublevar(const string &, const string &, double & );
  static int get_gui_intvar(const string &, const string &, int & );
  static void set_gui_var(const string &, const string &, const string & );
};

class PartPort {
private:
  Part *part_;

public:
  PartPort( Part *part) : part_(part) {}
  virtual ~PartPort() {}

  vector<GuiVar *> &get_gui_vars() { return part_->get_gui_vars(); }
};

} // namespace SCIRun

#endif // SCI_Part_h
