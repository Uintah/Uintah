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
 *  TCL.h: Interface to TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_project_TCL_h
#define sci_project_TCL_h 1

#include <Core/Containers/Array1.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiManager.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace SCIRun {

class GuiVar;

class SCICORESHARE TCLArgs {
    vector<string> args_;
public:
    bool have_error_;
    bool have_result_;
    string string_;

    TCLArgs(int argc, char* argv[]);
    ~TCLArgs();
    int count();
    string operator[](int i);

    void error(const string&);
    void result(const string&);
    void append_result(const string&);
    void append_element(const string&);

    static string make_list(const string&, const string&);
    static string make_list(const string&, const string&, const string&);
    static string make_list(const Array1<string>&);
    static string make_list(const vector<string>&);
};

class SCICORESHARE TCL {
    vector<GuiVar*> vars;
    friend class GuiVar;
    void register_var(GuiVar*);
    void unregister_var(GuiVar*);
public:
    virtual void emit_vars(std::ostream& out, string& midx);
    static void initialize();
    static void execute(const string& str);
    static int eval(const string& str, string& result);
    static void source_once(const string&);
    static void add_command(const string&, TCL*, void*);
    static void delete_command( const string& command );

    TCL();
    virtual ~TCL();
    virtual void tcl_command(TCLArgs&, void*)=0;

    void reset_vars();

    // To get at tcl variables
    int get_gui_stringvar(const string& base, const string& name,
			  string& value);
    int get_gui_boolvar(const string& base, const string& name,
			int& value);
    int get_gui_doublevar(const string& base, const string& name,
			  double& value);
    int get_gui_intvar(const string& base, const string& name,
		       int& value);

    void set_guivar(const string& base, const string& name,
		    const string& value);

};

struct TCLCommandData {
    TCL* object;
    void* userdata;
};

} // End namespace SCIRun


#endif
