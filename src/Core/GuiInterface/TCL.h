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
#include <Core/Containers/String.h>

namespace SCIRun {

class GuiVar;

class SCICORESHARE TCLArgs {
    Array1<clString> args_;
public:
    bool have_error_;
    bool have_result_;
    clString string_;

    TCLArgs(int argc, char* argv[]);
    ~TCLArgs();
    int count();
    clString operator[](int i);

    void error(const clString&);
    void result(const clString&);
    void append_result(const clString&);
    void append_element(const clString&);

    static clString make_list(const clString&, const clString&);
    static clString make_list(const clString&, const clString&, const clString&);
    static clString make_list(const Array1<clString>&);
};

class SCICORESHARE TCL {
    Array1<GuiVar*> vars;
    friend class GuiVar;
    void register_var(GuiVar*);
    void unregister_var(GuiVar*);
public:
    virtual void emit_vars(std::ostream& out, clString& midx);
    static void initialize();
    static void execute(const clString&);
    static void execute(char*);
    static int eval(const clString&, clString& result);
    static int eval(char*, clString& result);
    static void source_once(const clString&);
    static void add_command(const clString&, TCL*, void*);
    static void delete_command( const clString& command );

    TCL();
    virtual ~TCL();
    virtual void tcl_command(TCLArgs&, void*)=0;

    void reset_vars();

    // To get at tcl variables
    int get_gui_stringvar(const clString& base, const clString& name,
			  clString& value);
    int get_gui_boolvar(const clString& base, const clString& name,
			int& value);
    int get_gui_doublevar(const clString& base, const clString& name,
			  double& value);
    int get_gui_intvar(const clString& base, const clString& name,
		       int& value);

    void set_guivar(const clString& base, const clString& name,
		    const clString& value);

};

} // End namespace SCIRun


#endif
