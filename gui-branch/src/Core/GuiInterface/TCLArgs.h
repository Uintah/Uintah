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
 *  TCLArgs.h: Interface to TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef TCLArgs_h
#define TCLArgs_h 

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace SCIRun {

class TCLArgs {
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
    static string make_list(const vector<string>&);
};


} // End namespace SCIRun


#endif
