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
 *  StringUtil.h: some useful string functions
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Core_StringUtil_h
#define SCI_Core_StringUtil_h 1

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  using std::string;
  using std::vector;

bool string_to_int(const string &str, int &result);
bool string_to_double(const string &str, double &result);

string to_string(int val);
string to_string(unsigned int val);
string to_string(double val);

//////////
// Remove directory name
string basename(const string &path);

//////////
// Return directory name
string pathname(const string &path);

  // Split a string into multiple parts, separated by the character sep
  vector<string> split_string(const std::string& str, char sep);

//////////
// Unsafe cast from string to char *, used to export strings to C functions.
char * ccast_unsafe(const string &str);

} // End namespace SCIRun

#endif
