/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#ifndef SCI_Core_StringUtil_h
#define SCI_Core_StringUtil_h 1

#include <string>
#include <vector>

#include <Core/Containers/share.h>

namespace SCIRun {
  using std::string;
  using std::vector;

SCISHARE bool string_to_int(const string &str, int &result);
SCISHARE bool string_to_double(const string &str, double &result);
SCISHARE bool string_to_unsigned_long(const string &str, unsigned long &res);

SCISHARE string to_string(int val);
SCISHARE string to_string(unsigned int val);
SCISHARE string to_string(unsigned long val);
SCISHARE string to_string(double val);

SCISHARE string string_toupper( const string & inString );
SCISHARE string string_tolower( const string & inString );

//////////
// Remove directory name
SCISHARE string basename(const string &path);

//////////
// Return directory name
SCISHARE string pathname(const string &path);

// Split a string into multiple parts, separated by any of the separator characters.
SCISHARE vector<string> split_string( const string & str, const vector<char> & separators );
SCISHARE string         concatStrings( const vector<string> strings );

/////////
// C++ify a string, turn newlines into \n, use \t, \r, \\ \", etc.
SCISHARE string string_Cify(const string &str);

//////////
// Remove leading and trailing white space (blanks, tabs, \n, \r) from string.
SCISHARE void collapse( string & str );

//////////
// Unsafe cast from string to char *, used to export strings to C functions.
SCISHARE char* ccast_unsafe( const string & str );

// replaces all occurances of 'substr' in 'str' with 'replacement'.  'str' is updated in place.
SCISHARE void replace_substring( string & str, 
                                 const string &substr, 
                                 const string &replacement );

// Returns true if 'str' ends with the string 'substr'
SCISHARE bool ends_with( const string & str, const string & substr );

// Returns the number of 'substr' in 'str'.  (ie: if str is 'aaaa' and substr is 'aaa', then 2 is returned.)
SCISHARE unsigned int count_substrs( const string & str, const string & substr );

} // End namespace SCIRun

#endif
