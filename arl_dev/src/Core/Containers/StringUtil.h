/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

SCISHARE bool string_to_int(const std::string &str, int &result);
SCISHARE bool string_to_double(const std::string &str, double &result);
SCISHARE bool string_to_unsigned_long(const std::string &str, unsigned long &res);

SCISHARE std::string to_string(int val);
SCISHARE std::string to_string(unsigned int val);
SCISHARE std::string to_string(unsigned long val);
SCISHARE std::string to_string(double val);

SCISHARE std::string string_toupper( const std::string & inString );
SCISHARE std::string string_tolower( const std::string & inString );

//////////
// Remove directory name
SCISHARE std::string basename(const std::string &path);

//////////
// Return directory name
SCISHARE std::string pathname(const std::string &path);

// Split a std::string into multiple parts, separated by any of the separator characters.
SCISHARE std::vector<std::string> split_string( const std::string & str, const std::vector<char> & separators );
SCISHARE std::string concatStrings( const std::vector<std::string> strings );

/////////
// C++ify a std::string, turn newlines into \n, use \t, \r, \\ \", etc.
SCISHARE std::string string_Cify(const std::string &str);

//////////
// Remove leading and trailing white space (blanks, tabs, \n, \r) from std::string.
SCISHARE void collapse( std::string & str );

//////////
// Unsafe cast from std::string to char *, used to export strings to C functions.
SCISHARE char* ccast_unsafe( const std::string & str );

// replaces all occurances of 'substr' in 'str' with 'replacement'.  'str' is updated in place.
SCISHARE void replace_substring( std::string & str,
                                 const std::string &substr,
                                 const std::string &replacement );

// Returns true if 'str' ends with the std::string 'substr'
SCISHARE bool ends_with( const std::string & str, const std::string & substr );

// Returns the number of 'substr' in 'str'.  (ie: if str is 'aaaa' and substr is 'aaa', then 2 is returned.)
SCISHARE unsigned int count_substrs( const std::string & str, const std::string & substr );

} // End namespace SCIRun

#endif
