/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

namespace Uintah {

bool string_to_int(const std::string &str, int &result);
bool string_to_double(const std::string &str, double &result);
bool string_to_unsigned_long(const std::string &str, unsigned long &res);

std::string to_string(int val);
std::string to_string(unsigned int val);
std::string to_string(unsigned long val);
std::string to_string(double val);

std::string string_toupper( const std::string & inString );
std::string string_tolower( const std::string & inString );

//////////
// Remove directory name
std::string basename(const std::string &path);

//////////
// Return directory name
std::string pathname(const std::string &path);

// Split a std::string into multiple parts, separated by any of the separator characters.
std::vector<std::string> split_string( const std::string & str, const std::vector<char> & separators );
std::string concatStrings( const std::vector<std::string> strings );

/////////
// C++ify a std::string, turn newlines into \n, use \t, \r, \\ \", etc.
std::string string_Cify(const std::string &str);

//////////
// Remove leading and trailing white space (blanks, tabs, \n, \r) from std::string.
void collapse( std::string & str );

//////////
// Unsafe cast from std::string to char *, used to export strings to C functions.
char* ccast_unsafe( const std::string & str );

// replaces all occurances of 'substr' in 'str' with 'replacement'.  'str' is updated in place.
void replace_substring( std::string & str,
                                 const std::string &substr,
                                 const std::string &replacement );

// Returns true if 'str' ends with the std::string 'substr'
bool ends_with( const std::string & str, const std::string & substr );

// Returns the number of 'substr' in 'str'.  (ie: if str is 'aaaa' and substr is 'aaa', then 2 is returned.)
unsigned int count_substrs( const std::string & str, const std::string & substr );

} // End namespace Uintah

#endif
