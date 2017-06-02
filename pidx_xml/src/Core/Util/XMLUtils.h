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
 *  XMLUtil.h: Helper functions for parsing XML files on the fly.
 *
 *  Normally/originally we use the XML class (libxml2) to store/parse xml files...
 *  However, as the number of Patches in a simulation has increased, the memory overhead
 *  in creating those XML datastructures is too much, so for the larger XML files
 *  we are reading them in one line at a time and pulling the data out directly.
 *
 *  Written by:
 *   J. Davison de St. Germain
 *   Department of Computer Science
 *   University of Utah
 *   Nov 2014
 *
 */

#ifndef SCI_Core_XMLUtil_h
#define SCI_Core_XMLUtil_h 1

#include <stdio.h>

#include <string>
#include <vector>

namespace Uintah {
namespace UintahXML {

enum CheckType { INT_TYPE, FLOAT_TYPE };

// 'validateType()' determines if the input string is a valid int or float (based on 'type').
//  If the string is not valid, a ProblemSetupException is thrown.
//
void                     validateType( const std::string & stringValue, CheckType type );

// Returns a (collapsed (no begin/end spaces)) string that holds one line from the flie.
// 'fp' is updated to point to following line in file.
//
// NOTES: getLine() skips blank lines.  If the return result is "", then the end of the file
//            was reached.
//
std::string              getLine( FILE * fp );                    

// Used for breaking <tag> value </tag> into three separate strings.
//
std::vector<std::string> splitXMLtag( const std::string & line );


} // End namespace UintahXML
} // End namespace Uintah

#endif
