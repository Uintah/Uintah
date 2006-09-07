/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/* FileUtils.h 
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   Copyright (c) 2000
 *   University of Utah
 */

#ifndef FILEUTILS_H
#define FILEUTILS_H 1

#include <map>
#include <string>
#include <vector>

using std::string;
using std::vector;

#include <Core/Util/share.h>

namespace SCIRun {

////////////////////////////////////
// InsertStringInFile()
// Inserts "insert" in front of all occurrances of 
// "match" within the file named "filename"

SCISHARE void InsertStringInFile(char* filename, char* match, char* insert);


////////////////////////////////////
// GetFilenamesEndingWith()
// returns a std::map of strings that contains
// all the files with extension "ext" inside
// the directory named "dir"

SCISHARE std::map<int,char*>* GetFilenamesEndingWith(char* dir, char* ext);

SCISHARE vector<string> GetFilenamesStartingWith(const string & dir,
                                                 const string & prefix);

SCISHARE std::pair<string, string> split_filename(string fname);

SCISHARE std::string findFileInPath(const std::string &filename, 
                                    const std::string &path);
SCISHARE bool validFile(std::string filename);
SCISHARE bool validDir(std::string filename);

} // End namespace SCIRun

#endif

