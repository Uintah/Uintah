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

namespace SCIRun {

////////////////////////////////////
// InsertStringInFile()
// Inserts "insert" in front of all occurrances of 
// "match" within the file named "filename"

void InsertStringInFile(char* filename, char* match, char* insert);


////////////////////////////////////
// GetFilenamesEndingWith()
// returns a std::map of strings that contains
// all the files with extension "ext" inside
// the directory named "dir"

std::map<int,char*>* GetFilenamesEndingWith(char* dir, char* ext);

} // End namespace SCIRun

#endif

