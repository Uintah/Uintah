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

namespace PSECore {
namespace Dataflow {

////////////////////////////////////
//
// InsertStringInFile()
//
// Inserts "insert" in front of all occurrances of 
// "match" within the file named "filename"
// 

void InsertStringInFile(char* filename, char* match, char* insert);


////////////////////////////////////
//
// GetFilenamesEndingWith()
//
// returns a std::map of strings that contains
// all the files with extension "ext" inside
// the directory named "dir"
//

std::map<int,char*>* GetFilenamesEndingWith(char* dir, char* ext);

} // Dataflow
} // PSECore

#endif

