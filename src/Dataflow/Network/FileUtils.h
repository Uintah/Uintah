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

} // Dataflow
} // PSECore

#endif

