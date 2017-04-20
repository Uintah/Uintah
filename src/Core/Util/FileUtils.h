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

/* FileUtils.h 
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   University of Utah
 */

#ifndef FILEUTILS_H
#define FILEUTILS_H 1

#include <map>
#include <string>
#include <vector>

namespace Uintah {

////////////////////////////////////
// InsertStringInFile()
// Inserts "insert" in front of all occurrances of 
// "match" within the file named "filename"

void InsertStringInFile( char* filename, const char* match, const char* insert );

////////////////////////////////////
// GetFilenamesEndingWith()
// returns a std::map of strings that contains
// all the files with extension "ext" inside
// the directory named "dir"


std::map<int,char*>* GetFilenamesEndingWith(const char* dir,
                                                     std::string ext);

std::vector<std::string> GetFilenamesStartingWith(const std::string & dir,
                                                 const std::string & prefix);

std::pair<std::string, std::string> split_filename( const std::string & fname );

std::string findFileInPath( const std::string &filename,
                                const std::string &path );

bool getInfo( const std::string & filename );  // prints out size, type, timestamp, etc about the file. Returns false if file does not exist.
bool validFile( const std::string & filename );
bool validDir( const std::string & filename );
bool isSymLink( const std::string & filename );

// Creates a temp file (in directoryPath), writes to it, checks the
// resulting files size, and then deletes it...  Informational
// messages about the test are returned in the 'error_stream'.  A
// 'procNumber' of -1 means you are not running MPI... otherwise pass
// in the processor's rank.
bool testFilesystem( const std::string & directoryPath,
                              std::stringstream & error_stream,
                              int procNumber = -1 );

std::string autocomplete( const std::string & instr );
std::string canonicalize( const std::string & filename );
std::string substituteTilde(const std::string &dirstr);

// Replaces '/' with '\' or vice-versa between unix and windows paths
void convertToWindowsPath( std::string & unixPath );
void convertToUnixPath( std::string & winPath );

// System copy, move, and delete commands.  The strings are not
// references since windows has to convert '/' to '\\', and we do that
// in the same std::string
int copyFile( const std::string & src, const std::string & dest );
int moveFile( const std::string & src, const std::string & dest );
int deleteFile( const std::string & filename);
int copyDir( const std::string & src, const std::string & dest);
int deleteDir( const std::string & filename);

// Replaces the existing extension of the filename with the value of ext
std::string changeExtension( const std::string & filename, const std::string &ext );

} // End namespace Uintah

#endif // FILEUTILS_H

