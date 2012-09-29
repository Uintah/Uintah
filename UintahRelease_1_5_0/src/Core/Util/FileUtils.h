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
#include <sstream>

using std::string;
using std::vector;

#include <Core/Util/share.h>

namespace SCIRun {

////////////////////////////////////
// InsertStringInFile()
// Inserts "insert" in front of all occurrances of 
// "match" within the file named "filename"

SCISHARE void InsertStringInFile( char* filename, const char* match, const char* insert );

////////////////////////////////////
// GetFilenamesEndingWith()
// returns a std::map of strings that contains
// all the files with extension "ext" inside
// the directory named "dir"


SCISHARE std::map<int,char*>* GetFilenamesEndingWith(const char* dir, 
                                                     string ext);

SCISHARE vector<string> GetFilenamesStartingWith(const string & dir,
                                                 const string & prefix);

SCISHARE std::pair<string, string> split_filename( const string & fname );

SCISHARE string findFileInPath( const string &filename, 
                                const string &path );

SCISHARE bool getInfo( const string & filename );  // prints out size, type, timestamp, etc about the file. Returns false if file does not exist.
SCISHARE bool validFile( const string & filename );
SCISHARE bool validDir( const string & filename );
SCISHARE bool isSymLink( const string & filename );

// Creates a temp file (in directoryPath), writes to it, checks the
// resulting files size, and then deletes it...  Informational
// messages about the test are returned in the 'error_stream'.  A
// 'procNumber' of -1 means you are not running MPI... otherwise pass
// in the processor's rank.
SCISHARE bool testFilesystem( const string & directoryPath,
                              std::stringstream & error_stream,
                              int procNumber = -1 );

SCISHARE string autocomplete( const string & instr );
SCISHARE string canonicalize( const string & filename );
SCISHARE string substituteTilde(const string &dirstr);

// Replaces '/' with '\' or vice-versa between unix and windows paths
SCISHARE void convertToWindowsPath( string & unixPath );
SCISHARE void convertToUnixPath( string & winPath );

// System copy, move, and delete commands.  The strings are not
// references since windows has to convert '/' to '\\', and we do that
// in the same string
SCISHARE int copyFile( const string & src, const string & dest );
SCISHARE int moveFile( const string & src, const string & dest );
SCISHARE int deleteFile( const string & filename);
SCISHARE int copyDir( const string & src, const string & dest);
SCISHARE int deleteDir( const string & filename);

// Replaces the existing extension of the filename with the value of ext
SCISHARE string changeExtension( const string & filename, const string &ext );

} // End namespace SCIRun

#ifdef _WIN32
// windows doesn't have dirent... make them here
struct dirent
{
    char *d_name;
};

struct DIR;

SCISHARE DIR *opendir(const char *);
SCISHARE int closedir(DIR *);
SCISHARE dirent *readdir(DIR *);

// not implemented yet...
SCISHARE void rewinddir(DIR *);
#endif // _WIN32

#endif // FILEUTILS_H

