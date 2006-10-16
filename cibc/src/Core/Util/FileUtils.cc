/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


/* FileUtils.cc */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#include <dirent.h>
#else
#include <io.h>
typedef unsigned short mode_t;
#define MAXPATHLEN 256
#endif
#include <sys/stat.h>
#include <Core/Util/FileUtils.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Assert.h>

#include <iostream>

namespace SCIRun {


/* Normally, I would just use sed via system() to edit a file,
   but for some reason system() calls never work from Dataflow 
   processes in linux.  Oh well, sed isn't natively available
   under windows, so I'd have to do something like this anyhow
   - Chris Moulding */ 

void InsertStringInFile(char* filename, char* match, char* replacement)
{
  char* newfilename = new char[strlen(filename)+2];
  char c;
  sprintf(newfilename,"%s~",filename);
  FILE* ifile;
  FILE* ofile;

  /* create a copy of the original file */
  ifile = fopen(filename,"r");
  ofile = fopen(newfilename,"w");

  c = (char)fgetc(ifile);
  while (c!=(char)EOF) {
    fprintf(ofile,"%c",c);
    c = (char)fgetc(ifile);
  }
  fclose(ifile);
  fclose(ofile);

  /* search the copy for an instance of "match" */
  int index1 = 0;
  unsigned int index2 = 0;
  int foundat = -1;
  ifile = fopen(newfilename,"r");
  c = (char)fgetc(ifile);
  while (c!=(char)EOF) {
    if (c==match[index2]) {
      foundat = index1;
      while (index2<strlen(match) && c!=(char)EOF && c==match[index2]) {
	c = (char)fgetc(ifile);
	index1++;
	index2++;
      }
      if (foundat>=0 && index2!=strlen(match)) {
	foundat = -1;
	index2 = 0;
      } else
	break;
    }
    c = (char)fgetc(ifile);
    index1++;
  }
  fclose(ifile);

  /* if an instance of match was found, 
     insert the indicated string */
  if (foundat>=0) {
    index1 = 0;
    ifile = fopen(newfilename,"r");
    ofile = fopen(filename,"w");
    c = (char)fgetc(ifile);
    while (c!=(char)EOF) {
      if (index1==foundat)
        fprintf(ofile,"%s",replacement);
      fprintf(ofile,"%c",c);
      c = (char)fgetc(ifile);
      index1++;
    }
    fclose(ifile);
    fclose(ofile);
  } 
}

#if 0
void InsertStringInFile(char* filename, char* match, char* replacement)
{
  char* string = 0;
  char* mod = 0;

  mod = new char[strlen(replacement)+strlen(match)+25];
  sprintf(mod,"%s%s",replacement,match);

  string = new char[strlen(match)+strlen(replacement)+100];
  sprintf(string,"sed -e 's,%s,%s,g' %s > %s.mod &\n",match,mod,
	  filename,filename);
  system(string);

  sprintf(string,"mv -f %s.mod %s\n",filename,filename);
  system(string);

  delete[] string;
  delete[] mod;
}
#endif

std::map<int,char*>* GetFilenamesEndingWith(char* d, char* ext)
{
  std::map<int,char*>* newmap = 0;
  dirent* file = 0;
  DIR* dir = opendir(d);
  char* newstring = 0;

  if (!dir) 
    return 0;

  newmap = new std::map<int,char*>;

  file = readdir(dir);
  while (file) {
    if ((strlen(file->d_name)>=strlen(ext)) && 
	(strcmp(&(file->d_name[strlen(file->d_name)-strlen(ext)]),ext)==0)) {
      newstring = new char[strlen(file->d_name)+1];
      sprintf(newstring,"%s",file->d_name);
      newmap->insert(std::pair<int,char*>(newmap->size(),newstring));
    }
    file = readdir(dir);
  }

  closedir(dir);
  return newmap;
}


vector<string>
GetFilenamesStartingWith(const string &dirstr,
                         const string &prefix)
{
  vector<string> files(0);
  DIR* dir = opendir(dirstr.c_str());
  if (!dir) {
    return files;
  }

  dirent* file = readdir(dir);
  while (file) {
    ASSERT(file->d_name);
    string dname = file->d_name;
    string::size_type pos = dname.find(prefix);
    if (pos == 0) {
      files.push_back(dname);
    }
    file = readdir(dir);
  }

  closedir(dir);
  return files;
}


std::pair<string, string>
split_filename(string fname) {
  if (fname[fname.size()-1] == '/') {
    fname = fname.substr(0, fname.size()-1);
  }
  
  if (validDir(fname)) {
    return make_pair(fname, string(""));
  }
    
  string::size_type pos = fname.find_last_of("/");
  std::pair<string, string> dirfile = std::make_pair
    (fname.substr(0, pos+1), fname.substr(pos+1, fname.length()-pos-1));

  return dirfile;
}



bool
validFile(std::string filename) 
{
  struct stat buf;
  if (stat(filename.c_str(), &buf) == 0)
  {
    mode_t &m = buf.st_mode;
    return (m & S_IRUSR && S_ISREG(m) && !S_ISDIR(m));
  }
  return false;
}



bool
validDir(std::string dirname) 
{
  struct stat buf;
  if (stat(dirname.c_str(), &buf) == 0)
  {
    mode_t &m = buf.st_mode;
    return (m & S_IRUSR && !S_ISREG(m) && S_ISDIR(m));
  }
  return false;
}
    

// findFileInPath
// Searches the colon-seperated 'path' string variable for a file named
// 'file'.  From left to right in the path string each directory is
// tested to see if the file named 'file' is in it.
// 
// If the file is found, it returns the DIRECTORY that the file is located in
// Otherwise if the file is not found in the path, returns an empty string
//
// If the file is found in multiple directories in the 'path', only
// the first matching directory is returned
std::string
findFileInPath(const std::string &file, const std::string &path)
{
  string::size_type beg = 0;
  string::size_type end = 0;

  while (beg < path.length()) {
#ifdef _WIN32
    end = path.find(":",beg+2);
#else
    end = path.find(":",beg);
#endif
    if (end == string::npos) end = path.size();

    string dir(path, beg, end-beg);
    ASSERT(!dir.empty());
    // Append a slash if there isn't one
    if (validDir(dir)) {
      if (dir[dir.length()-1] != '/') dir = dir + "/";
      string filename = dir + file;
      if (validFile(filename)) {
        // see comments at function start
        return dir;
      }
    }

    beg = end+1;
  }
  return "";
}
    



string
autocomplete(const string &instr) {
  string str = instr;
  std::pair<string, string> dirfile = split_filename(str);
  if (!validDir(dirfile.first)) return str;
  
  vector<string> files = 
    GetFilenamesStartingWith(dirfile.first, dirfile.second);
  
  if (files.empty()) {
    return str;
  } if (files.size() == 1) {
    str = dirfile.first + files[0];
    if (validDir(str)) {
      str = str + "/";
    }
  } else {
    unsigned int j0 = dirfile.second.size();
    unsigned int j = j0;
    do {
      for (unsigned int i = 1; i < files.size(); ++i) {
        if ((j == files[i].size()) || (files[i][j] != files[i-1][j])) {
          str = str + files[i].substr(j0, j-j0);
          return str;
        }
      }
      ++j;
    } while (1);
  }
  return str;
}

    
    


} // End namespace SCIRun
