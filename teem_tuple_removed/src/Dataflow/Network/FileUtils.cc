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

/* FileUtils.cc */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <Dataflow/Network/FileUtils.h>

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

  return newmap;
}

} // End namespace SCIRun

