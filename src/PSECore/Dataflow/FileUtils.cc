/* FileUtils.cc */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <PSECore/Dataflow/FileUtils.h>

namespace PSECore {
namespace Dataflow {

void InsertStringInFile(char* filename, char* match, char* replacement)
{
  char* string = 0;
  char* mod = 0;

  mod = new char[strlen(replacement)+strlen(match)+25];
  sprintf(mod,"%s%s",replacement,match);

  string = new char[strlen(match)+strlen(replacement)+100];
  sprintf(string,"sed -e 's,%s,%s,g' %s > %s.mod\n",match,mod,
	  filename,filename);
  system(string);

  sprintf(string,"mv -f %s.mod %s\n",filename,filename);
  system(string);

  delete[] string;
  delete[] mod;
}

std::map<int,char*>* GetFilenamesEndingWith(char* d, char* ext)
{
  std::map<int,char*>* newmap = 0;
  dirent* file = 0;
  DIR* dir = opendir(d);
  char* newstring = 0;

  if (!dir) {
    printf("directory not found: %s\n",d);
    return 0;
  }

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

} // Dataflow
} // PSECore

