/* FileUtils.cc */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

} // Dataflow
} // PSECore

