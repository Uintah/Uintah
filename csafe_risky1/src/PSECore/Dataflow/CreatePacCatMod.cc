/*
 *  CreatePacCatMod.cc: 
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <stdlib.h>
#include <stdio.h>

namespace PSECore {
namespace Dataflow {

#ifndef _WIN32

int CreatePac(const char* pac)
{
  char command[10000]="\0";

  /* copy the NEW_PACKAGE directory from the scripts directory, and
     edit the contents of some of the files inside it */

  sprintf(command,"cp -r ./scripts/NEW_PACKAGE %s",pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/sub.in > %s/sub.mk",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/components.in > %s/components.xml",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/share/share.in > %s/share/share.h",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/share/DllEntry.in > %s/share/DllEntry.cc",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/Datatypes/sub.in > %s/Datatypes/sub.mk",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/Datatypes/none/sub.in > %s/Datatypes/none/sub.mk",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/GUI/sub.in > %s/GUI/sub.mk",
                   pac,pac,pac);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/g\' \
                   %s/Modules/sub.in > %s/Modules/sub.mk",
                   pac,pac,pac);
  system(command);

  /* delete the intermediate and CVS files */

  sprintf(command,"find ./%s -name \"*.in\" | xargs rm",pac);
  system(command);
  sprintf(command,"find ./%s -name CVS | xargs rm -rf",pac);
  system(command);

  return 1;
}  

int CreateCat(const char* pac, const char* cat)
{
  char command[10000]="\0";

  /* copy the NEW_CATEGORY directory from the scripts directory, and
     edit the contents of the file inside it */

  sprintf(command,"cp -r ./scripts/NEW_CATEGORY %s/Modules/%s",pac,cat);
  system(command);

  sprintf(command,"sed -e \'s/PACKAGE_NAME/%s/g\' \
                       -e \'s/CATEGORY_NAME/%s/g\' \
                   %s/Modules/%s/sub.in > %s/Modules/%s/sub.mk",
                   pac,cat,pac,cat,pac,cat);
  system(command);

  /* edit the makefile (sub.mk) above this new dirctory to include the new
     category as a "SUBDIR" */

  sprintf(command,"mv ./%s/Modules/sub.mk ./%s/Modules/sub.in",pac,pac);
  system(command);

  sprintf(command,"sed -e \'s/#\\[INSERT NEW SUBDIRS HERE\\]/"
	          "	$(SRCDIR)\\/%s\\\\\\\n"
                  "#\\[INSERT NEW SUBDIRS HERE\\]/g\'"
                  " %s/Modules/sub.in > %s/Modules/sub.mk",cat,pac,pac);
  system(command);

  /* edit the components.xml file to include an entry for the new category */

  sprintf(command,"mv ./%s/components.xml ./%s/components.in",pac,pac);
  system(command);

  sprintf(command,"sed -e \'s/\\<!-- INSERT NEW CATEGORY HERE --\\>/"
                  "\\<scirun-library category=\"%s\"\\>\\\n"
                  "  \\<soNames\\>\\\n"
                  "    \\<soName\\>lib%s_Modules_%s.so\\<\\/soName\\>\\\n"
                  "    \\<soName\\>lib%s.so\\<\\/soName\\>\\\n"
                  "  \\<\\/soNames\\>\\\n"
                  "\\\n"
                  "  \\<!-- INSERT NEW %s COMPONENT HERE --\\>\\\n"
                  "  \\<\\/scirun-library\\>\\\n"
                  "  \\<!-- INSERT NEW CATEGORY HERE --\\>/g\'"
                  " %s/components.in > %s/components.xml",
                  cat,pac,cat,pac,cat,pac,pac);
  system(command);

  /* delete the intermediate files */

  sprintf(command,"find ./%s -name \"*.in\" | xargs rm",pac);
  system(command);

  return 1;
}  

int CreateMod(const char* pac, const char* cat, const char* mod)
{
  char command[10000]="\0";

  /* copy the NEW_MODULE file from the scripts directory, and edit
     it's contents */

  sprintf(command,"cp ./scripts/NEW_MODULE %s/Modules/%s/%s.in",pac,cat,mod);
  system(command);

  sprintf(command,"sed -e \'s/PACKAGE_NAME/%s/g\'\
                       -e \'s/NEW_MODULE/%s/g\'\
                       -e \'s/MODULE_AUTHOR/%s/g\'\
                       -e \'s/MODULE_DATE/%s/g\'\
                   %s/Modules/%s/%s.in > %s/Modules/%s/%s.cc",
                   pac,mod,getenv("USER"),"<TODAYS DATE HERE>",
                   pac,cat,mod,pac,cat,mod);
  system(command);

  /* edit the makefile (sub.mk) to include the new module */

  sprintf(command,"mv ./%s/Modules/%s/sub.mk ./%s/Modules/%s/sub.in",
                  pac,cat,pac,cat);
  system(command);

  sprintf(command,"sed -e \'s/PACKAGE_NAME/%s/g\'"
                  "    -e \'s/CATEGORY_NAME/%s/g\'"
                  "    -e \'s/#\\[INSERT NEW MODULE HERE\\]/"
	  	  "	$(SRCDIR)\\/%s.cc\\\\\\\n"
                  "#\\[INSERT NEW MODULE HERE\\]/g\'"
                  " %s/Modules/%s/sub.in > %s/Modules/%s/sub.mk",
	          pac,cat,mod,pac,cat,pac,cat);
  system(command);
  
  /* edit the components.xml file to include an entry for the new module */

  sprintf(command,"mv ./%s/components.xml ./%s/components.in",pac,pac);
  system(command);

  sprintf(command,"sed -e \'s/\\<!-- INSERT NEW %s COMPONENT HERE -->/"
                  "\\<dataflow-component name=\"%s\"\\>\\\n"
                  "    \\<meta\\>\\\n"
                  "      \\<authors\\>\\\n"
                  "        \\<author\\>%s\\<\\/author\\>\\\n"
                  "      \\<\\/authors\\>\\\n"
                  "      \\<version-date\\>\\[TODAYS DATE HERE\\]"
                  "\\<\\/version-date\\>\\\n"
                  "      \\<version\\>1.0\\<\\/version\\>\\\n"
                  "      \\<description\\>No description available"
                  "\\<\\/description\\>\\\n"
                  "    \\<\\/meta\\>\\\n"
                  "    \\<inputs\\>\\<\\/inputs\\>\\\n"
                  "    \\<outputs\\>\\<\\/outputs\\>\\\n"
                  "    \\<parameters\\>\\<\\/parameters\\>\\\n"
	          "    \\<implementation\\>\\\n"
                  "      \\<creationFunction\\>make_%s"
                  "\\<\\/creationFunction\\>\\\n"
                  "    \\<\\/implementation\\>\\\n"
                  "  \\<\\/dataflow-component\\>\\\n"
                  "\\\n"
                  "  \\<!-- INSERT NEW %s COMPONENT HERE -->/g\'"
                  " %s/components.in > %s/components.xml",
                  cat,mod,getenv("USER"),mod,cat,pac,pac);
  system(command);

  /* delete the intermediate files */

  sprintf(command,"find ./%s -name \"*.in\" | xargs rm",pac);
  system(command);

  return 1;
}

#else

int CreatePac(const char*)
{
  return 0;
}  

int CreateCat(const char*, const char*)
{
  return 0;
}  

int CreateMod(const char*, const char*, const char*)
{
  return 0;
}

#endif

} // Dataflow
} // PSECore

