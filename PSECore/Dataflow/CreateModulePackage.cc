/*
 *  CreateModulePackage.cc: 
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

int CreateNewPackage(const char* name)
{
  char* top = 0;
  char command[10000]="\0";

  sprintf(command,"cp -r ./scripts/NEW_PACKAGE %s",name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/sub.in > %s/sub.mk",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/components.in > %s/components.xml",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/share/share.in > %s/share/share.h",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/share/DllEntry.in > %s/share/DllEntry.cc",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/Datatypes/sub.in > %s/Datatypes/sub.mk",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/Datatypes/none/sub.in > %s/Datatypes/none/sub.mk",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/GUI/sub.in > %s/GUI/sub.mk",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/Modules/sub.in > %s/Modules/sub.mk",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/Modules/none/sub.in > %s/Modules/none/sub.mk",
                   name,name,name);
  system(command);
  sprintf(command,"sed -e \'s/NEW_PACKAGE/%s/\' \
                   %s/Modules/none/none.in > %s/Modules/none/none.cc",
                   name,name,name);
  system(command);

  sprintf(command,"find ./%s -name \"*.in\" | xargs rm",name);
  system(command);
  return 1;
}

int CreateNewModule(const char* name, const char* category)
{
  return 0;
}

#else
#endif

} // Dataflow
} // PSECore

