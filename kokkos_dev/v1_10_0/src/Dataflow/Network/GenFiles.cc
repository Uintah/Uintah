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

/* GenFiles.cc */

#include <stdio.h>
#include <sys/stat.h>
#include <string.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/Network/SkeletonFiles.h>
#include <Dataflow/Network/GenFiles.h>
#include <Dataflow/Network/FileUtils.h>

#define PERM S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH

namespace SCIRun {

  /*! these functions all assume that the files and/or directories
      about to be generated, do not already exist */ 

int GenPackage(char* packname, char* psepath)
{
  int check=0,checkall=0;
  char* string = 0;
  FILE* file = 0;

  /* allocate enough space to hold the largest path */
  string = new char[strlen(packname)+strlen(psepath)+50];

  /* create all directories associated with a package */
  sprintf(string,"%s/src/Packages/%s",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/Packages/%s/Dataflow",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/Packages/%s/Core",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/Packages/%s/Dataflow/Modules",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);
  
  sprintf(string,"%s/src/Packages/%s/Dataflow/GUI",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/Packages/%s/Dataflow/XML",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);
  
  sprintf(string,"%s/src/Packages/%s/share",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/Packages/%s/Core/Datatypes",psepath,packname);
  checkall |= check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  if (checkall) {
    printf("Could not create one or more directories.  Giving up.");
    return 0;
  }

  /* create all the non-directory files associated with a package */
  sprintf(string,"%s/src/Packages/%s/share/share.h",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,share_skeleton,packname,packname,packname,
	  packname,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/share/DllEntry.cc",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,dllentry_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,package_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/Dataflow/sub.mk",
	  psepath,packname);
  file = fopen(string,"w");
  fprintf(file,dataflow_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/Core/sub.mk",
	  psepath,packname);
  file = fopen(string,"w");
  fprintf(file,core_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/Dataflow/Modules/sub.mk",
	  psepath,packname);
  file = fopen(string,"w");
  fprintf(file,modules_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/Core/Datatypes/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,datatypes_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/Packages/%s/Dataflow/GUI/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,gui_submk_skeleton,packname,packname);
  fclose(file);

  delete[] string;

  return 1;
}

int GenCategory(char* catname, char* packname, char* psepath)
{
  int check;
  char* string = 0;
  FILE* file = 0;

  string = new char[strlen(packname)+strlen(psepath)+
		    strlen(catname)+50];

  /* create category directory */
  sprintf(string,"%s/src/Packages/%s/Dataflow/Modules/%s",
	  psepath,packname,catname);
  check = mkdir(string,PERM);
  if (check) {
    printf("could not create directory \"%s\".  Giving up.\n",string);
    return 0;
  }

  /* create category sub.mk file */
  sprintf(string,"%s/src/Packages/%s/Dataflow/Modules/%s/sub.mk",
	  psepath,packname,catname);
  file = fopen(string,"w");
  fprintf(file,category_submk_skeleton,packname,catname,packname);
  fclose(file);

  /* edit the modules sub.mk file - add the new category */

  char* modname = new char[strlen(catname)+50];
  sprintf(modname,"\t$(SRCDIR)/%s\\\n",catname);
  string = new char[strlen(psepath)+strlen(packname)+50];
  sprintf(string,"%s/src/Packages/%s/Dataflow/Modules/sub.mk",
	  psepath,packname);
  InsertStringInFile(string,"#[INSERT NEW CATEGORY DIR HERE]",modname);
  delete[] string;

  return 1;
}

int GenComponent(component_node* n, char* packname, char* psepath)
{
  char* filename = 0;
  char* string = 0;
  FILE* file = 0;
  int length;

  /* generate a skeleton .cc file */
  length = strlen(n->name)+strlen(psepath)+
    strlen(packname)+strlen(n->category)+50;
  filename = new char[length];
  sprintf(filename,"%s/src/Packages/%s/Dataflow/Modules/%s/%s.cc",
	  psepath,packname,n->category,n->name);
  file = fopen(filename,"w");
  fprintf(file,component_skeleton,n->name,
	  getenv("USER"),"TODAY'S DATE HERE",
	  packname,packname,
	  packname,n->name,n->name,n->name,
	  n->name,n->name,n->name,n->name,
	  n->category,packname,n->name,n->name,
	  n->name,n->name,packname);
  fclose(file);
  delete[] filename;

  /* generate a full component .xml file */
  length = strlen(n->name)+strlen(psepath)+
    strlen(packname)+50;
  filename = new char[length];
  sprintf(filename,"%s/src/Packages/%s/Dataflow/XML/%s.xml",psepath,
	  packname,n->name);
  WriteComponentNodeToFile(n,filename);
  delete[] filename;

  if (n->gui->parameters->size()) {
    /* generate a skeleton .tcl file */
    length = strlen(n->name)+strlen(psepath)+
      strlen(packname)+50;
    filename = new char[length];
    sprintf(filename,"%s/src/Packages/%s/Dataflow/GUI/%s.tcl",psepath,
	    packname,n->name);
    file = fopen(filename,"w");
    fprintf(file,gui_skeleton,packname,n->category,n->name,n->name,filename);
    fclose(file);
    delete[] filename;
  }

  /* edit the category sub.mk file - add the new component */
  char* modname = new char[strlen(n->name)+50];
  sprintf(modname,"\t$(SRCDIR)/%s.cc\\\n",n->name);
  string = new char[strlen(psepath)+strlen(packname)+strlen(n->category)+50];
  sprintf(string,"%s/src/Packages/%s/Dataflow/Modules/%s/sub.mk",
	  psepath,packname,n->category);
  InsertStringInFile(string,"#[INSERT NEW CODE FILE HERE]",modname);
  delete[] string;
  delete[] modname;

  if (n->gui->parameters->size()) {
    /* edit the GUI sub.mk file - add the new component */
    modname = new char[strlen(n->name)+50];
    sprintf(modname,"\t$(SRCDIR)/%s.tcl\\\n",n->name);
    string = new char[strlen(psepath)+strlen(packname)+strlen(n->category)+50];
    sprintf(string,"%s/src/Packages/%s/Dataflow/GUI/sub.mk",psepath,packname);
    InsertStringInFile(string,"#[INSERT NEW TCL FILE HERE]",modname);
    delete[] string;
  }

  return 1;
}

} // End namespace SCIRun



