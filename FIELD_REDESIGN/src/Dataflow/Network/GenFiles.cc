/* GenFiles.cc */

#include <stdio.h>
#include <sys/stat.h>
#include <PSECore/Dataflow/ComponentNode.h>
#include <PSECore/Dataflow/SkeletonFiles.h>
#include <PSECore/Dataflow/GenFiles.h>
#include <PSECore/Dataflow/FileUtils.h>
#include <unistd.h>
#include <sys/wait.h>

#define PERM S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH

namespace PSECore {
namespace Dataflow {

void GenPackage(char* packname, char* psepath)
{
  int check;
  char* string = 0;
  FILE* file = 0;

  /* allocate enough space to hold the largest path */
  string = new char[strlen(packname)+strlen(psepath)+25];

  /* create all directories associated with a package */
  sprintf(string,"%s/src/%s",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/%s/Modules",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);
  
  sprintf(string,"%s/src/%s/GUI",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/%s/XML",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);
  
  sprintf(string,"%s/src/%s/share",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/%s/ThirdParty",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  sprintf(string,"%s/src/%s/Datatypes",psepath,packname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  /* create all the non-directory files associated with a package */
  sprintf(string,"%s/src/%s/share/share.h",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,share_skeleton,packname,packname,packname,
	  packname,packname);
  fclose(file);

  sprintf(string,"%s/src/%s/share/DllEntry.cc",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,dllentry_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/%s/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,package_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/%s/Modules/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,modules_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/%s/Datatypes/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,datatypes_submk_skeleton,packname);
  fclose(file);

  sprintf(string,"%s/src/%s/GUI/sub.mk",psepath,packname);
  file = fopen(string,"w");
  fprintf(file,gui_submk_skeleton,packname);
  fclose(file);

  delete[] string;
}

void GenCategory(char* catname, char* packname, char* psepath)
{
  int check;
  char* string = 0;
  FILE* file = 0;

  string = new char[strlen(packname)+strlen(psepath)+
		    strlen(catname)+25];

  /* create category directory */
  sprintf(string,"%s/src/%s/Modules/%s",psepath,packname,catname);
  check = mkdir(string,PERM);
  if (check)
    printf("could not create directory \"%s\"\n",string);

  /* create category sub.mk file */
  sprintf(string,"%s/src/%s/Modules/%s/sub.mk",psepath,packname,catname);
  file = fopen(string,"w");
  fprintf(file,category_submk_skeleton,packname,catname,packname);
  fclose(file);

  delete[] string;

  /* edit the modules sub.mk file - add the new category */
  /* fork a process to do the editing.  I'd do it in this process,
     but for some reason system() calls from this process 
     (SCIRun main process?) don't work under linux - Chris Moulding */

  if (fork()==0) {
    /* Only the child process gets in here */
    char* modname = new char[strlen(catname)+25];
    sprintf(modname,"\t%s\\\\\\\n",catname);
    string = new char[strlen(psepath)+strlen(packname)+25];
    sprintf(string,"%s/src/%s/Modules/sub.mk",psepath,packname);
    InsertStringInFile(string,"#\\[INSERT NEW CATEGORY DIR HERE\\]",modname);
    delete[] string;
    exit(0);
  } else
    wait(0); /* the parent process should wait for the child to exit */
}

void GenComponent(component_node* n, char* packname, char* psepath)
{
  char* filename = 0;
  char* string = 0;
  FILE* file = 0;
  int length;

  /* generate a skeleton .cc file */
  length = strlen(n->name)+strlen(psepath)+
    strlen(packname)+strlen(n->category)+19;
  filename = new char[length];
  sprintf(filename,"%s/src/%s/Modules/%s/%s.cc",psepath,packname,
	  n->category,n->name);
  file = fopen(filename,"w");
  fprintf(file,component_skeleton,n->name,
	  getenv("USER"),"TODAY'S DATE HERE",
	  packname,packname,
	  packname,n->name,n->name,n->name,
	  packname,n->name,n->name,n->name,
	  n->name,n->name,n->name,n->name,
	  n->name,n->name,packname);
  fclose(file);
  delete[] filename;

  /* generate a full component .xml file */
  length = strlen(n->name)+strlen(psepath)+
    strlen(packname)+15;
  filename = new char[length];
  sprintf(filename,"%s/src/%s/XML/%s.xml",psepath,
	  packname,n->name);
  WriteComponentNodeToFile(n,filename);
  delete[] filename;

  /* generate a skeleton .tcl file */
  length = strlen(n->name)+strlen(psepath)+
    strlen(packname)+15;
  filename = new char[length];
  sprintf(filename,"%s/src/%s/GUI/%s.tcl",psepath,
	  packname,n->name);
  file = fopen(filename,"w");
  fprintf(file,gui_skeleton,packname,n->category,n->name,n->name);
  fclose(file);
  delete[] filename;

  /* edit the category sub.mk file - add the new component */
  if (fork()==0) {
    /* Only the child process gets in here */
    char* modname = new char[strlen(n->name)+25];
    sprintf(modname,"\t$(SRCDIR)/%s.cc\\\\\\\n",n->name);
    string = new char[strlen(psepath)+strlen(packname)+strlen(n->category)+25];
    sprintf(string,"%s/src/%s/Modules/%s/sub.mk",psepath,packname,n->category);
    InsertStringInFile(string,"#\\[INSERT NEW CODE FILE HERE\\]",modname);
    delete[] string;
    delete[] modname;
    exit(0);
  } else
    wait(0); /* the parent process should wait for the child to exit */ 

  /* edit the GUI sub.mk file - add the new component */
  if (fork()==0) {
    /* Only the child process gets in here */
    char* modname = new char[strlen(n->name)+25];
    sprintf(modname,"\t$(SRCDIR)/%s.tcl\\\\\\\n",n->name);
    string = new char[strlen(psepath)+strlen(packname)+strlen(n->category)+25];
    sprintf(string,"%s/src/%s/GUI/sub.mk",psepath,packname);
    InsertStringInFile(string,"#\\[INSERT NEW TCL FILE HERE\\]",modname);
    delete[] string;
    exit(0);
  } else
    wait(0); /* the parent process should wait for the child to exit */ 
}

} // Dataflow
} // PSECore



