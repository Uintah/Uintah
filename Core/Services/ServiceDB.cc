/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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


// ServiceDB.cc - Interface to module-finding and loading mechanisms


#include <Dataflow/XMLUtil/StrX.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <Core/Services/ServiceNode.h>
#include <Core/Services/ServiceDB.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Dataflow/Network/FileUtils.h>
#include <stdlib.h>
#include <Core/Util/sci_system.h>

#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <string>
#include <vector>
#include <sys/stat.h>


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif


#ifdef __APPLE__
static std::string lib_ext = ".dylib";
#else
static std::string lib_ext = ".so";
#endif

namespace SCIRun {

// Constructor fo this class
ServiceDB::ServiceDB() :
  lock("service database lock"), ref_cnt(0)
{
}

ServiceDB::~ServiceDB() 
{ 
  // Free all memory we allocated in generating the information
  // database on available services
  std::map<std::string,ServiceInfo *>::iterator  pi;
  for (pi = servicedb_.begin(); pi!=servicedb_.end(); pi++) 
    {
      if((*pi).second)
        {
          delete (*pi).second;
          (*pi).second = 0;
        }
    }
}

// This function was copied from the package database
// files.

LIBRARY_HANDLE ServiceDB::findlib(string lib) 
{
  LIBRARY_HANDLE handle=0;
  const char *env = sci_getenv("PACKAGE_LIB_PATH");
  std::string temppaths(env?env:"");
  
  // try to find the library in the specified path
  while (temppaths!="") 
    {
      std::string dir;
      const size_t firstcolon = temppaths.find(':');
      if(firstcolon < temppaths.size()) 
      {
          dir=temppaths.substr(0,firstcolon);
          temppaths=temppaths.substr(firstcolon+1);
      } 
      else 
      {
          dir=temppaths;
          temppaths="";
      }

      handle = GetLibraryHandle((dir+"/"+lib).c_str());
      if (handle)	return(handle);
    }

  // if not yet found, try to find it in the rpath 
  // or the LD_LIBRARY_PATH (last resort)
  
  handle = GetLibraryHandle(lib.c_str());
  return(handle);
}


bool ServiceDB::findmaker(ServiceInfo* info) 
{
  std::string cat_bname, pak_bname;
  if(info->classpackagename == "SCIRun") 
  {
      pak_bname = "Core_Services";
  }	
  else 
  {
      pak_bname = "Packages_" + info->classpackagename + "_Services";
  }
  
  std::string errstr;

  // try the large version of the shared library
  LIBRARY_HANDLE package_so = findlib("lib" + pak_bname+lib_ext);
  if (!package_so) errstr = string(" - ")+SOError()+string("\n");

  if (!package_so) 
  {
      std::cerr << "Unable to load services of package '"  << info->classpackagename << "'\n :" << errstr;
      return(false);
  }

  std::string makename = "make_service_" + info->classname;
  if (package_so)
  {
      info->maker = (ServiceMaker) GetHandleSymbolAddress(package_so,makename.c_str());
  }
  if (!info->maker) 
  {
      std::cerr << "Unable to load service '" << info->servicename << "' :\n - cannot find symbol '" + makename + "'\n";
      return(false);
  }
  return(true);
}


void ServiceDB::loadpackages() 
{

  // Inform the user on what the program is doing
  std::cout << "Loading services, please wait..." << std::endl;

  // Find the path to the package directory
	
  std::string packagepath;
	
  const char *srcdir = sci_getenv("SCIRUN_SRCDIR");
  packagepath = srcdir + std::string("/Packages");

  // if the user specifies it, build the complete package path
  const char *packpath = sci_getenv("PACKAGE_SRC_PATH");
  if (packpath) packagepath = string(packpath) + ":" + packagepath;


  // the format of LOAD_PACKAGE is a comma seperated list of package names.
  // build the complete list of packages to load
  ASSERT(sci_getenv("SCIRUN_LOAD_PACKAGE"));
  std::string loadpackage = std::string(sci_getenv("SCIRUN_LOAD_PACKAGE"));
  std::string packageelt;
	
  while(loadpackage!="") 
  {
      // Strip off the first element, leave the rest for the next
      // iteration.
      const size_t firstcomma = loadpackage.find(',');
      if(firstcomma < loadpackage.size()) 
      {
          packageelt=loadpackage.substr(0,firstcomma);
          loadpackage=loadpackage.substr(firstcomma+1);
      } 
      else 
      {
          packageelt=loadpackage;
          loadpackage="";
      }

      std::string tmppath = packagepath;
      std::string pathelt;

      while (tmppath!="") 
      {
          if (packageelt=="SCIRun") 
          {
              tmppath = "found";
              break;
          }
          const size_t firstcolon = tmppath.find(':');
          if(firstcolon < tmppath.size()) 
          {
              pathelt=tmppath.substr(0,firstcolon);
              tmppath=tmppath.substr(firstcolon+1);
          } 
          else 
          {
              pathelt=tmppath;
              tmppath="";
          }
      
          struct stat buf;
          lstat((pathelt+"/"+packageelt).c_str(),&buf);
          if (S_ISDIR(buf.st_mode)) 
          {
              tmppath = "found";
              break;
          }
      }

      if (tmppath=="") 
      {
          std::cerr << "Unable to load package " << packageelt << ":\n - Can't find " << packageelt << " directory in package path" << std::endl;
          continue;
      }

      std::string xmldir;
    
      if(packageelt == "SCIRun") 
      {
          xmldir = std::string(srcdir) + "/Core/Services";
      } 
      else 
      {
          xmldir = pathelt+"/"+packageelt+"/Services";
      }
      std::map<int,char*>* files;
      files = GetFilenamesEndingWith((char*)xmldir.c_str(),".xml");

      if (!files) 
      {
          // THIS PACKAGE APPARENTLY DOES NOT CONTAIN ANY SERVICES
          continue;
      }

      for (std::map<int,char*>::iterator i=files->begin(); i!=files->end(); i++) 
      {
          ServiceNode node;
          ReadServiceNodeFromFile(node,(xmldir+"/"+(*i).second).c_str());

          if (node.servicename == "") continue;
		
          ServiceInfo* new_service = scinew ServiceInfo;
		
          // zero the non object entries
          new_service->activated = false;
          new_service->packagename = packageelt;
          new_service->maker = 0;
          new_service->servicename = node.servicename;
          new_service->classname = node.classname;
          new_service->disabled = false;
          if (node.classpackagename == "")
          {
            new_service->classpackagename = packageelt;
          }
          else
          { // This options allows to define a services whose class
            // lives in a different package. Mostly in the SCIRun main
            // class.
            new_service->classpackagename = node.classpackagename;
          } 
          new_service->parameters = node.parameter;
          new_service->version = node.version;
			
          parse_and_find_service_rcfile(new_service,xmldir);

          // Now the XML and RC file are loaded use these to set up some 
          // other features. Mainly preprocess data so we do not have to do
          // that every time a connection has to be made
			
          // Compile a list of hosts that will be allowed to conenct to this
          // socket.
          std::string patternlist = new_service->parameters["rhosts"];
          if (patternlist != "") new_service->rhosts.insert(patternlist);

          std::string password = new_service->parameters["password"];
          if (password != "") new_service->passwd = password;

          std::string disabled = new_service->parameters["disabled"];
          if ((disabled=="1")||(disabled=="true")||(disabled=="TRUE")||(disabled=="yes")||(disabled=="YES")) new_service->disabled =true;

          // Store the information so all threads can use it
          servicedb_[node.servicename] = new_service;
      }
  }

  std::map<std::string,ServiceInfo *>::iterator  pi;
  for (pi = servicedb_.begin(); pi!=servicedb_.end(); pi++) 
  {
      if(!findmaker((*pi).second))
      {
          std::cerr << "Unable to load service '" << (*pi).second->servicename <<
            "' :\n - can't find symbol 'make_service_" << (*pi).second->classname << "'" << std::endl;			
      }
  }
}
  

void ServiceDB::parse_and_find_service_rcfile(ServiceInfo *new_service,std::string xmldir)
{
  // If no default rcfile is specified, this service does not have one
  // Hence we do not have to load and parse an rcfile
	
  std::string rcname = new_service->parameters["rcfile"];
  if (rcname == "") return;
	
  // First check whether the SCIRun/services directory exists in the HOME directory

  std::string filename;
  bool foundrc = false;
  
  // 1. check the user's home/SCIRun/services directory
  char *HOME;
  if (HOME = getenv("HOME")) 
  {
      filename = HOME+std::string("/SCIRun/services/") + rcname;
      foundrc = parse_service_rcfile(new_service,filename);
      if (foundrc) return;
      filename = HOME+std::string("/SCIRun/") + rcname;
      foundrc = parse_service_rcfile(new_service,filename);
      if (foundrc) return;
      filename = HOME+std::string("/") + rcname;
      foundrc = parse_service_rcfile(new_service,filename);
      if (foundrc) return;
  }
  
  // 2. check sourcedir where xml file was located as well

  filename = xmldir + std::string("/") + rcname;
  foundrc = parse_service_rcfile(new_service,filename);
  if (!foundrc) 
  {
      std::cerr << "Could not locate the configuration file '" << rcname << "' for service " << new_service->servicename << std::endl;	
      return;
  }
	
  // If this one is a success 
  // try to copy this rc file to the SCIRun directory in the
  // users home directory
	
  if (HOME = getenv("HOME")) 
  {
      std::string dirname = HOME+std::string("/SCIRun/services");
      struct stat buf;
      if(::lstat(dirname.c_str(),&buf) < 0)
      {
          // The target directory does not exist
			
          // Test whether SCIRun directory exists
          // if not create it

          dirname = HOME+std::string("/SCIRun");
          if (::lstat(dirname.c_str(),&buf) < 0)
          {
              std::string cmd = std::string("mkdir ") + std::string(HOME) + std::string("/SCIRun");
              sci_system(cmd.c_str());
              if(::lstat(dirname.c_str(),&buf) < 0) return;  // Try to get information once more
          }
          if (!S_ISDIR(buf.st_mode))
          {
              std::cerr << "Could not create " << dirname << " directory" << std::endl; 
              return;
          }
		
          // SCIRun directory must exist now
		
          dirname = HOME+std::string("/SCIRun/services");
          if (::lstat(dirname.c_str(),&buf) < 0)
          {
              std::string cmd = std::string("mkdir ") + std::string(HOME) + std::string("/SCIRun/services");
              sci_system(cmd.c_str());
              if(::lstat(dirname.c_str(),&buf) < 0) return;  // Try to get information once more
          }
          if (!S_ISDIR(buf.st_mode))
          { 
              std::cerr << "Could not create " << dirname << " directory" << std::endl; 
              return;
          }
        }
		
      if (!S_ISDIR(buf.st_mode))
      {
          std::cerr << "Could not create " << dirname << " directory" << std::endl; 
          return;
      }

      std::cout << "Copying " << rcname << " to local /SCIRun/services directory" << std::endl;
      std::string cmd = std::string("cp -f ") + filename + " " + std::string(HOME) + std::string("/SCIRun/services/") + rcname;
      sci_system(cmd.c_str());
		
  }

  return;
}

bool ServiceDB::parse_service_rcfile(ServiceInfo *new_service,std::string filename)
{
  FILE* filein = fopen(filename.c_str(),"r");
  if (!filein) return(false);

  std::string read_buffer;
  size_t read_buffer_length = 512;
  read_buffer.resize(read_buffer_length);

  bool done = false;
  bool need_new_read = true;
	
  std::string linebuffer;
  size_t bytesread;
  size_t linestart;
  size_t lineend;
  size_t lineequal;
  size_t buffersize;
  size_t linedata;
  size_t linesize;
  size_t linetag;
  std::string tag;
  std::string data;
	
  while(!done)
  {
      bytesread = fread(&(read_buffer[0]),1,read_buffer_length,filein);
      if ((bytesread == 0)&&(!feof(filein)))
      {
          std::cerr << "Detected error while reading from file: " << filename << std::endl;
          return(false);
      }
      if (bytesread > 0) linebuffer += read_buffer.substr(0,bytesread);

      if (feof(filein)) done = true;
	
      need_new_read = false;
      while (!need_new_read)
      {	
          linestart = 0;
          buffersize = linebuffer.size();
          // Skip all newlines returns tabs and spaces at the start of a line
          while((linestart < buffersize)&&((linebuffer[linestart]=='\n')||(linebuffer[linestart]=='\r')||(linebuffer[linestart]=='\0')||(linebuffer[linestart]=='\t')||(linebuffer[linestart]==' '))) linestart++;			

          std::string newline;
          // if bytesread is 0, it indicates an EOF, hence we just need to add the remainder
          // of what is in the buffer. The file has not properly terminated strings....
          if (bytesread == 0)
          {	
              if(linestart < linebuffer.size()) newline = linebuffer.substr(linestart);
          }
          else
          {
              lineend = linestart;
              while((lineend < buffersize)&&(linebuffer[lineend]!='\n')&&(linebuffer[lineend]!='\r')&&(linebuffer[lineend]!='\0')) lineend++;
              if (lineend == linebuffer.size())
              {	// end of line not yet read
                  need_new_read = true;
              }
              else
              {	// split of the latest line read
                  newline = linebuffer.substr(linestart,lineend);
                  linebuffer = linebuffer.substr(lineend+1);
                  need_new_read = false;
              }
          }
			
          if (!need_new_read)
          {
              if ((newline[0] == '#')||(newline[0] == '%'))
              {   // Comment
                  if (newline.substr(0,11) == "###VERSION=")
                    new_service->rcfileversion = newline.substr(11);
              }
              else
              {
                  lineequal = 0;
                  linetag = 0;
                  linesize = newline.size();
                  while((linetag<linesize)&&((newline[linetag] != ' ')&&(newline[linetag] != '\t')&&(newline[linetag] != '='))) linetag++;
                  tag = newline.substr(0,linetag);
                  lineequal = linetag;
                  while((lineequal < linesize)&&(newline[lineequal] != '=')) lineequal++;
                  linedata = lineequal+1;
                  while((linedata < linesize)&&((newline[linedata] == ' ')||(newline[linedata] == '\t'))) linedata++;
                  data = newline.substr(linedata);
                  new_service->parameters[tag] = data;
              }
          }
      }
  }
			
  // Keep this for debugging purposes, it will end up in the log file
  new_service->rcfile = filename;

  fclose(filein);
  return(true);	
}



void ServiceDB::activateall()
{
  std::map<std::string,ServiceInfo *>::iterator  pi;
  for (pi = servicedb_.begin(); pi!=servicedb_.end(); pi++) 
  {
      if((*pi).second->maker)
      {
          (*pi).second->activated = true;
      }
  }
}

void ServiceDB::deactivateall()
{
  std::map<std::string,ServiceInfo *>::iterator  pi;
  for (pi = servicedb_.begin(); pi!=servicedb_.end(); pi++) 
  {
      if((*pi).second->maker)
      {
          (*pi).second->activated = false;
      }
    }
}

void ServiceDB::activate(std::string name)
{
  if(servicedb_[name])
  {
      servicedb_[name]->activated = true;
  }
}

void ServiceDB::deactivate(std::string name)
{
  if(servicedb_[name])
  {
      servicedb_[name]->activated = false;
  }
}


bool ServiceDB::isservice(const std::string servicename)
{
  if(servicedb_[servicename]) 
  {
      if (servicedb_[servicename]->activated == false) return(false);
      return(true);
  }
  return(false);
}


ServiceInfo* ServiceDB::getserviceinfo(const std::string servicename)
{
  return(servicedb_[servicename]);
}

 
ServiceDB* ServiceDB::clone()
{
  ServiceDB* newptr = scinew ServiceDB();
	
  if (newptr == 0)
  {
      std::cerr << "Could not clone Service Database" << std::endl;
      return(0);
  }
	
  std::map<std::string,ServiceInfo*>::iterator  pi;
  for (pi = servicedb_.begin(); pi!=servicedb_.end(); pi++) 
  {
	
      ServiceInfo *info = scinew ServiceInfo;
      *info = *((*pi).second);
      newptr->servicedb_[info->servicename] = info;
  } 
	
  return(newptr);
}

void ServiceDB::printservices()
{
    std::cout << "list of available services:" << std::endl;
    std::map<std::string,ServiceInfo*>::iterator it;
    it = servicedb_.begin();
    for ( ;it !=servicedb_.end(); it++)
    {
        ServiceInfo *info = (*it).second;
        std::cout << " " << info->servicename << "(" << info->packagename << ")  version=" << info->version << " rcfile=" << info->rcfile << " disabled=" << info->disabled << std::endl; 
    }
}

}
  
