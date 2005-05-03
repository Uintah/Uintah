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

//    File   : DynamicLoader.cc
//    Author : Martin Cole
//    Date   : Mon May 21 10:57:38 2001

#include <sci_defs/compile_defs.h>

#include <Core/Util/DynamicLoader.h>
#include <Core/Util/soloader.h>
#include <Core/Util/sci_system.h>
#include <Core/Util/Environment.h>
#include <Core/Containers/StringUtil.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <sgi_stl_warnings_off.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace SCIRun {

using namespace std;

#ifdef __APPLE__
  // This mutex is used in Core/Persistent/Persistent.cc.  It is
  // declared here because it is not initializing properly when declared
  // in Persistent.cc.
  Mutex persistentTypeIDMutex("Persistent Type ID Table Lock");
  const string ext("dylib");
#elif defined(_WIN32)
  const string ext("dll");
#else
  const string ext("so");
#endif

  // Need these mutexes to be created here for use in another library
  // as they must "construct" _now_ because they are used before the
  // library that would normally construct them gets loaded...
  Mutex colormapIEPluginMutex("ColorMap Import/Export Plugin Table Lock");
  Mutex fieldIEPluginMutex("Field Import/Export Plugin Table Lock");
  Mutex matrixIEPluginMutex("Matrix Import/Export Plugin Table Lock");

DynamicLoader *DynamicLoader::scirun_loader_ = 0;
Mutex DynamicLoader::scirun_loader_init_lock_("SCIRun loader init lock");


CompileInfo::CompileInfo(const string &fn, const string &bcn,
			 const string &tcn, const string &tcdec) :
  filename_(fn),
  base_class_name_(bcn),
  template_class_name_(tcn),
  template_arg_(tcdec),
  post_include_extra_(""),
  ref_cnt(0)
{
}


void
CompileInfo::add_include(const string &inc)
{
  //std::remove(includes_.begin(), includes_.end(), inc);
  includes_.push_front(inc);
}


void
CompileInfo::add_post_include(const string &post)
{
  post_include_extra_ = post_include_extra_ + post;
}


//! CompileInfo::create_cc
//!
//! Generate the code for a .cc file from the compile info.
//! If boolean empty == true, It contains an empty maker function.
//! Used if the actual compilation fails.
void
CompileInfo::create_cc(ostream &fstr, bool empty) const
{
  const string STD_STR("std::");

  fstr << "// This is an automatically generated file, do not edit!" << endl;

  // generate standard includes
  list<string>::const_iterator iter = includes_.begin();
  while (iter != includes_.end())
  {
    const string &s = *iter;
    if (s.substr(0, 5) == STD_STR)
    {
      string std_include = s.substr(5, s.length() -1);
      fstr << "#include <" << std_include << ">" << endl;
    }
    ++iter;
  }

  ASSERT(sci_getenv("SCIRUN_SRCDIR"));
  const std::string srcdir(sci_getenv("SCIRUN_SRCDIR"));
  // Generate other includes.
  iter = includes_.begin();
  while (iter != includes_.end())
  {
    const string &s = *iter;

    if (!((s.substr(0, 5) == STD_STR) || s == "builtin"))
    {
      string::size_type loc = s.find(srcdir);
      if( loc != string::npos )
      {
	string::size_type endloc = loc+srcdir.size()+1;
	fstr << "#include <" << s.substr(endloc) << ">\n";
      }
      else
      {
	// when using TAU, we will have the prefix .inst.h instead of .h
        // we fix it here
        if (s.find(".inst.") != string::npos)
        {
          int pos = s.find(".inst.");
          string newString = s;
          newString.replace(pos,6,".");
          fstr << "#include \"" << newString << "\"\n";
        }
        else
        {
          fstr << "#include \"" << s << "\"\n";
        }
      }
    }
    ++iter;
  }

  // output namespaces
  ci_map_type::const_iterator nsiter = namespaces_.begin();
  while (nsiter != namespaces_.end())
  {
    const string &s = (*nsiter).first;
    if (s != "builtin")
    {
      fstr << "using namespace " << s << ";" << endl;
    }
    ++nsiter;
  }

  // Add in any post_include construction, usually specific class instances.
  if (post_include_extra_ != "")
  {
    fstr << "\n" << post_include_extra_ << "\n";
  }

  // Delcare the maker function
  fstr << endl << "extern \"C\" {"  << endl
       << base_class_name_ << "* maker() {" << endl;

  // If making an empty maker, return nothing instead of newing up the class.
  // Comments out the next line that news up the class.
  if (empty)
  {
    fstr << "  return 0;" << endl << "//";
  }

  fstr << "  return scinew " << template_class_name_ << "<"
       << template_arg_ << ">;" << endl
       << "}" << endl << "}" << endl;
}


DynamicAlgoBase::DynamicAlgoBase() :
  ref_cnt(0),
  lock("DynamicAlgoBase ref_cnt lock")
{
}


DynamicAlgoBase::~DynamicAlgoBase()
{
}


string
DynamicAlgoBase::to_filename(const string s)
{
  string result;
  for (unsigned int i = 0; i < s.size(); i++)
  {
    if (isalnum(s[i]))
    {
      result += s[i];
    }
  }
  return result;
}


DynamicLoader::DynamicLoader() :
  map_crowd_("DynamicLoader: One compilation at a time."),
  compilation_cond_("DynamicLoader: waits for compilation to finish."),
  map_lock_("DynamicLoader: controls mutable access to the map.")
{
  map_lock_.lock();
  algo_map_.clear();
  map_lock_.unlock();
}


DynamicLoader::~DynamicLoader()
{
  map_lock_.lock();
  algo_map_.clear();
  map_lock_.unlock();
}


//! DynamicLoader::scirun_loader
//!
//! How to get at the global loader for scirun.
DynamicLoader&
DynamicLoader::scirun_loader()
{
  if (scirun_loader_)
  {
    return *scirun_loader_;
  }
  else
  {
    scirun_loader_init_lock_.lock();
    if (!scirun_loader_)
    {
      scirun_loader_ = new DynamicLoader();
    }
    scirun_loader_init_lock_.unlock();
    return *scirun_loader_;
  }
}


//! DynamicLoader::entry_exists
//!
//! Convenience function to query the map, but thread safe.
bool
DynamicLoader::entry_exists(const string &entry)
{
  map_lock_.lock();
  bool rval = algo_map_.find(entry) != algo_map_.end();
  map_lock_.unlock();
  return rval;
}


//! DynamicLoader::entry_is_null
//!
//! Convenience function to query the value in the map, but thread safe.
bool
DynamicLoader::entry_is_null(const string &entry)
{
  map_lock_.lock();
  bool rval =  (algo_map_.find(entry) != algo_map_.end() &&
		algo_map_[entry] == 0);
  map_lock_.unlock();
  return rval;
}


//! DynamicLoader::wait_for_current_compile
//!
//! Block if the lib associated with entry is compiling now.
//! The only way false is returned, is for the compile to fail.
bool
DynamicLoader::wait_for_current_compile(const string &entry)
{
  while (entry_is_null(entry))
  {
    // another thread is compiling this lib, so wait.
    map_lock_.lock();
    compilation_cond_.wait(map_lock_);
    map_lock_.unlock();
  }
  // if the map entry no longer exists, compilation failed.
  if (! entry_exists(entry)) return false;
  // The maker fun has been stored by another thread.
  ASSERT(! entry_is_null(entry));
  return true;
}


//! DynamicLoader::compile_and_store
//!
//! Compile and store the maker function mapped to the lib name.
//! The sychronization code allows multiple threads to compile different
//! libs at the same time, but forces only one thread can compile any one
//! lib.
bool
DynamicLoader::compile_and_store(const CompileInfo &info, bool maybe_compile_p,
				 ProgressReporter *pr)
{
  bool do_compile = false;

  if (! entry_exists(info.filename_))
  {
    // first attempt at creation of this lib
    map_crowd_.writeLock();
    if (! entry_exists(info.filename_))
    {
      // create an empty entry, to catch threads chasing this one.
      map_lock_.lock();
      algo_map_[info.filename_] = 0;
      map_lock_.unlock();
      do_compile = true; // this thread is compiling.
    }
    map_crowd_.writeUnlock();
  }

  if (!do_compile && !wait_for_current_compile(info.filename_)) return false;

  // Try to load a dynamic library that is already compiled
  string full_so = otf_dir() + string("/") + info.filename_ + ext;

  LIBRARY_HANDLE so = 0;
  struct stat buf;
  if (stat(full_so.c_str(), &buf) == 0)
  {
    so = GetLibraryHandle(full_so.c_str());
  }
  else
  {
    // the lib does not exist.
    create_cc(info, false, pr);
    if (compile_so(info, pr))
    {
      so = GetLibraryHandle(full_so.c_str());
    }
    if (maybe_compile_p && so == 0)
    {
      create_cc(info, true, pr);
      compile_so(info, pr);
      so = GetLibraryHandle(full_so.c_str());
    }

    if (so == 0)
    { // does not compile
      pr->error("DYNAMIC COMPILATION ERROR: " + full_so +
                " does not compile!!");
      pr->msgStream() << SOError() << endl;
      // Remove the null ref for this lib from the map.
      map_lock_.lock();
      algo_map_.erase(info.filename_);
      map_lock_.unlock();
      // wake up all sleepers.
      compilation_cond_.conditionBroadcast();
      return false;
    }
  }

  maker_fun maker = 0;
  maker = (maker_fun)GetHandleSymbolAddress(so, "maker");

  if (maker == 0)
  {
    pr->error("DYNAMIC LIB ERROR: " + full_so +
              " no maker function!!");
    pr->msgStream() << SOError() << endl;
    // Remove the null ref for this lib from the map.
    map_lock_.lock();
    algo_map_.erase(info.filename_);
    map_lock_.unlock();
    // wake up all sleepers.
    compilation_cond_.conditionBroadcast();
    return false;
  }
  // store this so that we can get at it again.
  store(info.filename_, maker);
  // wake up all sleepers.
  compilation_cond_.conditionBroadcast();
  return true;
}



//! DynamicLoader::compile_so
//!
//! Attempt to compile file into a dynamic library, return true if it succeeded
//! false otherwise.
bool
DynamicLoader::compile_so(const CompileInfo &info, ProgressReporter *pr)
{
  string command = ("cd " + otf_dir() + "; " + MAKE_COMMAND + " " +
		    info.filename_ + ext);

  pr->msgStream() << "DynamicLoader - Executing: " << command << endl;

  FILE *pipe = 0;
  bool result = true;
#ifdef __sgi
  command += " 2>&1";
  pipe = popen(command.c_str(), "r");
  if (pipe == NULL)
  {
    pr->remark("DynamicLoader::compile_so() syscal error unable to make.");
    result = false;
  }
#else
  command += " > " + info.filename_ + "log 2>&1";

#ifdef _WIN32
  // we need to create a separate process here, because TCL's interpreter is active.
  // For some reason, calling make in 'system' will hang until the interpreter closes.
  // the batch file is for convenience of not having to create files manually

  STARTUPINFO si_;
  PROCESS_INFORMATION pi_;

  memset(&si_, 0, sizeof(si_));
  memset(&pi_, 0, sizeof(pi_));

  DWORD status = 1;

  HANDLE logfile;
  char logfilename[256];
  char otfdir[256];
  strcpy(otfdir, otf_dir().c_str());
  // We need to make Windows create a command and read it since windows
  // system can't handle a ';' to split commands
  int loc = command.find(';');

  // give it \ instead of / so windows can read it correctly
  string command1 = command.substr(0, loc);
  for (unsigned i = 0; i < command1.length(); i++)
  {
    if (command1[i] == '/')
    {
      command1[i] = '\\';
    }
  }
  for (unsigned i = 0; i < strlen(otfdir); i++)
  {
    if (otfdir[i] == '/')
    {
      otfdir[i] = '\\';
    }
  }
  string command2 = command.substr(loc+1, command.length());

  string batch_filename = string(otfdir)+"\\" + info.filename_ + "bat";
  FILE* batch = fopen(batch_filename.c_str(), "w");
  fprintf(batch, "\n%s\n%s\n", command1.c_str(), command2.c_str());
  fclose(batch);

  si_.cb = sizeof(STARTUPINFO);

  // the CREATE_NO_WINDOW is so the process will not run in the same shell.
  // Otherwise it will get confused while trying to run in the same shell as the
  // tcl interpreter.
  bool retval =
    CreateProcess(batch_filename.c_str(),0,0,0,0,CREATE_NO_WINDOW,0,0, &si_, &pi_);
  if (!retval)
  {
    cerr << SOError() << "\n";
  }
  else
  {
    WaitForSingleObject(pi_.hProcess, INFINITE);
    GetExitCodeProcess(pi_.hProcess, &status);
    CloseHandle(pi_.hProcess);
    CloseHandle(pi_.hThread);
  }
#else
  const int status = sci_system(command.c_str());
#endif // def _WIN32
  if(status != 0)
  {
    pr->remark("DynamicLoader::compile_so() syscal error " +
	       to_string(status) + ": command was '" + command + "'.");
    result = false;
  }
  pipe = fopen(string(otf_dir() + "/" + info.filename_ + "log").c_str(), "r");
#endif // __sgi

  char buffer[256];
  while (pipe && fgets(buffer, 256, pipe) != NULL)
  {
    pr->msgStream() << buffer;
    pr->msgStream_flush();
  }

#ifdef __sgi
  if (pipe) { pclose(pipe); }
#else
  if (pipe) { fclose(pipe); }
#endif

  if (result)
  {
    pr->msgStream() << "DynamicLoader - Successfully compiled " <<
      info.filename_ << ext << endl;
  }
  return result;
}


void
DynamicLoader::cleanup_failed_compile(CompileInfoHandle info)
{
  if (sci_getenv_p("SCIRUN_NOCLEANUPCOMPILE")) { return; }

  const string base = otf_dir() + "/" + info->filename_;

  const string full_cc = base + "cc";
  unlink(full_cc.c_str());

  const string full_d = base + "d";
  unlink(full_d.c_str());

  const string full_log = base + "log";
  unlink(full_log.c_str());

  const string full_o = base + "o";
  unlink(full_o.c_str());

  const string full_so = base + "so";
  unlink(full_so.c_str());
}


//! DynamicLoader::create_cc
//!
//! Write a .cc file, from the compile info.
//! If boolean empty == true, It contains an empty maker function.
//! Used if the actual compilation fails.
bool
DynamicLoader::create_cc(const CompileInfo &info, bool empty,
                         ProgressReporter *pr)
{
  // Try to open the file for writing.
  string full = otf_dir() + "/" + info.filename_ + "cc";
  ofstream fstr(full.c_str());

  if (!fstr)
  {
    pr->error(string("DynamicLoader::create_cc(empty = ") +
              (empty ? "true":"false")
              + ") - Could not create file " + full + ".");
    return false;
  }

  info.create_cc(fstr, empty);

  pr->msgStream() << "DynamicLoader - Successfully created " << full << endl;
  return true;
}


void
DynamicLoader::store(const string &name, maker_fun m)
{
  map_lock_.lock();
  algo_map_[name] = m;
  map_lock_.unlock();
}


bool
DynamicLoader::fetch(const CompileInfo &ci, DynamicAlgoHandle &algo)
{
  bool rval = false;
  // block in case we get here while it is compiling.
  if (! wait_for_current_compile(ci.filename_)) return false;

  map_crowd_.readLock();
  map_lock_.lock();
  map_type::iterator loc = algo_map_.find(ci.filename_);
  if (loc != algo_map_.end())
  {
    maker_fun m = loc->second;
    algo = DynamicAlgoHandle(m());
    rval = true;
  }
  map_lock_.unlock();
  map_crowd_.readUnlock();
  return rval;
}


bool
DynamicLoader::get(const CompileInfo &ci, DynamicAlgoHandle &algo)
{
  ProgressReporter pr;
  return (fetch(ci, algo) ||
	  (compile_and_store(ci, false, &pr) && fetch(ci, algo)));
}


bool
DynamicLoader::maybe_get(const CompileInfo &ci, DynamicAlgoHandle &algo)
{
  ProgressReporter pr;
  return (fetch(ci, algo) ||
	  (compile_and_store(ci, true, &pr) && fetch(ci, algo)));
}


string
DynamicLoader::otf_dir()
{
  ASSERT(sci_getenv("SCIRUN_ON_THE_FLY_LIBS_DIR"));
  return string(sci_getenv("SCIRUN_ON_THE_FLY_LIBS_DIR"));
}

} // End namespace SCIRun
