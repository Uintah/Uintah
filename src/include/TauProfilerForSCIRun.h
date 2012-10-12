/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
// This file defines all the TAU profiling macros to be "nothing".
// If TAU is installed and SCIRun is configured to USE_TAU_PROFILING
// then the real TAU Profile.h file will be used.



/*
   Instructions for using TAU with PDT (note that the version numbers used in
   the example below will change in the future):


   # get and build pdt and tau:
   
   wget http://www.cs.uoregon.edu/research/paracomp/proj/pdtoolkit/Download/pdtoolkit-3.3.1.tar.gz
   tar -xzf pdtoolkit-3.3.1.tar.gz
   cd pdtoolkit-3.3.1
   ./configure
   make ; make install
   cd ..
   
   
   wget http://www.cs.uoregon.edu/research/paracomp/proj/tau/tauprofile/dist/tau_latest.tar.gz
   tar -xzf tau_latest.tar.gz
   cd tau-2.14.2.1
   ./configure -pdt=/path/to/pdtoolkit-3.3.1 -pthread
   make install -j
   cd ..
   
   # configure SCIRun with
   --with-tau=/path/to/tau-2.14.2.1/include/Makefile
   
   # clean and build SCIRun:
   gmake cleanreally
   gmake
   
   
   # At this point, you must clear the network before exiting to get profile data. 
   # You should find files named profile.X.X.X in your directory after clearing
   # the network.  Run /path/to/tau-2.14.2.1/ARCH/bin/paraprof (where ARCH is your 
   # architecture, e.g. i386_linux) to view the profile data. 
   # email amorris@cs.uoregon.edu or sameer@cs.uoregon.edu if you have questions
*/



#ifdef USE_TAU_PROFILING

//  RNJ - The following define is supplied
//        by the TAU makefile, but if it is
//        not, let's go ahead and define it.

#ifndef TAU_DOT_H_LESS_HEADERS
#  define TAU_DOT_H_LESS_HEADERS
#endif

#  include <Profile/Profiler.h>

#else

  // Dummy out all the macros...
#define TAU_MAPPING(stmt, group)
#define TAU_MAPPING_OBJECT(FuncInfoVar) 
#define TAU_MAPPING_LINK(FuncInfoVar, Group) 
#define TAU_MAPPING_PROFILE(FuncInfoVar) 
#define TAU_MAPPING_CREATE(name, type, key, groupname, tid) 
#define TAU_MAPPING_PROFILE_TIMER(Timer, FuncInfoVar, tid)
#define TAU_MAPPING_PROFILE_START(Timer, tid) 
#define TAU_MAPPING_PROFILE_STOP(tid) 
#define TAU_MAPPING_PROFILE_EXIT(msg, tid)  
#define TAU_MAPPING_PROFILE_SET_NODE(node, tid)  

#define TAU_PROFILE_INIT(argc, argv)

#define TAU_PROFILE(name, type, group)
#define TAU_PROFILE_TIMER(var, name, type, group)
#define TAU_PROFILE_START(var)
#define TAU_PROFILE_STOP(var)
#define TAU_PROFILE_STMT(stmt)
#define TAU_PROFILE_EXIT(msg)
#define TAU_PROFILE_INIT(argc, argv)
#define TAU_PROFILE_SET_NODE(node)
#define TAU_PROFILE_SET_CONTEXT(context)
#define TAU_PROFILE_CALLSTACK()

#define TAU_REGISTER_THREAD()
#define TAU_REGISTER_FORK(id, op)
#define TAU_DB_DUMP()

#define TAU_PHASE_CREATE_STATIC(var, name, type, group) 
#define TAU_PHASE_CREATE_DYNAMIC(var, name, type, group) 
#define TAU_PHASE_START(var) 
#define TAU_PHASE_STOP(var) 


#endif

