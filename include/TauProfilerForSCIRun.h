//
// This file defines all the TAU profiling macros to be "nothing".
// If TAU is installed and SCIRun is configured to USE_TAU_PROFILING
// then the real TAU Profile.h file will be used.
// 

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
#define TAU_DB_DUMP()

#endif

