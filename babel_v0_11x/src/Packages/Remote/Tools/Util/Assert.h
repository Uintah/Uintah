/*
 *  Assert.h: Utility for specifying data invariants (Assertions)
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 *  Hacked by DaveMc, 1998.
 */

#ifndef _assert_h
#define _assert_h

#include <iostream>

namespace Remote {
using namespace std;
// This assertion always causes a fatal error. It doesn't depend on debug mode.
#define ASSERTERR(condition,msg) {if(!(condition)){cerr << "Fatal Error: (" << (#condition) << ") at " << __FILE__ <<":" << __LINE__ << " " << (#msg) << endl; exit(1);}}

// This kind does not depend on debug mode.
#define ASSERT0(condition) {if(!(condition)){cerr << "Assertion Failed: (" << (#condition) << ") at " << __FILE__ <<":" << __LINE__ << endl; exit(1);}}

// These two depend on debug mode.
#if SCI_ASSERTION_LEVEL >= 1
#define ASSERT1(condition) {if(!(condition)){cerr << "Assertion1 Failed: (" << (#condition) << ") at " << __FILE__ <<":" << __LINE__ << endl; while(1);}}
#define ASSERT1M(condition,msg) {if(!(condition)){cerr << "Assertion1 Failed: (" << (#condition) << ") at " << __FILE__ <<":" << __LINE__ << " " << (#msg) << endl; while(1);}}
#else
#define ASSERT1(condition)
#define ASSERT1M(condition,msg)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERT2(condition) {if(!(condition)){cerr << "Assertion2 Failed: (" << (#condition) << ") at " << __FILE__ <<":" << __LINE__ << endl; while(1);}}
#else
#define ASSERT2(condition)
#endif

#if SCI_ASSERTION_LEVEL == 0
#define ASSERT1(condition)
#define ASSERT1M(condition,msg)
#define ASSERT2(condition)
#endif

#define ASSERT(condition) ASSERT2(condition)

/*
#if SCI_ASSERTION_LEVEL >= 0
#define GL_ASSERT() {GLenum sci_err; while \
  ((sci_err = glGetError()) != GL_NO_ERROR) \
    cerr << "OpenGL error: " << (char *)gluErrorString(sci_err) \
	 << " at " << __FILE__ <<":" << __LINE__ << endl;}
#else
*/
#define GL_ASSERT()
//#endif

} // End namespace Remote


#endif
