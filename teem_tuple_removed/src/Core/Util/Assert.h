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
 */

#ifndef SCI_Containers_Assert_h
#define SCI_Containers_Assert_h 1

#include <sci_config.h>

#include <Core/Exceptions/AssertionFailed.h>
#include <Core/Exceptions/ArrayIndexOutOfBounds.h>

#define ASSERTFAIL(string) \
   SCI_THROW(SCIRun::AssertionFailed(string, __FILE__, __LINE__));

#define ASSERTMSG(condition,message) \
   if(!(condition)){ \
      SCI_THROW(SCIRun::AssertionFailed(message, __FILE__, __LINE__)); \
   }

#if SCI_ASSERTION_LEVEL >= 1
#  define IFASSERT(x) x
#  define ASSERTL1(condition) \
     if(!(condition)){ \
        SCI_THROW(SCIRun::AssertionFailed(#condition, __FILE__, __LINE__)); \
     }
#else
#  define ASSERTL1(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#  define ASSERTL2(condition) \
     if(!(condition)){ \
        SCI_THROW(SCIRun::AssertionFailed(#condition, __FILE__, __LINE__)); \
     }
#else
#  define ASSERTL2(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 3
#  define ASSERTL3(condition) \
     if(!(condition)){ \
        SCI_THROW(SCIRun::AssertionFailed(#condition, __FILE__, __LINE__)); \
     }
#  define CHECKARRAYBOUNDS(value, lower, upper) \
     if(value < lower || value >= upper){ \
        SCI_THROW(SCIRun::ArrayIndexOutOfBounds(value, lower, upper)); \
     }
#else
#  define ASSERTL3(condition)
#  define CHECKARRAYBOUNDS(value, lower, upper)
#endif

#if SCI_ASSERTION_LEVEL == 0
#  define USE_IF_ASSERTS_ON(line)
#  define ASSERTL1(condition)
#  define ASSERTL2(condition)
#  define ASSERTL3(condition)
#  define ASSERTEQ(c1, c2)
#  define ASSERTRANGE(c, l, h)
#  define IFASSERT(x)
#else
#  define USE_IF_ASSERTS_ON(line) line
#endif

/* USE_IF_ASSERTS_ON allows us to remove lines that are necessary for
   code that is needed if asserts are on but causes warnings if 
   asserts are off (ie: in optimized builds).  All it does it remove
   the line or put the line in. */

#define ASSERT(condition) ASSERTL2(condition)

#endif
