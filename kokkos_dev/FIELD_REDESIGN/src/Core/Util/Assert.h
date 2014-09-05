
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

#include <SCICore/Exceptions/AssertionFailed.h>
#include <SCICore/Exceptions/ArrayIndexOutOfBounds.h>

#define ASSERTFAIL(string) \
   SCI_THROW(SCICore::Exceptions::AssertionFailed(string, __FILE__, __LINE__));

#if SCI_ASSERTION_LEVEL >= 1
#define ASSERTL1(condition) \
   if(!(condition)){ \
      SCI_THROW(SCICore::Exceptions::AssertionFailed(#condition, __FILE__, __LINE__)); \
   }
#else
#define ASSERTL1(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTL2(condition) \
   if(!(condition)){ \
      SCI_THROW(SCICore::Exceptions::AssertionFailed(#condition, __FILE__, __LINE__)); \
   }
#else
#define ASSERTL2(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 3
#define ASSERTL3(condition) \
   if(!(condition)){ \
      SCI_THROW(SCICore::Exceptions::AssertionFailed(#condition, __FILE__, __LINE__)); \
   }
#define CHECKARRAYBOUNDS(value, lower, upper) \
   if(value < lower || value >= upper){ \
      SCI_THROW(SCICore::Exceptions::ArrayIndexOutOfBounds(value, lower, upper)); \
   }
#else
#define ASSERTL3(condition)
#define CHECKARRAYBOUNDS(value, lower, upper)
#endif

#if SCI_ASSERTION_LEVEL == 0

#define ASSERTL1(condition)
#define ASSERTL2(condition)
#define ASSERTL3(condition)
#define ASSERTEQ(c1, c2)
#define ASSERTRANGE(c, l, h)

#endif

#define ASSERT(condition) ASSERTL2(condition)

#endif
