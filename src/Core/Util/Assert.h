
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
#ifdef SCI_ASSERTION_LEVEL_0
#define SCI_ASSERTION_LEVEL 0
#endif
#ifdef SCI_ASSERTION_LEVEL_1
#define SCI_ASSERTION_LEVEL 1
#endif
#ifdef SCI_ASSERTION_LEVEL_2
#define SCI_ASSERTION_LEVEL 2
#endif
#ifdef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL 3
#endif

#include <SCICore/Exceptions/Exceptions.h>

#define ASSERTFAIL(string) \
       { \
           SCICore::ExceptionsSpace::AssertionFailed exc(string); \
           EXCEPTION(exc); \
       }

#if SCI_ASSERTION_LEVEL >= 1
#define ASSERTL1(condition) \
	if(!(condition)){ \
		SCICore::ExceptionsSpace::AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL1(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTL2(condition) \
	if(!(condition)){ \
		SCICore::ExceptionsSpace::AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL2(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 3
#define ASSERTL3(condition) \
	if(!(condition)){ \
		SCICore::ExceptionsSpace::AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL3(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTEQ(c1, c2) \
	if(c1 != c2){ \
		SCICore::ExceptionsSpace::AssertionEQFailed exc(#c1, #c2, (int)c1, (int)c2); \
		EXCEPTION(exc); \
	}
#define ASSERTRANGE(c, l, h) \
        if(c < l || c >= h){ \
		SCICore::ExceptionsSpace::AssertionRangeFailed exc(#c, #l, #h, c, l, h); \
		EXCEPTION(exc); \
        }
#else
#define ASSERTEQ(c1, c2)
#define ASSERTRANGE(c, l, h)
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
