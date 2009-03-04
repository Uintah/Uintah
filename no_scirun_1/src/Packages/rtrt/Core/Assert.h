/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#if 0
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

#ifndef SCI_Classlib_Assert_h
#define SCI_Classlib_Assert_h 1

#define SCI_ASSERTION_LEVEL 2

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
#ifdef SCI_ASSERTION_LEVEL_4
#define SCI_ASSERTION_LEVEL 4
#endif

#include <Packages/rtrt/Core/Exceptions.h>

#if SCI_ASSERTION_LEVEL >= 1
#define ASSERTL1(condition) \
	if(!(condition)){ \
		AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL1(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTL2(condition) \
	if(!(condition)){ \
		AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL2(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 3
#define ASSERTL3(condition) \
	if(!(condition)){ \
		AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL3(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 4
#define ASSERTL4(condition) \
	if(!(condition)){ \
		AssertionFailed exc(#condition); \
		EXCEPTION(exc); \
	}
#else
#define ASSERTL4(condition)
#endif

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTEQ(c1, c2) \
	if(c1 != c2){ \
		AssertionEQFailed exc(#c1, #c2, (int)c1, (int)c2); \
		EXCEPTION(exc); \
	}
#define ASSERTRANGE(c, l, h) \
        if(c < l || c >= h){ \
		AssertionRangeFailed exc(#c, #l, #h, c, l, h); \
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
#define ASSERTL4(condition)
#define ASSERTEQ(c1, c2)
#define ASSERTRANGE(c, l, h)

#endif

#define ASSERT(condition) ASSERTL2(condition)

#endif
#else
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Assert.h>
#endif
