
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

#ifndef SCI_Containers_FancyAssert_h
#define SCI_Containers_FancyAssert_h

#include <sci_config.h>
#include <SCICore/Exceptions/AssertionFailed.h>
#include <sstream>

/*
 * Note - a normal AssertionFailed exception cannot be used here.  We
 * must use one that takes a string
 */

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTEQ(c1, c2) \
   if(c1 != c2){ \
      std::ostringstream msg; \
      msg << #c1 << "(value=" << c1 << ") == " << #c2 << "(value=" << c2 << ")"; \
      SCI_THROW(SCICore::Exceptions::AssertionFailed(msg.str().c_str(), __FILE__, __LINE__)); \
   }

#define ASSERTNE(c1, c2) \
   if(c1 != c2){ \
      std::ostringstream msg; \
      msg << #c1 << "(value=" << c1 << ") != " << #c2 << "(value=" << c2 << ")"; \
      SCI_THROW(SCICore::Exceptions::AssertionFailed(msg.str().c_str(), __FILE__, __LINE__)); \
   }

#define ASSERTRANGE(c, l, h) \
   if(c < l || c >= h){ \
      std::ostringstream msg; \
      msg << #l "(value=" << l << ") <= " #c << "(value=" << c << ") < " << #h << "(value=" << h << ")"; \
      SCI_THROW(SCICore::Exceptions::AssertionFailed(msg.str().c_str(), __FILE__, __LINE__)); \
   }
#else
#define ASSERTEQ(c1, c2)
#define ASSERTRANGE(c, l, h)
#endif

#endif
