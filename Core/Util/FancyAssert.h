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

#ifndef SCI_Containers_FancyAssert_h
#define SCI_Containers_FancyAssert_h

#include <sci_config.h>
#include <Core/Exceptions/AssertionFailed.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

/*
 * Note - a normal AssertionFailed exception cannot be used here.  We
 * must use one that takes a string
 */

#if SCI_ASSERTION_LEVEL >= 2
#define ASSERTEQ(c1, c2) \
   if(c1 != c2){ \
      std::ostringstream msg; \
      msg << #c1 << "(value=" << c1 << ") == " << #c2 << "(value=" << c2 << ")"; \
      SCI_THROW(SCIRun::AssertionFailed(msg.str().c_str(), __FILE__, __LINE__)); \
   }

#define ASSERTNE(c1, c2) \
   if(c1 != c2){ \
      std::ostringstream msg; \
      msg << #c1 << "(value=" << c1 << ") != " << #c2 << "(value=" << c2 << ")"; \
      SCI_THROW(SCIRun::AssertionFailed(msg.str().c_str(), __FILE__, __LINE__)); \
   }

#define ASSERTRANGE(c, l, h) \
   if(c < l || c >= h){ \
      std::ostringstream msg; \
      msg << #l "(value=" << l << ") <= " #c << "(value=" << c << ") < " << #h << "(value=" << h << ")"; \
      SCI_THROW(SCIRun::AssertionFailed(msg.str().c_str(), __FILE__, __LINE__)); \
   }
#else
#define ASSERTEQ(c1, c2)
#define ASSERTRANGE(c, l, h)
#endif

#endif
