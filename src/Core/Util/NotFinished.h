
/*
 *  NotFinished.h:  Consistent way to keep track of holes in the code...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_NotFinished_h
#define SCI_project_NotFinished_h

#include <iostream>

#define NOT_FINISHED(what) std::cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ") " << std::endl

#endif /* SCI_project_NotFinished_h */

