/*  The contents of this file are subject to the University of Utah Public
 *  License (the "License"); you may not use this file except in compliance
 *  with the License.
 *  
 *  Software distributed under the License is distributed on an "AS IS"
 *  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 *  License for the specific language governing rights and limitations under
 *  the License.
 *  
 *  The Original Source Code is SCIRun, released March 12, 2001.
 *  
 *  The Original Source Code was developed by the University of Utah.
 *  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
 *  University of Utah. All Rights Reserved.
 *  
 *  File   : sci_values.h
 *  Author : J. Davison de St. Germain
 *  Date   : Dec. 9, 2003
 *
 *    This file encapsulates differences between different platforms
 *  with respect to the values.h file.
 */

#if !defined(SCI_VALUES_H)
#define SCI_VALUES_H

#ifdef __APPLE__

#include <float.h>
#define MAXDOUBLE DBL_MAX
#define MAXSHORT  SHRT_MAX
#define MAXINT    INT_MAX

#else

#include <values.h>

#endif


#endif  /* #define SCI_VALUES_H */
