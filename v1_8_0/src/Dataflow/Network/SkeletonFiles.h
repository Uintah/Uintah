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

/* SkeletonFiles.h 
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   Copyright (c) 2000
 *   University of Utah
 */

#ifndef SKELETONFILES_H
#define SKELETONFILES_H 1

namespace SCIRun {

/* the following are skeleton files
   stored as string literals. */

extern char component_skeleton[];
extern char gui_skeleton[];
extern char share_skeleton[];
extern char dllentry_skeleton[];
extern char package_submk_skeleton[];
extern char dataflow_submk_skeleton[];
extern char core_submk_skeleton[];
extern char modules_submk_skeleton[];
extern char category_submk_skeleton[];
extern char datatypes_submk_skeleton[];
extern char gui_submk_skeleton[];

} // End namespace SCIRun

#endif


