
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
 *  GuiManager.cc: 
 *
 *  This class keeps a dynamic array of connections for use by TCL variables
 *  needing to get their values from the Master.  These are kept in a pool.
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/GuiInterface/GuiManager.h>

namespace SCIRun {

GuiManager *gm;
  
 
} // End namespace SCIRun


