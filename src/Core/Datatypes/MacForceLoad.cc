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
  Portions created by UNIVERSITY are Copyright (C) 2003, 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Core/Datatypes/MacForceLoad.h>

// Read comments in .h file to see what this file is for.

#include <stdio.h>

#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TriSurfField.h>

using namespace SCIRun;

void
macForceLoad()
{
  printf( "Forcing load of Core/Datatypes (for Macintosh)\n" );

  // It appears that we need more than one type to force the
  // instantiation of all of Core/Datatypes.  Until we find a better
  // solution (or upgrade to the next OS version (jaguar) which I
  // think will fix this) I suggest that we just add the types to this
  // file as we find them "missing"...  -Dd

  PointCloudField<double> pcfd;
  TriSurfField<double> tsfd;
}
