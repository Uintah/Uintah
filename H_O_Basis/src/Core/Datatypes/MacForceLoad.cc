/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


#include <Core/Datatypes/MacForceLoad.h>

// Read comments in .h file to see what this file is for.

#include <stdio.h>

#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/StructHexVolField.h>

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
  StructQuadSurfField<double> sqsfd;
  TriSurfMesh tsmd;
  StructHexVolField<double> shvfd;

}
