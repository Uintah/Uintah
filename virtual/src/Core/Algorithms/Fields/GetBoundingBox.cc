/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Core/Basis/NoData.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Algorithms/Fields/GetBoundingBox.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/BBox.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool GetBoundingBoxAlgo::GetBoundingBox(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output)
{
  // Safety checks
  if (input.get_rep() == 0)
  {
    pr->error("GetBoundingBox: No input field");
    return (false);
  }

  // Get mesh
  MeshHandle mesh = input->mesh();
  
  if (mesh.get_rep() == 0)
  {
    pr->error("GetBoundingBox: No mesh associated with input field");
    return (false);
  }

  // Get bounding box from input. This is one of the few operations one can do
  // with virtual functions.
  BBox box = mesh->get_bounding_box();
  Point min = box.min();
  Point max = box.max();
  
  // Create a simple latvol field based on the minimum and maximum point
  LockingHandle<LatVolMesh<HexTrilinearLgn<Point> > > omesh = scinew LatVolMesh<HexTrilinearLgn<Point> >(2,2,2,min,max); 
  output = dynamic_cast<SCIRun::Field *>(scinew GenericField<LatVolMesh<HexTrilinearLgn<Point> >, NoDataBasis<double> , FData3d<double,LatVolMesh<HexTrilinearLgn<Point> > > >(omesh.get_rep()));

  // If this somehow failed return an error
  if (output.get_rep() == 0)
  {
    pr->error("GetBoundingBox: Could not allocate output mesh");
    return (false);  
  }

  return (true);
}   

} // namespace SCIRunAlgo
