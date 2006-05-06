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

#include <Packages/ModelCreation/Core/Converter/MatrixToField.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/GenericField.h>

namespace ModelCreation {

using namespace SCIRun;


bool MatrixToFieldAlgo::MatrixToField(ProgressReporter* pr, MatrixHandle input, FieldHandle& output,std::string datalocation)
{
  MatrixHandle mat = dynamic_cast<Matrix *>(input->dense());
  if (mat.get_rep() == 0)
  {
    pr->error("MatrixToField: Could not convert matrix into dense matrix");
    return (false);    
  } 

  int m = mat->ncols();
  int n = mat->nrows();
  double* dataptr = mat->get_data_pointer();
  int k = 0;
  
  if (datalocation == "Node")
  {
    ImageMesh<QuadBilinearLgn<Point> >::handle_type mesh_handle = scinew ImageMesh<QuadBilinearLgn<Point> >(m,n,Point(static_cast<double>(m),0.0,0.0),Point(0.0,static_cast<double>(n),0.0));
    GenericField<ImageMesh<QuadBilinearLgn<Point> >,QuadBilinearLgn<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >* field = scinew GenericField<ImageMesh<QuadBilinearLgn<Point> >,QuadBilinearLgn<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >(mesh_handle);
    output = dynamic_cast<Field *>(field);
    ImageMesh<QuadBilinearLgn<Point> >::Node::iterator it, it_end;
    mesh_handle->begin(it);
    mesh_handle->end(it_end);
    while (it != it_end)
    {
      field->set_value(dataptr[k++],*it);
      ++it;
    }
  }
  else if (datalocation == "Element")
  {
    ImageMesh<QuadBilinearLgn<Point> >::handle_type mesh_handle = scinew ImageMesh<QuadBilinearLgn<Point> >(m+1,n+1,Point(static_cast<double>(m+1),0.0,0.0),Point(0.0,static_cast<double>(n+1),0.0));
    GenericField<ImageMesh<QuadBilinearLgn<Point> >,ConstantBasis<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >* field = scinew GenericField<ImageMesh<QuadBilinearLgn<Point> >,ConstantBasis<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >(mesh_handle);
    output = dynamic_cast<Field *>(field);
    ImageMesh<QuadBilinearLgn<Point> >::Elem::iterator it, it_end;
    mesh_handle->begin(it);
    mesh_handle->end(it_end);
    while (it != it_end)
    {
      field->set_value(dataptr[k++],*it);
      ++it;
    }  
  }
  else
  {
    pr->error("MatrixToField: Data location information is not recognized");
    return (false);      
  }
  
  return (true);
}

} // end namespace
