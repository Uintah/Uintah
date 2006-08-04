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

#include <Core/Algorithms/Converter/NrrdToField.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/GenericField.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class NrrdToFieldAlgoT {
public:
  template<class T>
  bool NrrdToField(ProgressReporter* pr,NrrdDataHandle input, FieldHandle& output,std::string datalocation);
};

template<class T>
bool NrrdToFieldAlgoT::NrrdToField(ProgressReporter* pr,NrrdDataHandle input, FieldHandle& output,std::string datalocation)
{
  Nrrd *nrrd = input->nrrd_;

  if (nrrd == 0)
  {
    pr->error("NrrdToField: NrrdData does not contain Nrrd");
    return (false);      
  }

  size_t dim = nrrd->dim;
  std::vector<double> min(dim), max(dim);
  std::vector<int> size(dim);
  
  for (size_t p=0; p<dim; p++) 
  {
    size[p] = nrrd->axis[p].size;
    
    if (airExists(nrrd->axis[p].min)) 
    {
      min[p] = nrrd->axis[p].min;
    }
    else
    {
      min[p] = 0.0;
    }
    
    if (airExists(nrrd->axis[p].max)) 
    {
      max[p] = nrrd->axis[p].max;
    }
    else
    {
      if (airExists(nrrd->axis[p].spacing)) 
      {
        max[p] = nrrd->axis[p].spacing*size[p];
      }
      else
      {
        max[p] = static_cast<double>(size[p]);
      }
    }
    if (nrrd->axis[p].center == 2)
    {
      min[p] += (max[p]-min[p])/(2*size[p]);
      max[p] -= (max[p]-min[p])/(2*size[p]);
    }
  }  

  Transform tf;
  tf.load_identity();
  bool use_tf = false;

  if (nrrd->spaceDim > 0)
  {
    
    Vector Origin;
    std::vector<Vector> SpaceDir(3);
    
    int sd = nrrd->spaceDim;
    for (int q=0 ; q<sd && q < 3; q++)
    {
      if (airExists(nrrd->spaceOrigin[q])) Origin[q] = nrrd->spaceOrigin[q]; else Origin[q] = 0.0;
      for (size_t p=0; p<dim && p < 3;p++)
        if (airExists(nrrd->axis[p].spaceDirection[q])) SpaceDir[p][q] = nrrd->axis[p].spaceDirection[q]; else SpaceDir[p][q] = 0.0;
    }
  
    if (dim == 1) 
    {
      SpaceDir[0].find_orthogonal(SpaceDir[1],SpaceDir[2]);
    }
    else if (dim == 2)
    {
      SpaceDir[2] = Cross(SpaceDir[0],SpaceDir[1]);
    }


    tf.load_basis(Point(Origin),SpaceDir[0],SpaceDir[1],SpaceDir[2]);
  
    use_tf = true;
  }

  T* dataptr = static_cast<T*>(nrrd->data);
 
  if (dim == 2)
  {
    int k = 0;
    
    if (datalocation == "Node")
    {
      ImageMesh<QuadBilinearLgn<Point> >::handle_type mesh_handle = scinew ImageMesh<QuadBilinearLgn<Point> >(size[0],size[1],Point(min[0],min[1],0.0),Point(max[0],max[1],0.0));
      GenericField<ImageMesh<QuadBilinearLgn<Point> >,QuadBilinearLgn<T>, FData2d<T, ImageMesh<QuadBilinearLgn<Point> > > >* field = scinew GenericField<ImageMesh<QuadBilinearLgn<Point> >,QuadBilinearLgn<T>, FData2d<T, ImageMesh<QuadBilinearLgn<Point> > > >(mesh_handle);
      output = dynamic_cast<Field *>(field);
      ImageMesh<QuadBilinearLgn<Point> >::Node::iterator it, it_end;
      mesh_handle->begin(it);
      mesh_handle->end(it_end);
      while (it != it_end)
      {
        field->set_value(dataptr[k++],*it);
        ++it;
      }
      
      if (use_tf) mesh_handle->set_transform(tf);
    }
    else if (datalocation == "Element")
    {
      ImageMesh<QuadBilinearLgn<Point> >::handle_type mesh_handle = scinew ImageMesh<QuadBilinearLgn<Point> >(size[0]+1,size[1]+1,Point(min[0],min[1],0.0),Point(max[0],max[1],0.0));
      GenericField<ImageMesh<QuadBilinearLgn<Point> >,ConstantBasis<T>, FData2d<T, ImageMesh<QuadBilinearLgn<Point> > > >* field = scinew GenericField<ImageMesh<QuadBilinearLgn<Point> >,ConstantBasis<T>, FData2d<T, ImageMesh<QuadBilinearLgn<Point> > > >(mesh_handle);
      output = dynamic_cast<Field *>(field);
      ImageMesh<QuadBilinearLgn<Point> >::Elem::iterator it, it_end;
      mesh_handle->begin(it);
      mesh_handle->end(it_end);
      while (it != it_end)
      {
        field->set_value(dataptr[k++],*it);
        ++it;
      }  
      if (use_tf) mesh_handle->set_transform(tf);
    }
    else
    {
      pr->error("NrrdToField: Data location information is not recognized");
      return (false);      
    }
  }
  else if (dim == 3)
  {
    int k = 0;
    
    if (datalocation == "Node")
    {
      LatVolMesh<HexTrilinearLgn<Point> >::handle_type mesh_handle = scinew LatVolMesh<HexTrilinearLgn<Point> >(size[0],size[1],size[2],Point(min[0],min[1],min[2]),Point(max[0],max[1],max[2]));
      GenericField<LatVolMesh<HexTrilinearLgn<Point> >,HexTrilinearLgn<T>, FData3d<T, LatVolMesh<HexTrilinearLgn<Point> > > >* field = scinew GenericField<LatVolMesh<HexTrilinearLgn<Point> >,HexTrilinearLgn<T>, FData3d<T, LatVolMesh<HexTrilinearLgn<Point> > > >(mesh_handle);
      output = dynamic_cast<Field *>(field);
      LatVolMesh<HexTrilinearLgn<Point> >::Node::iterator it, it_end;
      mesh_handle->begin(it);
      mesh_handle->end(it_end);
      while (it != it_end)
      {
        field->set_value(dataptr[k++],*it);
        ++it;
      }
      if (use_tf) mesh_handle->set_transform(tf);      
    }
    else if (datalocation == "Element")
    {
      LatVolMesh<HexTrilinearLgn<Point> >::handle_type mesh_handle = scinew LatVolMesh<HexTrilinearLgn<Point> >(size[0]+1,size[1]+1,size[2]+1,Point(min[0],min[1],min[2]),Point(max[0],max[1],max[2]));
      GenericField<LatVolMesh<HexTrilinearLgn<Point> >,ConstantBasis<T>, FData3d<T, LatVolMesh<HexTrilinearLgn<Point> > > >* field = scinew GenericField<LatVolMesh<HexTrilinearLgn<Point> >,ConstantBasis<T>, FData3d<T, LatVolMesh<HexTrilinearLgn<Point> > > >(mesh_handle);
      output = dynamic_cast<Field *>(field);
      LatVolMesh<HexTrilinearLgn<Point> >::Elem::iterator it, it_end;
      mesh_handle->begin(it);
      mesh_handle->end(it_end);
      while (it != it_end)
      {
        field->set_value(dataptr[k++],*it);
        ++it;
      }  
      if (use_tf) mesh_handle->set_transform(tf);      
    }
    else
    {
      pr->error("NrrdToField: Data location information is not recognized");
      return (false);      
    }
  }
  else
  {
    pr->error("NrrdToField: Nrrd is not 2D or 3D");
    return (false);        
  }
  
  return (true);
}


bool NrrdToFieldAlgo::NrrdToField(ProgressReporter *pr, NrrdDataHandle input, FieldHandle& output,std::string datalocation)
{
  if (input.get_rep() == 0)
  {
    pr->error("NrrdToField: No input Nrrd");
    return (false);    
  } 

  Nrrd *nrrd = input->nrrd_;

  if (nrrd == 0)
  {
    pr->error("NrrdToField: NrrdData does not contain Nrrd");
    return (false);      
  }

  NrrdToFieldAlgoT algo;

  switch (nrrd->type)
  {
    case nrrdTypeChar : 
      return(algo.NrrdToField<char>(pr,input,output,datalocation));
    case nrrdTypeUChar : 
      return(algo.NrrdToField<unsigned char>(pr,input,output,datalocation));
    case nrrdTypeShort : 
      return(algo.NrrdToField<short>(pr,input,output,datalocation));
    case nrrdTypeUShort : 
      return(algo.NrrdToField<unsigned short>(pr,input,output,datalocation));              
    case nrrdTypeInt : 
      return(algo.NrrdToField<int>(pr,input,output,datalocation));
    case nrrdTypeUInt : 
      return(algo.NrrdToField<unsigned int>(pr,input,output,datalocation));
    case nrrdTypeFloat : 
      return(algo.NrrdToField<float>(pr,input,output,datalocation));
    case nrrdTypeDouble : 
      return(algo.NrrdToField<double>(pr,input,output,datalocation));
    default: 
      pr->error("NrrdToField: This datatype is not supported");
      return (false);
  }
  return (true);
}

} // end namespace SCIRunAlgo


