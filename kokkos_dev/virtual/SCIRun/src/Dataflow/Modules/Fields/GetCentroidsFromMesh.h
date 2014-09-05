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


//    File   : GetCentroidsFromMesh.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(GetCentroidsFromMesh_h)
#define GetCentroidsFromMesh_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>

namespace SCIRun {

class GetCentroidsFromMeshAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class GetCentroidsFromMeshAlgoT : public GetCentroidsFromMeshAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle src);
};


template <class FIELD>
FieldHandle
GetCentroidsFromMeshAlgoT<FIELD>::execute(ProgressReporter *reporter, FieldHandle field_h)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  typename FIELD::mesh_handle_type mesh = ifield->get_typed_mesh();
  typedef PointCloudMesh<ConstantBasis<Point> >                 PCMesh;
  typedef ConstantBasis<double>                                 FDCdoubleBasis;
  typedef GenericField<PCMesh, FDCdoubleBasis, vector<double> > PCField;
  PCMesh::handle_type pcm(scinew PCMesh);

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi);
  mesh->end(ei);

  typename FIELD::mesh_type::Elem::size_type prsizetmp;
  mesh->size(prsizetmp);
  const unsigned int prsize = (unsigned int)prsizetmp;
  unsigned int prcounter = 0;

  Point p;
  while (bi != ei)
  {
    reporter->update_progress(prcounter++, prsize);

    mesh->get_center(p, *bi);
    pcm->add_node(p);
    ++bi;
  }

  return scinew PCField(pcm);
}


} // end namespace SCIRun

#endif // GetCentroidsFromMesh_h
