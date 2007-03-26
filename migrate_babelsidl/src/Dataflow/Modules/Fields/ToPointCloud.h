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


//    File   : ToPointCloud.h
//    Author : Michael Callahan
//    Date   : September 2001

#if !defined(ToPointCloud_h)
#define ToPointCloud_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Basis/Constant.h>

namespace SCIRun {

class ToPointCloudAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const string &mesh_dst,
					    const string &basis_dst,
					    const string &data_dst);
};


template <class FSRC, class FDST>
class ToPointCloudAlgoT : public ToPointCloudAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle src);
};


template <class FSRC, class FDST>
FieldHandle
ToPointCloudAlgoT<FSRC, FDST>::execute(ProgressReporter *reporter,
				      FieldHandle field_h)
{
  FSRC *ifield = dynamic_cast<FSRC *>(field_h.get_rep());
  typename FSRC::mesh_handle_type srcmesh = ifield->get_typed_mesh();
  typename FDST::mesh_handle_type dstmesh = scinew typename FDST::mesh_type();

  // Copy the points.
  typename FSRC::mesh_type::Node::iterator bn, en;
  srcmesh->begin(bn);
  srcmesh->end(en);

  Point np;

  while (bn != en) {
    srcmesh->get_center(np, *bn);
    dstmesh->add_point(np);
    ++bn;
  }

  // Really should copy normals
  dstmesh->synchronize(Mesh::NORMALS_E);

  FDST *ofield = scinew FDST(dstmesh);

  // Copy the data at the nodes
  if (field_h->basis_order() == 1) {
    typename FSRC::fdata_type::iterator in  = ifield->fdata().begin();
    typename FSRC::fdata_type::iterator end = ifield->fdata().end();
    typename FDST::fdata_type::iterator out = ofield->fdata().begin();

    while (in != end) {
      *out = *in;
      ++in; ++out;
    }
  } else {
    reporter->warning("Unable to copy data at this field data location.");
  }

  return ofield;
}


} // end namespace SCIRun

#endif // ToPointCloud_h
