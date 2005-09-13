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


//    File   : ToStructured.h
//    Author : Michael Callahan
//    Date   : July 2004

#if !defined(ToStructured_h)
#define ToStructured_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>

namespace SCIRun {

class ToStructuredAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *module, FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const string &partial_fdst);
};



template <class FSRC, class FDST>
class ToStructuredAlgoT : public ToStructuredAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *module, FieldHandle src);
};



template <class FSRC, class FDST>
FieldHandle
ToStructuredAlgoT<FSRC, FDST>::execute(ProgressReporter *module,
				       FieldHandle field_h)
{
  FSRC *ifield = dynamic_cast<FSRC *>(field_h.get_rep());
  typename FSRC::mesh_handle_type mesh = ifield->get_typed_mesh();

  typename FDST::mesh_handle_type outmesh =
    scinew typename FDST::mesh_type();

  vector<unsigned int> dims;
  mesh->get_dim(dims);
  outmesh->set_dim(dims);

  FDST *ofield = scinew FDST(outmesh, field_h->basis_order());

  typename FSRC::mesh_type::Node::iterator bn, en;
  mesh->begin(bn); mesh->end(en);
  while (bn != en)
  {
    Point np;
    mesh->get_center(np, *bn);
    outmesh->set_point(np, *bn);
    ++bn;
  }

  typename FSRC::fdata_type::iterator bdi, edi, bdo;
  bdi = ifield->fdata().begin();
  edi = ifield->fdata().end();
  bdo = ofield->fdata().begin();
  while (bdi != edi)
  {
    *bdo = *bdi;
    ++bdi;
    ++bdo;
  }

  return ofield;
}


} // end namespace SCIRun

#endif // ToStructured_h
