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

//    File   : InterpFieldToMatrix.h
//    Author : David Weinstein
//    Date   : December 2001

#if !defined(InterpFieldToMatrix_h)
#define InterpFieldToMatrix_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Util/ModuleReporter.h>

namespace SCIRun {

class InterpFieldToMatrixAlgoBase : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src, MatrixHandle &d, MatrixHandle &c) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc);
};


template <class Fld, class Loc>
class InterpFieldToMatrixAlgo : public InterpFieldToMatrixAlgoBase
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src, MatrixHandle &d, MatrixHandle &c);
};

template <class Fld, class Loc>
void
InterpFieldToMatrixAlgo<Fld, Loc>::execute(FieldHandle ifield_h, 
					   MatrixHandle &d, MatrixHandle &c)
{
  Fld *ifield = dynamic_cast<Fld *>(ifield_h.get_rep());
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
//  mesh->synchronize(Mesh::ALL_ELEMENTS_E);
  
  typename Loc::size_type ssize;  mesh->size(ssize);
  DenseMatrix *dmatrix = scinew DenseMatrix(ssize,ssize);
  dmatrix->zero();
  ColumnMatrix *cmatrix = scinew ColumnMatrix(ssize);
  cmatrix->zero();
  d=dmatrix;
  c=cmatrix;
  int index = 0;
  typename Loc::iterator iter; mesh->begin(iter);
  typename Loc::iterator eiter; mesh->end(eiter);

  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    for (int i=0; i<val.size(); i++) {
      unsigned int c=val[i].first;
      double v=val[i].second;
      (*dmatrix)[index][c]=v;
      if (i==0) (*cmatrix)[index]=c;
    }
    ++index;
    ++iter;
  }
  string units;
  dmatrix->set_property("weights", units, false);
  cmatrix->set_property("mapindex", units, false);
}

} // end namespace SCIRun

#endif // InterpFieldToMatrix_h
