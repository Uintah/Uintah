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

//    File   : PrismNrrdConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(PrismNrrdConverter_h)
#define PrismNrrdConverter_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/PrismVolField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <Teem/Core/Datatypes/NrrdData.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;



class PrismNrrdConverterMeshAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle pHandle,
		       NrrdDataHandle cHandle) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd,
					    const unsigned int pnt,
					    const unsigned int cnt);
};


template< class MESH, class PNTYPE, class CNTYPE >
class PrismNrrdConverterMeshAlgoT : public PrismNrrdConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle pHandle,
		       NrrdDataHandle cHandle);
};

template< class MESH, class PNTYPE, class CNTYPE >
void
PrismNrrdConverterMeshAlgoT
< MESH, PNTYPE, CNTYPE >::execute(MeshHandle mHandle,
				  NrrdDataHandle pHandle,
				  NrrdDataHandle cHandle)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  PNTYPE *pPtr = (PNTYPE *)(pHandle->nrrd->data);
  CNTYPE *cPtr = (CNTYPE *)(cHandle->nrrd->data);

  int npts = pHandle->nrrd->axis[1].size;

  for( int i=0; i<npts; i++ ) {
    double xVal = pPtr[i*3  ];
    double yVal = pPtr[i*3+1];
    double zVal = pPtr[i*3+2];

    imesh->add_point( Point(xVal, yVal, zVal) );
  }

  int nprisms = cHandle->nrrd->axis[1].size;

  for( int i=0; i<nprisms; i++ ) {
    imesh->add_prism((int) cPtr[i*6  ],
		     (int) cPtr[i*6+1],
		     (int) cPtr[i*6+2],
		     (int) cPtr[i*6+3],
		     (int) cPtr[i*6+4],
		     (int) cPtr[i*6+5]);
  }
}


class PrismNrrdConverterFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle fHandle,
		       NrrdDataHandle nHandle) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const unsigned int nt,
					    int rank);
};

template< class FIELD, class NTYPE >
class PrismNrrdConverterFieldAlgoScalar : public PrismNrrdConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle fHandle,
		       NrrdDataHandle nHandle);
};


template< class FIELD, class NTYPE >
void
PrismNrrdConverterFieldAlgoScalar<FIELD, NTYPE>::execute(FieldHandle fHandle,
							 NrrdDataHandle nHandle)
{
  FIELD *ifield = (FIELD *) fHandle.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  NTYPE *ptr = (NTYPE *)(nHandle->nrrd->data);

  int npts = nHandle->nrrd->axis[1].size;

  // Value
  for( int i=0; i<npts; i++ ) {
    ifield->set_value( ptr[i], *inodeItr);
    
    ++inodeItr;
  }
}

template< class FIELD, class NTYPE >
class PrismNrrdConverterFieldAlgoVector : public PrismNrrdConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle fHandle,
		       NrrdDataHandle nHandle);
};


template< class FIELD, class NTYPE >
void
PrismNrrdConverterFieldAlgoVector<FIELD, NTYPE>::execute(FieldHandle fHandle,
							 NrrdDataHandle nHandle)
{
  FIELD *ifield = (FIELD *) fHandle.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  NTYPE *ptr = (NTYPE *)(nHandle->nrrd->data);

  int npts = nHandle->nrrd->axis[1].size;

  // Value
  for( int i=0; i<npts; i++ ) {

    ifield->set_value( Vector( ptr[i*3  ],
			       ptr[i*3+1],
			       ptr[i*3+2]),
		       *inodeItr);	
    ++inodeItr;
  }
}

} // end namespace Fusion

#endif // PrismNrrdConverter_h
