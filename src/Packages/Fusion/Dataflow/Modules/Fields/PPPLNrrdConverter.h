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

//    File   : PPPLNrrdConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(PPPLNrrdConverter_h)
#define PPPLNrrdConverter_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>

#include <Teem/Core/Datatypes/NrrdData.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;



class PPPLNrrdConverterMeshAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle nHandle,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd,
					    const unsigned int nt);
};


template< class MESH, class NTYPE >
class PPPLNrrdConverterMeshAlgoT : public PPPLNrrdConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle nHandle,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap);
};

template< class MESH, class NTYPE >
void
PPPLNrrdConverterMeshAlgoT< MESH, NTYPE >::execute(MeshHandle mHandle,
				      NrrdDataHandle nHandle,
				      int idim, int jdim, int kdim,
				      int iwrap, int jwrap, int kwrap)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;
  
  NTYPE *ptr = (NTYPE *)(nHandle->nrrd->data);

  for( k=0; k<kdim + kwrap; k++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( i=0; i<idim + iwrap; i++ ) {
	
	int index = ((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim);

	// Grid
	float xVal = ptr[index*3 + 0];
	float yVal = ptr[index*3 + 1];
	float zVal = ptr[index*3 + 2];
	
	imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);

	++inodeItr;
      }
    }
  }
}


class PPPLNrrdConverterFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle fHandle,
		       NrrdDataHandle nHandle,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const unsigned int nt,
					    int rank);
};

template< class FIELD, class NTYPE >
class PPPLNrrdConverterFieldAlgoScalar : public PPPLNrrdConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle fHandle,
		       NrrdDataHandle nHandle,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap);
};


template< class FIELD, class NTYPE >
void
PPPLNrrdConverterFieldAlgoScalar<FIELD, NTYPE>::execute(FieldHandle fHandle,
					   NrrdDataHandle nHandle,
					   int idim, int jdim, int kdim,
					   int iwrap, int jwrap, int kwrap)
{
  FIELD *ifield = (FIELD *) fHandle.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  NTYPE *ptr = (NTYPE *)(nHandle->nrrd->data);

  register int i, j, k;
  
  for( k=0; k<kdim + kwrap; k++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( i=0; i<idim + iwrap; i++ ) {
	
	int index = ((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim);
	
	// Value
	ifield->set_value( ptr[index], *inodeItr);
	
	++inodeItr;
      }
    }
  }
}

template< class FIELD, class NTYPE >
class PPPLNrrdConverterFieldAlgoVector : public PPPLNrrdConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle fHandle,
		       NrrdDataHandle nHandle,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap);
};


template< class FIELD, class NTYPE >
void
PPPLNrrdConverterFieldAlgoVector<FIELD, NTYPE>::execute(FieldHandle fHandle,
					   NrrdDataHandle nHandle,
					   int idim, int jdim, int kdim,
					   int iwrap, int jwrap, int kwrap)
{
  FIELD *ifield = (FIELD *) fHandle.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  NTYPE *ptr = (NTYPE *)(nHandle->nrrd->data);

  register int i, j, k;

  for( k=0; k<kdim + kwrap; k++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( i=0; i<idim + iwrap; i++ ) {
	
	int index = (((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim)) * 3;
	
	// Value
	ifield->set_value( Vector( ptr[index],
				   ptr[index+1],
				   ptr[index+2]),
			   *inodeItr);
	
	++inodeItr;
      }
    }
  }

}

} // end namespace Fusion

#endif // PPPLNrrdConverter_h
