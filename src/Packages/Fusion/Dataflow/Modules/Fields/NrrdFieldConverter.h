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

//    File   : NrrdFieldConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(NrrdFieldConverter_h)
#define NrrdFieldConverter_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/QuadSurfField.h>

#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/HexVolField.h>

#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <Teem/Core/Datatypes/NrrdData.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class NrrdFieldConverterMeshAlgo : public DynamicAlgoBase
{
public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info( const std::string topoStr,
					     const TypeDescription *mtd,
					     const unsigned int ptype,
					     const unsigned int ctype);

};

class StructuredNrrdFieldConverterMeshAlgo : public NrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > mesh,
		       int idim, int jdim, int kdim) = 0;
};


template< class MESH, class PNTYPE, class CNTYPE >
class StructuredNrrdFieldConverterMeshAlgoT : public StructuredNrrdFieldConverterMeshAlgo
{
public:

  virtual void execute(MeshHandle mHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > mesh,
		       int idim, int jdim, int kdim);
};


template< class MESH, class PNTYPE, class CNTYPE >
void
StructuredNrrdFieldConverterMeshAlgoT< MESH, PNTYPE, CNTYPE >::
execute(MeshHandle mHandle,
	vector< NrrdDataHandle > nHandles,
	vector< int > mesh,
	int idim, int jdim, int kdim)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;

  std::string property;

  nHandles[mesh[0]]->get_property( "Coordinate System", property );

  if( property.find("Cartesian") != std::string::npos ) {
  
    PNTYPE *ptr = (PNTYPE *)(nHandles[mesh[0]]->nrrd->data);

    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
	
	  int index = (i * jdim + j) * kdim + k;

	  // Mesh
	  float xVal = ptr[index*3 + 0];
	  float yVal = ptr[index*3 + 1];
	  float zVal = ptr[index*3 + 2];
	
	  imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);

	  ++inodeItr;
	}
      }
    }
  }
}

class UnstructuredNrrdFieldConverterMeshAlgo : public NrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle pHandle,
		       NrrdDataHandle cHandle) = 0;
};

template< class MESH, class PNTYPE, class CNTYPE >
class UnstructuredNrrdFieldConverterMeshAlgoT : public UnstructuredNrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle pHandle,
		       NrrdDataHandle cHandle);
};


template< class MESH, class PNTYPE, class CNTYPE >
void
UnstructuredNrrdFieldConverterMeshAlgoT< MESH, PNTYPE, CNTYPE >::
execute(MeshHandle mHandle,
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

  int nelements = cHandle->nrrd->axis[1].size;

  typename MESH::Node::array_type array(6);

  for( int i=0; i<nelements; i++ ) {
    for( unsigned int j=0; j<6; j++ ) {
      array[j] = (int) cPtr[i*6+j];
    }

    imesh->add_elem( array );
  }
}





class NrrdFieldConverterFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle mHandle,
			      vector< NrrdDataHandle > nHandles,
			      vector< int > data,
			      int idim, int jdim, int kdim) = 0;
  
  virtual FieldHandle execute(MeshHandle mHandle,
			      vector< NrrdDataHandle > nHandles,
			      vector< int > data) = 0;
  
   //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd,
					    const std::string fname,
					    const unsigned int ntype,
					    int rank);
};

template< class FIELD, class MESH, class NTYPE >
class NrrdFieldConverterFieldAlgoScalar : public NrrdFieldConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle mHandle,
			      vector< NrrdDataHandle > nHandles,
			      vector< int > data,
			      int idim, int jdim, int kdim);

  virtual FieldHandle execute(MeshHandle mHandle,
			      vector< NrrdDataHandle > nHandles,
			      vector< int > data);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoScalar<FIELD, MESH, NTYPE>::
execute(MeshHandle mHandle,
	vector< NrrdDataHandle > nHandles,
	vector< int > data,
	int idim, int jdim, int kdim)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

  if( data.size() == 1 ) {
    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);

    register int i, j, k;

    for( i=0; i<idim; i++ ) {
      for( j=0; j<jdim; j++ ) {
	for( k=0; k<kdim; k++ ) {
	
	  int index = (i * jdim + j) * kdim + k;
	
	  // Value
	  ifield->set_value( ptr[index], *inodeItr);
	
	  ++inodeItr;
	}
      }
    }
  }

  return FieldHandle( ifield );
}



template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoScalar<FIELD, MESH, NTYPE>::
execute(MeshHandle mHandle,
	vector< NrrdDataHandle > nHandles,
	vector< int > data)

{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

  if( data.size() == 1 ) {
    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);

    int npts = nHandles[data[0]]->nrrd->axis[1].size;

    // Value
    for( int i=0; i<npts; i++ ) {
      ifield->set_value( ptr[i], *inodeItr);
    
      ++inodeItr;
    }
  }

  return FieldHandle( ifield );
}





template< class FIELD, class MESH, class NTYPE >
class NrrdFieldConverterFieldAlgoVector : public NrrdFieldConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual FieldHandle execute(MeshHandle mHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > data,
		       int idim, int jdim, int kdim);

  virtual FieldHandle execute(MeshHandle mHandle,
			      vector< NrrdDataHandle > nHandles,
			      vector< int > data);  
};


template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoVector<FIELD, MESH, NTYPE>::
execute(MeshHandle mHandle,
	vector< NrrdDataHandle > nHandles,
	vector< int > data,
	int idim, int jdim, int kdim)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

  if( data.size() == 1 ) {
    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    register int i, j, k;
				  
    NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);

     for( i=0; i<idim; i++ ) {
      for( j=0; j<jdim; j++ ) {
	for( k=0; k<kdim; k++ ) {
	
	  int index = (i * jdim + j) * kdim + k;
	
	  ifield->set_value( Vector( ptr[index*3  ],
				     ptr[index*3+1],
				     ptr[index*3+2] ), *inodeItr);
	
	  ++inodeItr;
	}
      }
    }
  }

  return FieldHandle( ifield );
  
}

template< class FIELD, class MESH, class NTYPE >
FieldHandle
NrrdFieldConverterFieldAlgoVector<FIELD, MESH, NTYPE>::
execute(MeshHandle mHandle,
	vector< NrrdDataHandle > nHandles,
	vector< int > data)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  FIELD *ifield = (FIELD *) scinew FIELD((MESH *) imesh, Field::NODE);

  if( data.size() == 1 ) {
    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);

    int npts = nHandles[data[0]]->nrrd->axis[1].size;

    for( int i=0; i<npts; i++ ) {

      ifield->set_value( Vector( ptr[i*3  ],
				 ptr[i*3+1],
				 ptr[i*3+2]),
			 *inodeItr);	
      ++inodeItr;
    }
  }

  return FieldHandle( ifield );
}

} // end namespace Fusion

#endif // NrrdFieldConverter_h
