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

#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>

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
class StructuredNrrdFieldConverterMeshAlgoT : 
  public StructuredNrrdFieldConverterMeshAlgo
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
  
    if( mesh.size() == 1 ) {
      PNTYPE *ptr = (PNTYPE *)(nHandles[mesh[0]]->nrrd->data);
    
      int rank = 0;
      string label(nHandles[mesh[0]]->nrrd->axis[0].label);

      if( label.find( ":Scalar" ) != string::npos )
	rank = nHandles[mesh[0]]->nrrd->axis[ nHandles[mesh[0]]->nrrd->dim-1].size;
      else if( label.find( ":Vector" ) != string::npos )
	rank = 3;

      for( k=0; k<kdim; k++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( i=0; i<idim; i++ ) {
	  
	    int index = (i * jdim + j) * kdim + k;

	    float xVal = 0, yVal = 0, zVal = 0;

	    // Mesh
	    if( rank >= 1 ) xVal = ptr[index*rank + 0];
	    if( rank >= 2 ) yVal = ptr[index*rank + 1];
	    if( rank >= 3 ) zVal = ptr[index*rank + 2];
	
	    imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);

	    ++inodeItr;
	  }
	}
      }
    } else {
      int rank = mesh.size();

      PNTYPE *ptr[3] = {NULL, NULL, NULL};

      if( rank >= 1 ) ptr[0] = (PNTYPE *)(nHandles[mesh[0]]->nrrd->data);
      if( rank >= 2 ) ptr[1] = (PNTYPE *)(nHandles[mesh[1]]->nrrd->data);
      if( rank >= 3 ) ptr[2] = (PNTYPE *)(nHandles[mesh[2]]->nrrd->data);

      for( k=0; k<kdim; k++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( i=0; i<idim; i++ ) {
	  
	    int index = (i * jdim + j) * kdim + k;

	    float xVal = 0, yVal = 0, zVal = 0;

	    // Mesh
	    if( ptr[0] ) xVal = ptr[0][index];
	    if( ptr[1] ) yVal = ptr[1][index];
	    if( ptr[2] ) zVal = ptr[2][index];
	
	    imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);

	    ++inodeItr;
	  }
	}
      }
    }
  }
}

class UnstructuredNrrdFieldConverterMeshAlgo : 
  public NrrdFieldConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       NrrdDataHandle pHandle,
		       NrrdDataHandle cHandle) = 0;
};

template< class MESH, class PNTYPE, class CNTYPE >
class UnstructuredNrrdFieldConverterMeshAlgoT : 
  public UnstructuredNrrdFieldConverterMeshAlgo
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

  int rank = 0;
  string label(pHandle->nrrd->axis[0].label);
  
  if( label.find( ":Scalar" ) != string::npos )
    rank = pHandle->nrrd->axis[ pHandle->nrrd->dim-1].size;
  else if( label.find( ":Vector" ) != string::npos )
    rank = 3;

  for( int index=0; index<npts; index++ ) {
    float xVal = 0, yVal = 0, zVal = 0;

    // Mesh
    if( rank >= 1 ) xVal = pPtr[index*rank + 0];
    if( rank >= 2 ) yVal = pPtr[index*rank + 1];
    if( rank >= 3 ) zVal = pPtr[index*rank + 2];
    
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

    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
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

    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
	  int index = (i * jdim + j) * kdim + k;
	
	  ifield->set_value( Vector( ptr[index*3  ],
				     ptr[index*3+1],
				     ptr[index*3+2] ), *inodeItr);
	
	  ++inodeItr;
	}
      }
    }
  } else {
    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    register int i, j, k;
				  
    NTYPE *ptr[3] ={NULL,NULL,NULL};

    ptr[0] = (NTYPE *)(nHandles[data[0]]->nrrd->data);
    ptr[1] = (NTYPE *)(nHandles[data[1]]->nrrd->data);
    ptr[2] = (NTYPE *)(nHandles[data[2]]->nrrd->data);

    for( k=0; k<kdim; k++ ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
	  int index = (i * jdim + j) * kdim + k;
	
	  ifield->set_value( Vector( ptr[0][index],
				     ptr[1][index],
				     ptr[2][index] ), *inodeItr);
	
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
  } else {
    typename FIELD::mesh_type::Node::iterator inodeItr;

    imesh->begin( inodeItr );

    NTYPE *ptr[3];

    ptr[0] = (NTYPE *)(nHandles[data[0]]->nrrd->data);
    ptr[1] = (NTYPE *)(nHandles[data[1]]->nrrd->data);
    ptr[2] = (NTYPE *)(nHandles[data[2]]->nrrd->data);

    int npts = nHandles[data[0]]->nrrd->axis[1].size;

    for( int i=0; i<npts; i++ ) {

      ifield->set_value( Vector( ptr[0][i],
				 ptr[1][i],
				 ptr[2][i]),
			 *inodeItr);	
      ++inodeItr;
   }
  }

  return FieldHandle( ifield );
}

} // end namespace Fusion

#endif // NrrdFieldConverter_h
