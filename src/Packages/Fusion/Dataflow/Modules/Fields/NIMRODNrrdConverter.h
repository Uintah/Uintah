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

//    File   : NIMRODNrrdConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(NIMRODNrrdConverter_h)
#define NIMRODNrrdConverter_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <Teem/Core/Datatypes/NrrdData.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;



class NIMRODNrrdConverterMeshAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle mHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > grid,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mtd,
					    const unsigned int nt);
};


template< class MESH, class NTYPE >
class NIMRODNrrdConverterMeshAlgoT : public NIMRODNrrdConverterMeshAlgo
{
public:
  virtual void execute(MeshHandle mHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > grid,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap);
};

template< class MESH, class NTYPE >
void
NIMRODNrrdConverterMeshAlgoT< MESH, NTYPE >::execute(MeshHandle mHandle,
				     vector< NrrdDataHandle > nHandles,
				     vector< int > grid,
				     int idim, int jdim, int kdim,
				     int iwrap, int jwrap, int kwrap)
{
  MESH *imesh = (MESH *) mHandle.get_rep();
  typename MESH::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k;
  
  NTYPE *ptrR   = (NTYPE *)(nHandles[grid[0]]->nrrd->data);
  NTYPE *ptrZ   = (NTYPE *)(nHandles[grid[1]]->nrrd->data);
  NTYPE *ptrPhi = (NTYPE *)(nHandles[grid[2]]->nrrd->data);

  for( i=0; i<idim + iwrap; i++ ) {
    int iPhi = (i%idim);
    double cosphi = cos( ptrPhi[iPhi] );
    double sinphi = sin( ptrPhi[iPhi] );

    for( j=0; j<jdim + jwrap; j++ ) {
      for( k=0; k<kdim + kwrap; k++ ) {
	
	int iRZ = (j%jdim) * kdim + (k%kdim);

	// Grid
	float xVal =  ptrR[iRZ] * cosphi;
	float yVal = -ptrR[iRZ] * sinphi;
	float zVal =  ptrZ[iRZ];
	
	imesh->set_point(Point(xVal, yVal, zVal), *inodeItr);

	++inodeItr;
      }
    }
  }
}


class NIMRODNrrdConverterFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle fHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > data,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const unsigned int nt,
					    int rank);
};

template< class FIELD, class NTYPE >
class NIMRODNrrdConverterFieldAlgoScalar : public NIMRODNrrdConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle fHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > data,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap);
};


template< class FIELD, class NTYPE >
void
NIMRODNrrdConverterFieldAlgoScalar<FIELD, NTYPE>::execute(FieldHandle fHandle,
					   vector< NrrdDataHandle > nHandles,
					   vector< int > data,
					   int idim, int jdim, int kdim,
					   int iwrap, int jwrap, int kwrap)
{
  FIELD *ifield = (FIELD *) fHandle.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  NTYPE *ptr   = (NTYPE *)(nHandles[data[0]]->nrrd->data);

  register int i, j, k;
  
  for( i=0; i<idim + iwrap; i++ ) {
    for( j=0; j<jdim + jwrap; j++ ) {
      for( k=0; k<kdim + kwrap; k++ ) {
	
	int index = ((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim);
	
	// Value
	ifield->set_value( ptr[index], *inodeItr);
	
	++inodeItr;
      }
    }
  }
}

template< class FIELD, class NTYPE >
class NIMRODNrrdConverterFieldAlgoVector : public NIMRODNrrdConverterFieldAlgo
{
public:
  //! virtual interface.
  virtual void execute(FieldHandle fHandle,
		       vector< NrrdDataHandle > nHandles,
		       vector< int > data,
		       int idim, int jdim, int kdim,
		       int iwrap, int jwrap, int kwrap);
};


template< class FIELD, class NTYPE >
void
NIMRODNrrdConverterFieldAlgoVector<FIELD, NTYPE>::execute(FieldHandle fHandle,
					   vector< NrrdDataHandle > nHandles,
					   vector< int > data,
					   int idim, int jdim, int kdim,
					   int iwrap, int jwrap, int kwrap)
{
  FIELD *ifield = (FIELD *) fHandle.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FIELD::mesh_type::Node::iterator inodeItr;

  imesh->begin( inodeItr );

  register int i, j, k, iPhi;
  double xVal, yVal, zVal, cosPhi, sinPhi;
				  
  if( data.size() == 2 ) {
    NTYPE *ptrData = (NTYPE *)(nHandles[data[0]]->nrrd->data);
    NTYPE *ptrPhi  = (NTYPE *)(nHandles[data[1]]->nrrd->data);

    for( i=0; i<idim + iwrap; i++ ) {
      iPhi = (i%idim);
      
      cosPhi = cos( ptrPhi[iPhi] );
      sinPhi = sin( ptrPhi[iPhi] );
      
      for( j=0; j<jdim + jwrap; j++ ) {
	for( k=0; k<kdim + kwrap; k++ ) {
	
	  int index = (((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim)) * 3;
	
	  xVal =  ptrData[index] * cosPhi - ptrData[index+2] * sinPhi;
	  yVal = -ptrData[index] * sinPhi - ptrData[index+2] * cosPhi;
	  zVal =  ptrData[index+1];

	  // Value
	  ifield->set_value( Vector( xVal, yVal, zVal ), *inodeItr);
	
	  ++inodeItr;
	}
      }
    }
  } else if( data.size() == 4 ) {

    NTYPE *ptrR = NULL;
    NTYPE *ptrZ = NULL;
    NTYPE *ptrPhi = NULL;
    NTYPE *ptrGridPhi = (NTYPE *)(nHandles[data[3]]->nrrd->data);

    for( int ic=0; ic<3; ic++ ) {
	vector< string > dataset;
	
	nHandles[data[ic]]->get_tuple_indecies(dataset);

	if( dataset[0].find( "-R:Scalar" ) != std::string::npos ) {
	  ptrR = (NTYPE *)(nHandles[data[ic]]->nrrd->data);
	}
	else if( dataset[0].find( "-Z:Scalar" ) != std::string::npos ) {
	  ptrZ = (NTYPE *)(nHandles[data[ic]]->nrrd->data);
	}
	else if( dataset[0].find( "-PHI:Scalar" ) != std::string::npos ) {
	  ptrPhi = (NTYPE *)(nHandles[data[ic]]->nrrd->data);
	}
    }

    for( i=0; i<idim + iwrap; i++ ) {
      iPhi = (i%idim);

      cosPhi = cos( ptrGridPhi[iPhi] );
      sinPhi = sin( ptrGridPhi[iPhi] );

      for( j=0; j<jdim + jwrap; j++ ) {
	for( k=0; k<kdim + kwrap; k++ ) {
	
	  int index = ((i%idim) * jdim + (j%jdim)) * kdim + (k%kdim);
	
	  // Value
	  xVal =  ptrR[index] * cosPhi - ptrPhi[index] * sinPhi;
	  yVal = -ptrR[index] * sinPhi - ptrPhi[index] * cosPhi;
	  zVal =  ptrZ[index];

	  // Value
	  ifield->set_value( Vector( xVal, yVal, zVal ), *inodeItr);
	
	  ++inodeItr;
	}
      }
    }
  }

}

} // end namespace Fusion

#endif // NIMRODNrrdConverter_h
