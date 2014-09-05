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

//    File   : NIMRODConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(NIMRODConverter_h)
#define NIMRODConverter_h

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

class NIMRODConverterAlgo : public DynamicAlgoBase
{
protected:
  enum { NONE = 0, MESH = 1, SCALAR = 2, REALSPACE = 4, PERTURBED = 8 };
  enum { R = 0, Z = 1, PHI = 2, K = 3 };

public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info( const string converter,
					     const unsigned int ntype );

public:
  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes,
				 int idim, int jdim, int kdim) = 0;
};

template< class NTYPE >
class NIMRODMeshConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes,
				 int idim, int jdim, int kdim);
};


template< class NTYPE >
NrrdDataHandle
NIMRODMeshConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
					   vector< int >& mesh,
					   vector< int >& data,
					   vector< int >& modes,
					   int idim, int jdim, int kdim)
{
  int sink_size = 1;
  int ndims = 3;

  NrrdData *nout = scinew NrrdData(false);

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim*3];

  register int i,j,k,cc = 0;

  NTYPE *ptrR   = (NTYPE *)(nHandles[mesh[R]]->nrrd->data);
  NTYPE *ptrZ   = (NTYPE *)(nHandles[mesh[Z]]->nrrd->data);
  NTYPE *ptrPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);
    
  if( modes[0] ) {
    for( i=0; i<idim; i++ ) {
      for( k=0; k<kdim; k++ ) {
	for( j=0; j<jdim; j++ ) {
      
	  int iRZ = k * jdim + j;
      
	  // Mesh
	  ndata[cc*3  ] = ptrPhi[i];
	  ndata[cc*3+1] = ptrR[iRZ];
	  ndata[cc*3+2] = ptrZ[iRZ];
	
	  ++cc;
	}
      }
    }
  } else {
    for( i=0; i<idim; i++ ) {
      double cosPhi = cos( ptrPhi[i] );
      double sinPhi = sin( ptrPhi[i] );
      
      for( k=0; k<kdim; k++ ) {
	for( j=0; j<jdim; j++ ) {
      
	  int iRZ = k * jdim + j;
      
	  // Mesh
	  ndata[cc*3  ] =  ptrR[iRZ] * cosPhi;
	  ndata[cc*3+1] = -ptrR[iRZ] * sinPhi;
	  ndata[cc*3+2] =  ptrZ[iRZ];
	
	  ++cc;
	}
      }
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[mesh[PHI]]->nrrd->type,
	   ndims+1, sink_size, idim, jdim, kdim);
  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[mesh[PHI]]->get_tuple_indecies(dataset);

  dataset[0].replace( dataset[0].find( "PHI:Scalar" ), 10, "XYZ:Vector" );

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Phi");
  nout->nrrd->axis[2].label = strdup("Theta");
  nout->nrrd->axis[3].label = strdup("Radial");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[mesh[PHI]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}


template< class NTYPE >
class NIMRODScalarConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes,
				 int idim, int jdim, int kdim);
};


template< class NTYPE >
NrrdDataHandle
NIMRODScalarConverterAlgoT< NTYPE >::
execute(vector< NrrdDataHandle >& nHandles,
	vector< int >& mesh,
	vector< int >& data,
	vector< int >& modes,
	int idim, int jdim, int kdim)
{
  int sink_size = 1;
  int ndims = 3;

  NrrdData *nout = scinew NrrdData(false);

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim];

  register int i,j,k,cc = 0;

  NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);
    
  for( i=0; i<idim; i++ ) {
    for( j=0; j<jdim; j++ ) {
      for( k=0; k<kdim; k++ ) {

	unsigned int index = (i * jdim + j) * kdim + k;
	
	ndata[cc] =  ptr[index];
	++cc;
      }
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	   ndims+1, sink_size, idim, jdim, kdim);
  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[data[0]]->get_tuple_indecies(dataset);

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Phi");
  nout->nrrd->axis[2].label = strdup("Theta");
  nout->nrrd->axis[3].label = strdup("Radial");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[0]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}


template< class NTYPE >
class NIMRODRealSpaceConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes,
				 int idim, int jdim, int kdim);
};


template< class NTYPE >
NrrdDataHandle
NIMRODRealSpaceConverterAlgoT< NTYPE >::
execute(vector< NrrdDataHandle >& nHandles,
	vector< int >& mesh,
	vector< int >& data,
	vector< int >& modes,
	int idim, int jdim, int kdim)
{
  int sink_size = 1;
  int ndims = 3;

  NrrdData *nout = scinew NrrdData(false);

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim*3];

  register int i,j,k,cc = 0;

  NTYPE *ptrR   = (NTYPE *)(nHandles[data[R]]->nrrd->data);
  NTYPE *ptrZ   = (NTYPE *)(nHandles[data[Z]]->nrrd->data);
  NTYPE *ptrPhi = (NTYPE *)(nHandles[data[PHI]]->nrrd->data);
  
  NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);

  for( i=0; i<idim; i++ ) {
    double cosPhi = cos( ptrMeshPhi[i] );
    double sinPhi = sin( ptrMeshPhi[i] );

    for( j=0; j<jdim; j++ ) {
      for( k=0; k<kdim; k++ ) {
      
	unsigned int index = (i * jdim + j) * kdim + k;

	// Value
	ndata[cc*3  ] =  ptrR[index] * cosPhi - ptrPhi[index] * sinPhi;
	ndata[cc*3+1] = -ptrR[index] * sinPhi - ptrPhi[index] * cosPhi;
	ndata[cc*3+2] =  ptrZ[index];
	  
	++cc;
      }
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[data[PHI]]->nrrd->type,
	   ndims+1, sink_size, idim, jdim, kdim);
  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[data[PHI]]->get_tuple_indecies(dataset);

  dataset[0].replace( dataset[0].find( "PHI:Scalar" ), 10, "XYZ:Vector" );

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Phi");
  nout->nrrd->axis[2].label = strdup("Theta");
  nout->nrrd->axis[3].label = strdup("Radial");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[PHI]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}

template< class NTYPE >
class NIMRODPerturbedConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes,
				 int idim, int jdim, int kdim);
};


template< class NTYPE >
NrrdDataHandle
NIMRODPerturbedConverterAlgoT< NTYPE >::
execute(vector< NrrdDataHandle >& nHandles,
	vector< int >& mesh,
	vector< int >& data,
	vector< int >& modes,
	int idim, int jdim, int kdim)
{
  unsigned int sink_size = 1;
  unsigned int ndims = 3;

  NrrdData *nout = scinew NrrdData(false);

  register int i,j,k,m,cc = 0;

  unsigned int rank = data.size() / 2;

  NTYPE *ptrs[data.size()];

  for( unsigned int i=0; i<data.size(); i++ )
    ptrs[i] = (NTYPE *)(nHandles[data[i]]->nrrd->data);

  NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);
  NTYPE *ptrMeshK   = (NTYPE *)(nHandles[mesh[K]]->nrrd->data);

  int nmodes = nHandles[mesh[K]]->nrrd->axis[1].size;

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim*rank];

  for( i=0; i<idim; i++ ) {

    double phi = ptrMeshPhi[i];

    for( j=0; j<jdim; j++ ) {
      for( k=0; k<kdim; k++ ) {
      	for( m=0; m<nmodes; m++ ) {  // Mode loop.

	  if( modes[m] || modes[nmodes] ) {
	    unsigned int index = (m * jdim + j) * kdim + k;
	
	    double angle = ptrMeshK[m] * phi; // Mode * phi slice.

	    for( unsigned int c=0; c<rank; c++ )
	      ndata[cc*rank+c] = 0;

	    for( unsigned int c=0; c<rank; c++ )
	      ndata[cc*rank+c] += 2.0 * ( cos( angle ) * ptrs[c     ][index] -
					  sin( angle ) * ptrs[c+rank][index] );
	  }
	}

	++cc;
      }
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	   ndims+1, sink_size, idim, jdim, kdim);
  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[data[0]]->get_tuple_indecies(dataset);

  char tmpstr[12];
  
  if( modes[nmodes] == 1 )
    sprintf( tmpstr,"SUM-ALL" );
  else
    sprintf( tmpstr,"SUM-MODE" );

  dataset[0].replace( dataset[0].find( "REAL" ), 4, tmpstr );

  string::size_type pos = dataset[0].find( "R:Scalar" );
  if( pos != string::npos )
    dataset[0].replace( pos, 10, "XYZ:Vector" );

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Phi");
  nout->nrrd->axis[2].label = strdup("Theta");
  nout->nrrd->axis[3].label = strdup("Radial");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[0]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}


} // end namespace Fusion

#endif // NIMRODConverter_h
