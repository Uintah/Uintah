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

//    File   : VULCANConverter.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#if !defined(VULCANConverter_h)
#define VULCANConverter_h

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

class VULCANConverterAlgo : public DynamicAlgoBase
{
protected:
  enum { NONE = 0, MESH = 1, SCALAR = 2, REALSPACE = 4, CONNECTION = 8 };
  enum { OMEGA = 0, ZR = 1, PHI = 2, LIST = 4 };

public:
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info( const string converter,
					     const unsigned int ntype );

public:
  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes) = 0;
};

template< class NTYPE >
class VULCANMeshConverterAlgoT : public VULCANConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
VULCANMeshConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
					   vector< int >& mesh,
					   vector< int >& data,
					   vector< int >& modes)
{
  int sink_size = 1;
  int ndims = 1;

  NrrdData *nout = scinew NrrdData(false);

  register int i, kj, cc = 0;

  NTYPE *ptrZR  = (NTYPE *)(nHandles[mesh[ZR]]->nrrd->data);
  NTYPE *ptrPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);

  
  int nZR  = nHandles[mesh[ZR ]]->nrrd->axis[1].size; // Points
  int nPhi = nHandles[mesh[PHI]]->nrrd->axis[1].size; // Phi

  NTYPE* ndata = scinew NTYPE[nPhi*nZR*3];

  int rank = nHandles[mesh[ZR]]->nrrd->axis[nHandles[mesh[ZR]]->nrrd->dim-1].size;

  // Mesh uprolling.
  if( modes[0] ) {
    for( i=0; i<nPhi; i++ ) {
      for( kj=0; kj<nZR; kj++ ) {
      	  // Mesh
	ndata[cc*3  ] = ptrPhi[i];           // Phi
	ndata[cc*3+1] = ptrZR[kj*rank + 1];  // R
	ndata[cc*3+2] = ptrZR[kj*rank + 0];  // Z
	
	++cc;
      }
    }
  } else {
    for( i=0; i<nPhi; i++ ) {
      double cosPhi = cos( ptrPhi[i] );
      double sinPhi = sin( ptrPhi[i] );
      
      for( kj=0; kj<nZR; kj++ ) {      
	
	// Mesh
	ndata[cc*3  ] =  ptrZR[kj*rank+1] * cosPhi;   // X
	ndata[cc*3+1] = -ptrZR[kj*rank+1] * sinPhi;   // Y
	ndata[cc*3+2] =  ptrZR[kj*rank+0];            // Z
	
	++cc;
      }
    }
  }

  string source;

  nHandles[mesh[PHI]]->get_property( string("Source"), source);

  nrrdWrap(nout->nrrd, ndata, nHandles[mesh[PHI]]->nrrd->type,
	   ndims+1, sink_size, nPhi*nZR);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[mesh[PHI]]->get_tuple_indecies(dataset);

  dataset[0].replace( dataset[0].find( "PHI:Scalar" ), 10, "XYZ:Vector" );

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Point List");


  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[mesh[PHI]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}


////////////////////////////////////////
template< class NTYPE >
class VULCANConnectionConverterAlgoT : public VULCANConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
VULCANConnectionConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
					   vector< int >& mesh,
					   vector< int >& data,
					   vector< int >& modes)
{
  int sink_size = 1;
  int ndims = 1;
  int hex = 8;


  NrrdData *nout = scinew NrrdData(false);

  register int i, j, kj, cc = 0;

  NTYPE *ptrCon = (NTYPE *)(nHandles[mesh[LIST]]->nrrd->data);
  
  int nCon = nHandles[mesh[LIST]]->nrrd->axis[1].size; // Connection list
  int nPhi = nHandles[mesh[PHI ]]->nrrd->axis[1].size; // Phi
  int nZR  = nHandles[mesh[ZR  ]]->nrrd->axis[1].size; // Points

  NTYPE* ndata = scinew NTYPE[nPhi*nCon*hex];

  int rank = nHandles[mesh[LIST]]->nrrd->axis[nHandles[mesh[LIST]]->nrrd->dim-1].size;

  // Mesh
  if( modes[0] ) { // Wrapping
    for( i=0, j=1; i<nPhi; i++, j++ ) {
      for( kj=0; kj<nCon; kj++ ) {      
	
	ndata[cc*hex  ] = ((int) ptrCon[kj*rank + 0] + i*nZR) % (nPhi*nZR);
	ndata[cc*hex+1] = ((int) ptrCon[kj*rank + 1] + i*nZR) % (nPhi*nZR);
	ndata[cc*hex+2] = ((int) ptrCon[kj*rank + 2] + i*nZR) % (nPhi*nZR);
	ndata[cc*hex+3] = ((int) ptrCon[kj*rank + 3] + i*nZR) % (nPhi*nZR);
	
	ndata[cc*hex+4] = ((int) ptrCon[kj*rank + 0] + j*nZR) % (nPhi*nZR);
	ndata[cc*hex+5] = ((int) ptrCon[kj*rank + 1] + j*nZR) % (nPhi*nZR);
	ndata[cc*hex+6] = ((int) ptrCon[kj*rank + 2] + j*nZR) % (nPhi*nZR);
	ndata[cc*hex+7] = ((int) ptrCon[kj*rank + 3] + j*nZR) % (nPhi*nZR);

	++cc;
      }
    }
  } else {
    for( i=0, j=1; i<nPhi-1; i++, j++ ) {
      for( kj=0; kj<nCon; kj++ ) {      
	
	// Mesh
	ndata[cc*hex  ] = ((int) ptrCon[kj*rank + 0] + i*nZR);
	ndata[cc*hex+1] = ((int) ptrCon[kj*rank + 1] + i*nZR);
	ndata[cc*hex+2] = ((int) ptrCon[kj*rank + 2] + i*nZR);
	ndata[cc*hex+3] = ((int) ptrCon[kj*rank + 3] + i*nZR);
	
	ndata[cc*hex+4] = ((int) ptrCon[kj*rank + 0] + j*nZR);
	ndata[cc*hex+5] = ((int) ptrCon[kj*rank + 1] + j*nZR);
	ndata[cc*hex+6] = ((int) ptrCon[kj*rank + 2] + j*nZR);
	ndata[cc*hex+7] = ((int) ptrCon[kj*rank + 3] + j*nZR);

	++cc;
      }
    }
  }

  string source;

  nHandles[mesh[LIST]]->get_property( string("Source"), source);

  nrrdWrap(nout->nrrd, ndata, nHandles[mesh[LIST]]->nrrd->type,
	   ndims+2, sink_size, nPhi*nCon, hex);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[mesh[LIST]]->get_tuple_indecies(dataset);

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Cells");
  nout->nrrd->axis[2].label = strdup("Connections");


  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[mesh[LIST]].get_rep()));

  nout->set_property( "Cell Type", string("Hex"), false );

  return  NrrdDataHandle( nout );	
}

////////////////////////////////////////
template< class NTYPE >
class VULCANScalarConverterAlgoT : public VULCANConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};

template< class NTYPE >
NrrdDataHandle
VULCANScalarConverterAlgoT< NTYPE >::
execute(vector< NrrdDataHandle >& nHandles,
	vector< int >& mesh,
	vector< int >& data,
	vector< int >& modes)
{
  int sink_size = 1;
  int ndims = 1;

  NrrdData *nout = scinew NrrdData(false);

  int nPhi = nHandles[mesh[PHI]]->nrrd->axis[1].size; // Phi
  int nZR  = nHandles[data[0]]->nrrd->axis[1].size;   // Points

  NTYPE* ndata = scinew NTYPE[nPhi*nZR];

  register int i, kj, cc=0;

  NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);
    
  for( i=0; i<nPhi; i++ ) {
    double cosPhi = cos( ptrMeshPhi[i] );
    double sinPhi = sin( ptrMeshPhi[i] );

    for( kj=0; kj<nZR; kj++ ) {
      
      ndata[cc] =  ptr[kj];
      cc++;
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	   ndims+1, sink_size, nPhi*nZR);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[data[0]]->get_tuple_indecies(dataset);

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Data");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[0]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );
}


template< class NTYPE >
class VULCANRealSpaceConverterAlgoT : public VULCANConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
VULCANRealSpaceConverterAlgoT< NTYPE >::
execute(vector< NrrdDataHandle >& nHandles,
	vector< int >& mesh,
	vector< int >& data,
	vector< int >& modes)
{
  int sink_size = 1;
  int ndims = 1;

  int nPhi = nHandles[mesh[PHI]]->nrrd->axis[1].size; // Phi
  int nZR  = nHandles[data[ZR ]]->nrrd->axis[1].size; // Radial
  
  NrrdData *nout = scinew NrrdData(false);

  NTYPE* ndata = scinew NTYPE[nPhi*nZR*3];

  register int i,kj,cc = 0;

  NTYPE *ptrZR      = (NTYPE *)(nHandles[data[ZR   ]]->nrrd->data);
//  NTYPE *ptrOMEGA   = (NTYPE *)(nHandles[data[OMEGA]]->nrrd->data);
  NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI  ]]->nrrd->data);
//  NTYPE *ptrMeshZR  = (NTYPE *)(nHandles[mesh[ZR   ]]->nrrd->data);

  int rank = nHandles[data[ZR]]->nrrd->axis[nHandles[data[ZR]]->nrrd->dim-1].size;

  for( i=0; i<nPhi; i++ ) {
    double cosPhi = cos( ptrMeshPhi[i] );
    double sinPhi = sin( ptrMeshPhi[i] );

    for( kj=0; kj<nZR; kj++ ) {
      
      // Value
      ndata[cc*3  ] =  ptrZR[kj*rank+1] * cosPhi;   // X
      ndata[cc*3+1] = -ptrZR[kj*rank+1] * sinPhi;   // Y
      ndata[cc*3+2] =  ptrZR[kj*rank+0];            // Z

      //ptrMeshZR[kj+1] * ptrOMEGA[kj];
      
      ++cc;
    }
  }

  string source;

  nHandles[data[ZR]]->get_property( string("Source"), source);

  nrrdWrap(nout->nrrd, ndata, nHandles[data[ZR]]->nrrd->type,
	   ndims+1, sink_size, nPhi*nZR);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  vector< string > dataset;

  nHandles[data[ZR]]->get_tuple_indecies(dataset);

  dataset[0].replace( dataset[0].find( "ZR:Scalar" ), 10, "XYZ:Vector" );

  nout->nrrd->axis[0].label = strdup(dataset[0].c_str());
  nout->nrrd->axis[1].label = strdup("Data");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[ZR]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}

} // end namespace Fusion

#endif // VULCANConverter_h
