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

#include <Core/Geometry/Point.h>
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
  enum { ZR = 0, PHI = 1, LIST = 2 };

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

//////////MESH CREATION//////////////////////////////
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

  vector< string > dataset;
  nHandles[mesh[ZR]]->get_tuple_indecies(dataset);

  int rank = 0;

  if( dataset[0].find( "ZR:Scalar" ) != string::npos &&
      nHandles[mesh[ZR]]->nrrd->dim == 3 ) {
    rank = nHandles[mesh[ZR]]->nrrd->axis[nHandles[mesh[ZR]]->nrrd->dim-1].size;
  } else if( dataset[0].find( "ZR:Vector" ) != string::npos &&
      nHandles[mesh[ZR]]->nrrd->dim == 2 ) {
    rank = 3;
  }

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

  nrrdWrap(nout->nrrd, ndata, nHandles[mesh[PHI]]->nrrd->type,
	   ndims+1, sink_size, nPhi*nZR);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, nrrdCenterNode);

  nout->nrrd->axis[0].label = strdup("XYZ:Vector");
  nout->nrrd->axis[1].label = strdup("Point List");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[mesh[PHI]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}


//////////CONNECTION CREATION//////////////////////////////
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
  if( 0 && modes[0] ) { // Wrapping
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

  nrrdWrap(nout->nrrd, ndata, nHandles[mesh[LIST]]->nrrd->type,
	   ndims+2, sink_size, cc, hex);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

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

//////////ASSUME CELL CENTERED FOR SCALAR DATA//////////////////////////////
template< class NTYPE >
class VULCANScalarConverterAlgoT : public VULCANConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
private:
  int get_weights(Point &p, Point *pts, double *w);

  inline double tri_area(const Point &a, const Point &b, const Point &c)
  {
    return Cross(b-a, c-a).length();
  }
};

template< class NTYPE >
int
VULCANScalarConverterAlgoT< NTYPE >::get_weights(Point &p, Point *pts, double *w)
{
  const Point &p0 = pts[0];
  const Point &p1 = pts[1];
  const Point &p2 = pts[2];
  const Point &p3 = pts[3];

  const double a0 = tri_area(p, p0, p1);
  if (a0 < 1.0e-6)
    {
      const Vector v0 = p0 - p1;
      const Vector v1 = p - p1;
      w[0] = Dot(v0, v1) / Dot(v0, v0);
      w[1] = 1.0 - w[0];
      w[2] = 0.0;
      w[3] = 0.0;
      return 4;
    }
    const double a1 = tri_area(p, p1, p2);
    if (a1 < 1.0e-6)
    {
      const Vector v0 = p1 - p2;
      const Vector v1 = p - p2;
      w[0] = 0.0;
      w[1] = Dot(v0, v1) / Dot(v0, v0);
      w[2] = 1.0 - w[1];
      w[3] = 0.0;
      return 4;
    }
    const double a2 = tri_area(p, p2, p3);
    if (a2 < 1.0e-6)
    {
      const Vector v0 = p2 - p3;
      const Vector v1 = p - p3;
      w[0] = 0.0;
      w[1] = 0.0;
      w[2] = Dot(v0, v1) / Dot(v0, v0);
      w[3] = 1.0 - w[2];
      return 4;
    }
    const double a3 = tri_area(p, p3, p0);
    if (a3 < 1.0e-6)
    {
      const Vector v0 = p3 - p0;
      const Vector v1 = p - p0;
      w[1] = 0.0;
      w[2] = 0.0;
      w[3] = Dot(v0, v1) / Dot(v0, v0);
      w[0] = 1.0 - w[3];
      return 4;
    }

    w[0] = tri_area(p0, p1, p2) / (a0 * a3);
    w[1] = tri_area(p1, p2, p0) / (a1 * a0);
    w[2] = tri_area(p2, p3, p1) / (a2 * a1);
    w[3] = tri_area(p3, p0, p2) / (a3 * a2);

    const double suminv = 1.0 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= suminv;
    w[1] *= suminv;
    w[2] *= suminv;
    w[3] *= suminv;
    return 4;
  }

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

  int nCon = nHandles[mesh[LIST]]->nrrd->axis[1].size; // Connection list
  int nPhi = nHandles[mesh[PHI ]]->nrrd->axis[1].size; // Phi
  int nZR  = nHandles[mesh[ZR  ]]->nrrd->axis[1].size; // Points

  int nVal = nHandles[data[0]]->nrrd->axis[1].size;    // Values

  NTYPE *ptrZR  = (NTYPE *)(nHandles[mesh[ZR]]->nrrd->data);
  NTYPE *ptrCon = (NTYPE *)(nHandles[mesh[LIST]]->nrrd->data);
  NTYPE *ptr    = (NTYPE *)(nHandles[data[0]]->nrrd->data);

  NTYPE* ndata = scinew NTYPE[nPhi*nZR];

  double* wt = scinew double[nZR];

  register int i, kj, cc=0;

  double w[4];
  Point p, pts[4];

  for( kj=0; kj<nPhi*nZR; kj++ )
    ndata[kj] = 0;

  for( kj=0; kj<nZR; kj++ )
    wt[kj] = 0;

  int rank = nHandles[mesh[LIST]]->nrrd->axis[nHandles[mesh[LIST]]->nrrd->dim-1].size;

  for( kj=0; kj<nCon; kj++ ) {
    int c0 = (int) ptrCon[kj*rank];
    int c1 = (int) ptrCon[kj*rank+1];
    int c2 = (int) ptrCon[kj*rank+2];
    int c3 = (int) ptrCon[kj*rank+3];
    /*
    pts[0] = Point( ptrZR[c0+1], 0, ptrZR[c0] );
    pts[1] = Point( ptrZR[c1+1], 0, ptrZR[c1] );
    pts[2] = Point( ptrZR[c2+1], 0, ptrZR[c2] );
    pts[3] = Point( ptrZR[c3+1], 0, ptrZR[c3] );

    p = Point( (ptrZR[c0+1]+ptrZR[c1+1]+ptrZR[c2+1]+ptrZR[c3+1])/4.0,
	       0,
	       (ptrZR[c0]+ptrZR[c1]+ptrZR[c2]+ptrZR[c3])/4.0 );

    get_weights( p, pts, w );
    */

    w[0] = w[1] = w[2] = w[3] = 0.25;

    ndata[ c0 ] += w[0] * ptr[kj];
    ndata[ c1 ] += w[1] * ptr[kj];
    ndata[ c2 ] += w[2] * ptr[kj];
    ndata[ c3 ] += w[3] * ptr[kj];

    wt[ c0 ] += w[0];
    wt[ c1 ] += w[1];
    wt[ c2 ] += w[2];
    wt[ c3 ] += w[3];
  }


  cc = 0;
  for( i=0; i<nPhi; i++ ) {
    for( kj=0; kj<nZR; kj++ ) {
      
      if( i == 0 )
	ndata[cc] /= wt[kj];
      else
	ndata[cc] = ndata[kj];

      cc++;
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	   ndims+1, sink_size, nPhi*nZR);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, nrrdCenterNode);

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


//////////ASSUME NODE CENTERED FOR REALSPACE//////////////////////////////
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

  NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);
  NTYPE *ptrZR      = (NTYPE *)(nHandles[data[ZR ]]->nrrd->data);

  int rank = 0;

  vector< string > dataset;
  nHandles[data[ZR]]->get_tuple_indecies(dataset);

  if( dataset[0].find( "ZR:Scalar" ) != string::npos &&
      nHandles[data[ZR]]->nrrd->dim == 3 ) {
    rank = nHandles[data[ZR]]->nrrd->axis[nHandles[data[ZR]]->nrrd->dim-1].size;
  } else if( dataset[0].find( "ZR:Vector" ) != string::npos &&
      nHandles[data[ZR]]->nrrd->dim == 2 ) {
    rank = 3;
  }

  for( i=0; i<nPhi; i++ ) {
    double cosPhi = cos( ptrMeshPhi[i] );
    double sinPhi = sin( ptrMeshPhi[i] );

    for( kj=0; kj<nZR; kj++ ) {

      // Vector
      ndata[cc*3  ] =  ptrZR[kj*rank+1] * cosPhi;   // X
      ndata[cc*3+1] = -ptrZR[kj*rank+1] * sinPhi;   // Y
      ndata[cc*3+2] =  ptrZR[kj*rank  ];            // Z

      ++cc;
    }
  }

  nrrdWrap(nout->nrrd, ndata, nHandles[data[ZR]]->nrrd->type,
	   ndims+1, sink_size, nPhi*nZR);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, nrrdCenterNode);

  nout->nrrd->axis[0].label = strdup("XYZ:Vector");
  nout->nrrd->axis[1].label = strdup("Domain");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[ZR]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );

  return  NrrdDataHandle( nout );	
}

} // end namespace Fusion

#endif // VULCANConverter_h
