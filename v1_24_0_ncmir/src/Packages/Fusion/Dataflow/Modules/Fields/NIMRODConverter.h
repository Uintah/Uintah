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

#include <Core/Datatypes/NrrdData.h>

namespace Fusion {

using namespace SCIRun;

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
				 vector< int >& modes) = 0;
};

template< class NTYPE >
class NIMRODMeshConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
NIMRODMeshConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
					   vector< int >& mesh,
					   vector< int >& data,
					   vector< int >& modes)
{
  int rank = 3;
  int ndims = 3;

  int idim = nHandles[mesh[R  ]]->nrrd->axis[0].size; // Radial
  int jdim = nHandles[mesh[Z  ]]->nrrd->axis[1].size; // Theta
  int kdim = nHandles[mesh[PHI]]->nrrd->axis[0].size; // Phi

  NrrdData *nout = scinew NrrdData();

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim*3];

  register int i,j,k,cc = 0;

  NTYPE *ptrR   = (NTYPE *)(nHandles[mesh[R]]->nrrd->data);
  NTYPE *ptrZ   = (NTYPE *)(nHandles[mesh[Z]]->nrrd->data);
  NTYPE *ptrPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);
    
  if( modes[0] ) {
    for( k=0; k<kdim; k++ ) {
      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
      
	  int iRZ = i * jdim + j;
      
	  // Mesh
	  ndata[cc*3  ] = ptrPhi[k];
	  ndata[cc*3+1] = ptrR[iRZ];
	  ndata[cc*3+2] = ptrZ[iRZ];
	
	  ++cc;
	}
      }
    }
  } else {
    for( k=0; k<kdim; k++ ) {      
      double cosPhi = cos( ptrPhi[k] );
      double sinPhi = sin( ptrPhi[k] );

      for( i=0; i<idim; i++ ) {
	for( j=0; j<jdim; j++ ) {
	  
	  int iRZ = i * jdim + j;
      
	  // Mesh
	  ndata[cc*3  ] =  ptrR[iRZ] * cosPhi;
	  ndata[cc*3+1] = -ptrR[iRZ] * sinPhi;
	  ndata[cc*3+2] =  ptrZ[iRZ];
	
	  ++cc;
	}
      }
    }
  }

  string source;

  nHandles[mesh[PHI]]->get_property( string("Source"), source);

  if( source == string("MDSPlus") )
    nrrdWrap(nout->nrrd, ndata, nHandles[mesh[PHI]]->nrrd->type,
	     ndims+1, rank, idim, jdim, kdim );
  else if( source == string("HDF5") )
    nrrdWrap(nout->nrrd, ndata, nHandles[mesh[PHI]]->nrrd->type,
	     ndims+1, rank, idim, jdim, kdim);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter,
		  nrrdCenterNode, nrrdCenterNode,
		  nrrdCenterNode, nrrdCenterNode);

  string nrrdName;
  nHandles[mesh[PHI]]->get_property( "Name", nrrdName );

  string::size_type pos = nrrdName.find( "PHI:Scalar" );
  if( pos != string::npos )
    nrrdName.replace( pos, 10, "XYZ:Vector" );


  nout->nrrd->axis[0].kind  = nrrdKind3Vector;
  nout->nrrd->axis[0].label = strdup("Mesh Points");
  nout->nrrd->axis[1].label = strdup("Radial");
  nout->nrrd->axis[2].label = strdup("Theta");
  nout->nrrd->axis[3].label = strdup("Phi");


  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[mesh[PHI]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );
  nout->set_property( "Name", nrrdName, false );

  return  NrrdDataHandle( nout );	
}


template< class NTYPE >
class NIMRODScalarConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
NIMRODScalarConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
					     vector< int >& mesh,
					     vector< int >& data,
					     vector< int >& modes)
{
  int ndims = 3;

  int idim = nHandles[data[0]]->nrrd->axis[0].size; // Radial
  int jdim = nHandles[data[0]]->nrrd->axis[1].size; // Theta
  int kdim = nHandles[data[0]]->nrrd->axis[2].size; // Phi

  NrrdData *nout = scinew NrrdData();

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim];

  register int i,j,k,cc = 0;

  NTYPE *ptr = (NTYPE *)(nHandles[data[0]]->nrrd->data);
    
  for( k=0; k<kdim; k++ ) {
    for( j=0; j<jdim; j++ ) {
      for( i=0; i<idim; i++ ) {

	unsigned int index = (k * jdim + j) * idim + i;

	ndata[cc] =  ptr[index];
	++cc;
      }
    }
  }

  string source;

  nHandles[data[0]]->get_property( string("Source"), source);

  if( source == string("MDSPlus") )
    nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	     ndims, idim, jdim, kdim);
  else if( source == string("HDF5") )
    nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	     ndims, idim, jdim, kdim);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter,
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);

  string nrrdName;
  nHandles[data[0]]->get_property( "Name", nrrdName );

  nout->nrrd->axis[0].kind  = nrrdKindDomain;
  nout->nrrd->axis[0].label = strdup("Radial");
  nout->nrrd->axis[1].label = strdup("Theta");
  nout->nrrd->axis[2].label = strdup("Phi");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[0]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );
  nout->get_property( "Name", nrrdName );

  return  NrrdDataHandle( nout );	
}


template< class NTYPE >
class NIMRODRealSpaceConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
NIMRODRealSpaceConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
						vector< int >& mesh,
						vector< int >& data,
						vector< int >& modes)
{
  int rank = 0;
  int ndims = 3;

  int idim = nHandles[data[0]]->nrrd->axis[1].size; // Radial
  int jdim = nHandles[data[0]]->nrrd->axis[2].size; // Theta
  int kdim = nHandles[data[0]]->nrrd->axis[3].size; // Phi

  string nrrdName;
  if( data.size() == 1 ) {
    nHandles[data[0]]->get_property( "Name", nrrdName );
    if( nrrdName.find( ":Scalar" ) != string::npos )
      rank = 1;
    else if( nrrdName.find( ":Vector" ) != string::npos )
      rank = 3;
  } else if( data.size() == 3 ) {
    rank = 3;
  }

  NrrdData *nout = scinew NrrdData();

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim*rank];

  register int i,j,k,cc = 0;

  if( data.size() == 1 ) {
    NTYPE *ptr        = (NTYPE *)(nHandles[data[0  ]]->nrrd->data);  
    NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);

    if( rank == 1 ) {
      for( k=0; k<kdim; k++ ) {
	for( j=0; j<jdim; j++ ) {
	  for( i=0; i<idim; i++ ) {
	  
	    unsigned int index = (k * jdim + j) * idim + i;

	    // Value
	    ndata[cc] =  ptr[index];
	  
	    ++cc;
	  }
	}
      }

    } else if( rank == 3 ) {
      for( k=0; k<kdim; k++ ) {
	double cosPhi = cos( ptrMeshPhi[k] );
	double sinPhi = sin( ptrMeshPhi[k] );

	for( j=0; j<jdim; j++ ) {
	  for( i=0; i<idim; i++ ) {
	  
	    unsigned int index = (k * jdim + j) * idim + i;

	    // Value
	    ndata[cc*3  ] =  ptr[index*3+1] * cosPhi - ptr[index*3] * sinPhi;
	    ndata[cc*3+1] = -ptr[index*3+1] * sinPhi - ptr[index*3] * cosPhi;
	    ndata[cc*3+2] =  ptr[index*3+2];
	  
	    ++cc;
	  }
	}
      }
    }

  } else if( data.size() == 3 ) {
    NTYPE *ptrR   = (NTYPE *)(nHandles[data[R]]->nrrd->data);
    NTYPE *ptrZ   = (NTYPE *)(nHandles[data[Z]]->nrrd->data);
    NTYPE *ptrPhi = (NTYPE *)(nHandles[data[PHI]]->nrrd->data);
  
    NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);

    for( k=0; k<kdim; k++ ) {
      double cosPhi = cos( ptrMeshPhi[k] );
      double sinPhi = sin( ptrMeshPhi[k] );

      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {

	  unsigned int index = (k * jdim + j) * idim + i;

	  // Value
	  ndata[cc*3  ] =  ptrR[index] * cosPhi - ptrPhi[index] * sinPhi;
	  ndata[cc*3+1] = -ptrR[index] * sinPhi - ptrPhi[index] * cosPhi;
	  ndata[cc*3+2] =  ptrZ[index];
	  
	  ++cc;
	}
      }
    }
  }

  string source;

  nHandles[data[0]]->get_property( string("Source"), source);

  if( source == string("MDSPlus") )
    nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	     ndims+1, rank, idim, jdim, kdim);
  else if( source == string("HDF5") )
    nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	     ndims+1, rank, idim, jdim, kdim);

  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter,
		  nrrdCenterNode, nrrdCenterNode,
		  nrrdCenterNode, nrrdCenterNode);

  nHandles[data[0]]->get_property( "Name", nrrdName );

  string::size_type pos = nrrdName.find( "R:Scalar" );
  if( pos != string::npos )
    nrrdName.replace( pos, 10, "XYZ:Vector" );

  pos = nrrdName.find( "PHI-R-Z:Vector" );
  if( pos != string::npos )
    nrrdName.replace( pos, 14, "XYZ:Vector" );

  pos = nrrdName.find( "R-Z-PHI:Vector" );
  if( pos != string::npos )
    nrrdName.replace( pos, 14, "XYZ:Vector" );

  nout->nrrd->axis[0].kind = nrrdKind3Vector;

  nout->nrrd->axis[0].label = strdup("Vector Data");
  nout->nrrd->axis[1].label = strdup("Radial");
  nout->nrrd->axis[2].label = strdup("Theta");
  nout->nrrd->axis[3].label = strdup("Phi");

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[0]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );
  nout->set_property( "Name", nrrdName, false );

  return  NrrdDataHandle( nout );	
}

template< class NTYPE >
class NIMRODPerturbedConverterAlgoT : public NIMRODConverterAlgo
{
public:

  virtual NrrdDataHandle execute(vector< NrrdDataHandle >& nHandles,
				 vector< int >& mesh,
				 vector< int >& data,
				 vector< int >& modes);
};


template< class NTYPE >
NrrdDataHandle
NIMRODPerturbedConverterAlgoT< NTYPE >::execute(vector< NrrdDataHandle >& nHandles,
						vector< int >& mesh,
						vector< int >& data,
						vector< int >& modes)
{
  unsigned int rank, ndims = 3;
  int idim, jdim, kdim;

  string nrrdName;

  nHandles[data[0]]->get_property( "Name", nrrdName );

  if( data.size() == 2 ) {
    if( nrrdName.find( ":Scalar" ) != string::npos ) {
      idim = nHandles[data[0]]->nrrd->axis[0].size; // Radial
      jdim = nHandles[data[0]]->nrrd->axis[1].size; // Theta
      rank = 1;
    } else if( nrrdName.find( ":Vector" ) != string::npos ) {
      idim = nHandles[data[0]]->nrrd->axis[1].size; // Radial
      jdim = nHandles[data[0]]->nrrd->axis[2].size; // Theta
      rank = 3;
    }
  } else if( data.size() == 6 ) {
    idim = nHandles[data[0]]->nrrd->axis[0].size; // Radial
    jdim = nHandles[data[0]]->nrrd->axis[1].size; // Theta
    rank = 3;
  }


  kdim = nHandles[mesh[PHI]]->nrrd->axis[0].size; // Phi

  NrrdData *nout = scinew NrrdData();

  register int i,j,k,m,cc = 0;


  NTYPE *ptrs[data.size()];

  for( unsigned int i=0; i<data.size(); i++ )
    ptrs[i] = (NTYPE *)(nHandles[data[i]]->nrrd->data);

  NTYPE *ptrMeshPhi = (NTYPE *)(nHandles[mesh[PHI]]->nrrd->data);
  NTYPE *ptrMeshK   = (NTYPE *)(nHandles[mesh[K]]->nrrd->data);

  int nmodes = nHandles[mesh[K]]->nrrd->axis[0].size;

  NTYPE* ndata = scinew NTYPE[idim*jdim*kdim*rank];

  double mmin=+1.0e16;
  double mmax=-1.0e16;

  if( data.size() == 2 ) {
    for( k=0; k<kdim; k++ ) {
    
      double phi = ptrMeshPhi[k];

      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {	

	  for( unsigned int c=0; c<rank; c++ )
	    ndata[cc*rank+c] = 0.0;

	  for( m=0; m<nmodes; m++ ) {  // Mode loop.

	    if( modes[m] || modes[nmodes] ) {
	      unsigned int index = (m * jdim + j) * idim + i;
	
	      double angle = ptrMeshK[m] * phi; // Mode * phi slice.

	      for( unsigned int c=0; c<rank; c++ ) {
		ndata[cc*rank+c] += 2.0 * ( cos( angle ) * ptrs[0][index*rank+c] -
					    sin( angle ) * ptrs[1][index*rank+c] );

		if( ptrs[0][index*rank+c] < mmin )
		  mmin = ptrs[0][index*rank+c];
		if( ptrs[1][index*rank+c] < mmin )
		  mmin = ptrs[1][index*rank+c];
		if( ptrs[0][index*rank+c] > mmax )
		  mmax = ptrs[0][index*rank+c];
		if( ptrs[1][index*rank+c] > mmax )
		  mmax = ptrs[1][index*rank+c];
	      }
	    }
	  }

	  ++cc;
	}
      }
    }
  }
  else if( data.size() == 6 ) {
    for( k=0; k<kdim; k++ ) {
    
      double phi = ptrMeshPhi[k];

      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {	
	  for( m=0; m<nmodes; m++ ) {  // Mode loop.

	    for( unsigned int c=0; c<rank; c++ )
	      ndata[cc*rank+c] = 0.0;

	    if( modes[m] || modes[nmodes] ) {
	      unsigned int index = (m * jdim + j) * idim + i;
	
	      double angle = ptrMeshK[m] * phi; // Mode * phi slice.

	      for( unsigned int c=0; c<rank; c++ )
		ndata[cc*rank+c] += 2.0 * ( cos( angle ) * ptrs[c  ][index] -
					   sin( angle ) * ptrs[c+3][index] );
	    }
	  }

	  ++cc;
	}
      }
    }
  }

  string source;

  nHandles[data[0]]->get_property( string("Source"), source);

  if( rank == 1 ) {
    if( source == string("MDSPlus") )
      nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	       ndims, idim, jdim, kdim);
    else if( source == string("HDF5") )
      nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	       ndims, idim, jdim, kdim);

    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter,
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
  } else {
    if( source == string("MDSPlus") )
      nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	       ndims+1, rank, kdim, idim, jdim);
    else if( source == string("HDF5") )
      nrrdWrap(nout->nrrd, ndata, nHandles[data[0]]->nrrd->type,
	       ndims+1, rank, idim, jdim, kdim);

    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter,
		    nrrdCenterNode, nrrdCenterNode,
		    nrrdCenterNode, nrrdCenterNode);
  }

  nHandles[data[0]]->get_property( "Name", nrrdName );

  char tmpstr[12];
  
  if( modes[nmodes] == 1 )
    sprintf( tmpstr,"SUM-ALL" );
  else
    sprintf( tmpstr,"SUM-MODE" );

  nrrdName.replace( nrrdName.find( "REAL" ), 4, tmpstr );

  string::size_type pos = nrrdName.find( "R:Scalar" );
  if( pos != string::npos )
    nrrdName.replace( pos, 10, "XYZ:Vector" );

  if( rank == 1 ) {
    nout->nrrd->axis[0].kind  = nrrdKindDomain;
    nout->nrrd->axis[0].label = strdup("Radial");
    nout->nrrd->axis[1].label = strdup("Theta");
    nout->nrrd->axis[2].label = strdup("Phi");
  } else {
    nout->nrrd->axis[0].kind  = nrrdKind3Vector; 
    nout->nrrd->axis[0].label = strdup("Vector Data");
    nout->nrrd->axis[1].label = strdup("Radial");
    nout->nrrd->axis[2].label = strdup("Theta");
    nout->nrrd->axis[3].label = strdup("Phi");
  }

  *((PropertyManager *)nout) =
    *((PropertyManager *)(nHandles[data[0]].get_rep()));

  nout->set_property( "Coordinate System", string("Cartesian - XYZ"), false );
  nout->set_property( "Name", nrrdName, false );

  return  NrrdDataHandle( nout );	
}


} // end namespace Fusion

#endif // NIMRODConverter_h
