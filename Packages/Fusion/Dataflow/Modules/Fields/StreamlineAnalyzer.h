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


/*
 *  StreamlineAnalyzer.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   SCI Institute
 *   University of Utah
 *   September 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#if !defined(StreamlineAnalyzer_h)
#define StreamlineAnalyzer_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/StructQuadSurfField.h>

#include <sstream>
using std::ostringstream;

namespace Fusion {

using namespace std;
using namespace SCIRun;

class StreamlineAnalyzerAlgo : public DynamicAlgoBase
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle& slsrc,
		       FieldHandle& dst,
		       FieldHandle& pccsrc,
		       FieldHandle& pccdst,
		       FieldHandle& pcssrc,
		       FieldHandle& pcsdst,
		       vector< double > &planes,
		       unsigned int color,
		       unsigned int showIslands,
		       unsigned int overlaps,
		       unsigned int maxWindings,
		       unsigned int override,
		       vector< pair< unsigned int,
		       unsigned int > > &topology ) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd,
					    const TypeDescription *tftd,
					    const string otd);
};


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
class StreamlineAnalyzerAlgoT : public StreamlineAnalyzerAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle& src,
		       FieldHandle& dst,
		       FieldHandle& pccsrc,
		       FieldHandle& pccdst,
		       FieldHandle& pcssrc,
		       FieldHandle& pcsdst,
		       vector< double > &planes,
		       unsigned int color,
		       unsigned int showIslands,
		       unsigned int overlaps,
		       unsigned int maxWindings,
		       unsigned int override,
		       vector< pair< unsigned int,
		       unsigned int > > &topology );

protected:
  Point interpert( Point lastPt, Point currPt, double t ) {

    return Point( Vector( lastPt ) + Vector( currPt - lastPt ) * t );
  }

  bool  ccw( Vector v0, Vector v1 ) {
    
    if( v0.x() * v1.z() > v0.z() * v1.x() ) return true;         //  CCW
    
    if( v0.x() * v1.z() < v0.z() * v1.x() ) return false;       //  CW
    
    if( v0.x() * v1.x() < 0 || v0.z()*v1.z() < 0 ) return false; //  CW
    
    if( v0.x()*v0.x()+v0.z()*v0.z() >=
	v1.x()*v1.x()+v1.z()*v1.z() ) return false;              //  ON LINE
    
    return true;                                                 //  CCW
  }

  unsigned int factorial( unsigned int n0, unsigned int n1 ) {

    unsigned int min = n0 < n1 ? n0 : n1;

    for( unsigned int i=min; i>1; i-- )
      if( n0 % i == 0 && n1 % i == 0 )
	return i;

    return 0;
  }

  pair< pair< double, double >, pair< double, double > >
  FingerCheck( vector< Point >& points, unsigned int nbins );

  double
  SafetyFactor( vector< Point >& points, unsigned int maxWindings,
		pair< unsigned int, unsigned int > &safetyFactor,
		pair< unsigned int, unsigned int > &safetyFactor2 );

  double
  CentroidCheck2( vector< Point >& points, unsigned int nbins,
		  pair< unsigned int, unsigned int > &safetyFactor );

  double
  CentroidCheck( vector< Point >& points, unsigned int nbins,
		 pair< unsigned int, unsigned int > &safetyFactor );

  bool
  BoundingBoxCheck( vector< Point >& points, unsigned int nbins );

  unsigned int
  removeOverlaps( vector< vector < pair< Point, double > > > &bins,
		  unsigned int &nnodes,
		  unsigned int winding,
		  unsigned int twist,
		  bool CCWstreamline,
		  Vector Centroid );

  virtual void
  loadCurve( FieldHandle &field_h,
	     vector < pair< Point, double > > &nodes,
	     unsigned int nplanes,
	     unsigned int nbins,
	     unsigned int nnodes,
	     unsigned int plane,
	     unsigned int bin,
	     unsigned int color,
	     double color_value ) = 0;

  virtual void
  loadSurface( FieldHandle &field_h,
	       vector < pair< Point, double > > &nodes,
	       unsigned int nplanes,
	       unsigned int nbins,
	       unsigned int nnodes,
	       unsigned int plane,
	       unsigned int bin,
	       unsigned int color,
	       double color_value ) = 0;
};


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
class StreamlineAnalyzerAlgoScalar :
  public StreamlineAnalyzerAlgoT< IFIELD, OFIELD, PCFIELD, TYPE >
{
protected:  
  virtual void
  loadCurve( FieldHandle &field_h,
	     vector < pair< Point, double > > &nodes,
	     unsigned int nplanes,
	     unsigned int nbins,
	     unsigned int nnodes,
	     unsigned int plane,
	     unsigned int bin,
	     unsigned int color,
	     double color_value );

  virtual void
  loadSurface( FieldHandle &field_h,
	       vector < pair< Point, double > > &nodes,
	       unsigned int nplanes,
	       unsigned int nbins,
	       unsigned int nnodes,
	       unsigned int plane,
	       unsigned int bin,
	       unsigned int color,
	       double color_value );

};

template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
class StreamlineAnalyzerAlgoVector :
  public StreamlineAnalyzerAlgoT< IFIELD, OFIELD, PCFIELD, TYPE >
{
protected:  
  virtual void
  loadCurve( FieldHandle &field_h,
	     vector < pair< Point, double > > &nodes,
	     unsigned int nplanes,
	     unsigned int nbins,
	     unsigned int nnodes,
	     unsigned int plane,
	     unsigned int bin,
	     unsigned int color,
	     double color_value );

  virtual void
  loadSurface( FieldHandle &field_h,
	       vector < pair< Point, double > > &nodes,
	       unsigned int nplanes,
	       unsigned int nbins,
	       unsigned int nnodes,
	       unsigned int plane,
	       unsigned int bin,
	       unsigned int color,
	       double color_value );
};


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
void
StreamlineAnalyzerAlgoScalar<IFIELD, OFIELD, PCFIELD, TYPE>::
loadCurve( FieldHandle &field_h,
	   vector < pair< Point, double > > &nodes,
	   unsigned int nplanes,
	   unsigned int nbins,
	   unsigned int nnodes,
	   unsigned int plane,
	   unsigned int bin,
	   unsigned int color,
	   double color_value ) {

  CurveField<TYPE> *ofield = (CurveField<TYPE> *) field_h.get_rep();
  typename CurveField<TYPE>::mesh_handle_type omesh = ofield->get_typed_mesh();

  typename CurveField<TYPE>::mesh_type::Node::index_type n1, n2;

  n1 = omesh->add_node(nodes[0].first);
  ofield->resize_fdata();
  if( color == 0 )
    ofield->set_value( nodes[0].second, n1);
  else if( color == 1 )
    ofield->set_value( (TYPE) color_value, n1);
  else if( color == 2 )
    ofield->set_value( (TYPE) (0*nbins+bin), n1);
  else if( color == 3 )
    ofield->set_value( (TYPE) color_value, n1);
  else if( color == 4 )
    ofield->set_value( (TYPE) bin, n1);
  else if( color == 5 )
    ofield->set_value( (TYPE) 0, n1);
  else
    ofield->set_value( (TYPE) color_value, n1);
  
  for( unsigned int i=1; i<nnodes; i++ ) {
    n2 = omesh->add_node(nodes[i].first);
    ofield->resize_fdata();

    if( color == 0 )
      ofield->set_value(nodes[i].second, n2);
    else if( color == 1 )
      ofield->set_value( (TYPE) color_value, n2);
    else if( color == 2 )
      ofield->set_value( (TYPE) (i*nbins+bin), n2);
    else if( color == 3 )
      ofield->set_value( (TYPE) color_value, n2);
    else if( color == 4 )
      ofield->set_value( (TYPE) bin, n2);
    else if( color == 5 )
      ofield->set_value( (TYPE) i, n2);
    else
      ofield->set_value( (TYPE) color_value, n2);
    
    omesh->add_edge(n1, n2);
	      
    n1 = n2;
  }
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
void
StreamlineAnalyzerAlgoScalar<IFIELD, OFIELD, PCFIELD, TYPE>::
loadSurface( FieldHandle &field_h,
	     vector < pair< Point, double > > &nodes,
	     unsigned int nplanes,
	     unsigned int nbins,
	     unsigned int nnodes,
	     unsigned int plane,
	     unsigned int bin,
	     unsigned int color,
	     double color_value ) {
  
  StructQuadSurfField<TYPE> *ofield =
    (StructQuadSurfField<TYPE> *) field_h.get_rep();
  typename StructQuadSurfField<TYPE>::mesh_handle_type omesh =
    ofield->get_typed_mesh();

  typename StructQuadSurfField<TYPE>::mesh_type::Node::index_type n1;

  n1.mesh_ = omesh.get_rep();

  n1.j_ = nplanes * bin + plane;

  for( unsigned int i=0; i<nnodes; i++ ) {

    n1.i_ = i;

    omesh->set_point(nodes[i].first, n1);

    if( color == 0 )
      ofield->set_value( nodes[i].second, n1);
    else if( color == 1 )
      ofield->set_value( (TYPE) color_value, n1);
    else if( color == 2 )
      ofield->set_value( (TYPE) (i*nbins+bin), n1);
    else if( color == 3 )
      ofield->set_value( (TYPE) color_value, n1);
    else if( color == 4 )
      ofield->set_value( (TYPE) bin, n1);
    else if( color == 5 )
      ofield->set_value( (TYPE) i, n1);
    else
      ofield->set_value( (TYPE) color_value, n1);
  }
}

template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
void
StreamlineAnalyzerAlgoVector<IFIELD, OFIELD, PCFIELD, TYPE>::
loadCurve( FieldHandle &field_h,
	   vector < pair< Point, double > > &nodes,
	   unsigned int nplanes,
	   unsigned int nbins,
	   unsigned int nnodes,
	   unsigned int plane,
	   unsigned int bin,
	   unsigned int color,
	   double color_value ) {

  if( nnodes < 2 )
    return;

  CurveField<TYPE> *ofield = (CurveField<TYPE> *) field_h.get_rep();
  typename CurveField<TYPE>::mesh_handle_type omesh = ofield->get_typed_mesh();

  typename CurveField<TYPE>::mesh_type::Node::index_type n1, n2;

  Vector tangent(1,1,1);

  unsigned int i = 0;

  n1 = omesh->add_node(nodes[i].first);
  ofield->resize_fdata();
  tangent = ((Vector) nodes[i].first - (Vector) nodes[i+1].first);
  tangent.safe_normalize();

  if( color == 0 )
    tangent *= nodes[0].second;
  else if( color == 1 )
    tangent *= color_value;
  else if( color == 2 )
    tangent *= (i*nbins+bin);
  else if( color == 3 )
    tangent *= color_value;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (TYPE) tangent, n1);
  
  for( i=1; i<nnodes-1; i++ ) {
    n2 = omesh->add_node(nodes[i].first);
    ofield->resize_fdata();

    tangent = ( ((Vector) nodes[i].first - (Vector) nodes[i+1].first) + 
		((Vector) nodes[i-1].first - (Vector) nodes[i].first) )  / 2.0;
    tangent.safe_normalize();
    if( color == 0 )
      tangent *= nodes[0].second;
    else if( color == 1 )
      tangent *= color_value;
    else if( color == 2 )
      tangent *= (i*nbins+bin);
    else if( color == 3 )
      tangent *= color_value;
    else if( color == 4 )
      tangent *= bin;
    else if( color == 5 )
      tangent *= i;
    else
      tangent *= color_value;
 
    ofield->set_value( (TYPE) tangent, n2);
    
    omesh->add_edge(n1, n2);
	      
    n1 = n2;
  }

  n2 = omesh->add_node(nodes[i].first);
  ofield->resize_fdata();

  tangent = ((Vector) nodes[i-1].first - (Vector) nodes[i].first);
  tangent.safe_normalize();
  if( color == 0 )
    tangent *= nodes[0].second;
  else if( color == 1 )
    tangent *= color_value;
  else if( color == 2 )
    tangent *= (i*nbins+bin);
  else if( color == 3 )
    tangent *= color_value;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (TYPE) tangent, n2);
    
  omesh->add_edge(n1, n2);	      
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
void
StreamlineAnalyzerAlgoVector<IFIELD, OFIELD, PCFIELD, TYPE>::
loadSurface( FieldHandle &field_h,
	     vector < pair< Point, double > > &nodes,
	     unsigned int nplanes,
	     unsigned int nbins,
	     unsigned int nnodes,
	     unsigned int plane,
	     unsigned int bin,
	     unsigned int color,
	     double color_value ) {
  
  if( nnodes < 2 )
    return;

  StructQuadSurfField<TYPE> *ofield =
    (StructQuadSurfField<TYPE> *) field_h.get_rep();
  typename StructQuadSurfField<TYPE>::mesh_handle_type omesh =
    ofield->get_typed_mesh();

  typename StructQuadSurfField<TYPE>::mesh_type::Node::index_type n1;

  n1.mesh_ = omesh.get_rep();

  n1.j_ = nplanes * bin + plane;

  unsigned int i = 0;

  n1.i_ = i;

  omesh->set_point(nodes[i].first, n1);

  Vector tangent = ((Vector) nodes[i].first - (Vector) nodes[i+1].first);
  tangent.safe_normalize();
  if( color == 0 )
    tangent *= nodes[i].second;
  else if( color == 1 )
    tangent *= color_value;
  else if( color == 2 )
    tangent *= (i*nbins+bin);
  else if( color == 3 )
    tangent *= color_value;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (TYPE) tangent, n1);

  for( unsigned int i=1; i<nnodes-1; i++ ) {

    n1.i_ = i;

    omesh->set_point(nodes[i].first, n1);

    tangent = ( ((Vector) nodes[i].first - (Vector) nodes[i+1].first) + 
		((Vector) nodes[i-1].first - (Vector) nodes[i].first) )  / 2.0;
    tangent.safe_normalize();
    if( color == 0 )
      tangent *= nodes[i].second;
    else if( color == 1 )
      tangent *= color_value;
    else if( color == 2 )
      tangent *= (i*nbins+bin);
    else if( color == 3 )
      tangent *= color_value;
    else if( color == 4 )
      tangent *= bin;
    else if( color == 5 )
      tangent *= i;
    else
      tangent *= color_value;
 
    ofield->set_value( (TYPE) tangent, n1);
  }

  n1.i_ = i;

  omesh->set_point(nodes[i].first, n1);

  tangent = ((Vector) nodes[i-1].first - (Vector) nodes[i].first);
  tangent.safe_normalize();
  if( color == 0 )
    tangent *= nodes[i].second;
  else if( color == 1 )
    tangent *= color_value;
  else if( color == 2 )
    tangent *= (i*nbins+bin);
  else if( color == 3 )
    tangent *= plane;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (TYPE) tangent, n1);
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
double
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
SafetyFactor( vector< Point >& points, unsigned int maxWindings,
	      pair< unsigned int, unsigned int > &safetyFactor,
	      pair< unsigned int, unsigned int > &safetyFactor2 ) {

  unsigned int n = points.size();

  Vector centroid(0,0,0);

  // Get the centroid.
  for( unsigned int i=0; i<n; i++ )
    centroid += (Vector) points[i];

  centroid /= n;

  double angle = 0;
  double safetyFactorAve = 0;

  unsigned int skip = 1;

  double minDiff = 1.0e12;
  unsigned int winding, winding2;
  unsigned int twist, twist2;

  bool twistCCW = ccw(Vector( (Vector) points[0] - centroid ), 
		      Vector( (Vector) points[1] - centroid ));  

  for( unsigned int i=skip, j=1; i<n; i++, j++ ) {
    Vector v0( points[i-1] - centroid );
    Vector v1( points[i  ] - centroid );
  
    angle += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );

    safetyFactorAve += ((2.0*M_PI) / (angle / (double) j));

    double tmp = angle / (2.0*M_PI);

    if( i < maxWindings ) {
        cerr << i << "  SafetyFactor "
 	    << safetyFactorAve << "  "
 	    << tmp << endl;

      bool groupCCW = ccw(Vector( (Vector) points[0] - centroid ), 
			  Vector( (Vector) points[i] - centroid ));

      if( 1 || groupCCW == twistCCW ) {
	if( minDiff > fabs(ceil(tmp) - tmp) ) {
	  minDiff = fabs(ceil(tmp) - tmp);
	  winding2 = winding;
	  twist2 = twist;
	  winding = i;
	  twist = (unsigned int) ceil(tmp);
	}

	if( minDiff > fabs(floor(tmp) - tmp) ) {
	  minDiff = fabs(floor(tmp) - tmp);
	  winding2 = winding;
	  twist2 = twist;
	  winding = i;
	  twist = (unsigned int) floor(tmp);
	}
      }
    }
  }

  unsigned int fact;
  while( fact = factorial( winding, twist ) ) {
    winding /= fact;
    twist /= fact;
  }

  // Average safety factor
  safetyFactorAve /= (double) (n-skip);

  safetyFactor = pair< unsigned int, unsigned int >( winding, twist );
  safetyFactor2 = pair< unsigned int, unsigned int >( winding2, twist2 );

  return safetyFactorAve;
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
double
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
CentroidCheck2( vector< Point >& points, unsigned int nbins,
		pair< unsigned int, unsigned int > &safetyFactor ) {

  // Get the centroid and the standard deviation using the first and
  // last point.
  Vector centroid = Vector( points[0] + points[nbins] ) / 2.0;
  
  double centroidStdDev =
    sqrt( ((Vector) points[0    ]-centroid).length2() +
	  ((Vector) points[nbins]-centroid).length2() );


  // Get the centroid using all of the points.
  centroid = Vector(0,0,0);

  for( unsigned int i=0; i<points.size(); i++ )
    centroid += (Vector) points[i];

  centroid /= points.size();

  // Determine the number of twists by totalling up the angles
  // between each point and dividing by 2 PI.
  double angle = 0;
      
  for( unsigned int i=1; i<=nbins; i++ ) {      
    Vector v0 = (Vector) points[i-1] - centroid;
    Vector v1 = (Vector) points[i  ] - centroid;
      
    angle += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );
  }

  // Because the total will not be an exact integer add a delta so
  // the rounded value will be the correct integer.
  unsigned int twist =
    (unsigned int) ((angle +  M_PI/(double)nbins) / (2.0*M_PI));

  unsigned int winding = nbins;

  unsigned int fact;
  while( fact = factorial( winding, twist ) ) {
    winding /= fact;
    twist /= fact;
  }

  safetyFactor = pair< unsigned int, unsigned int > ( winding, twist );

  return centroidStdDev;
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
double
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
CentroidCheck( vector< Point >& points, unsigned int nbins,
	       pair< unsigned int, unsigned int > &safetyFactor ) {

  unsigned int winding = nbins;
  unsigned int twist = 0;
  double stddev = 0;

  if( points.size() > nbins ) {

    Vector centroid(0,0,0);

    unsigned int n = points.size();

    // Determine the number of valid bins.
    unsigned int vbins;

    if( points.size() >= 2*nbins )
      vbins = nbins;
    else
      vbins = points.size() - nbins;

    vector< Vector > localCentroids;
    vector< double > stddevs;

    localCentroids.resize(nbins);
    stddevs.resize(nbins);

    for( unsigned int i=0; i<nbins; i++ ) {
      localCentroids[i] = Vector(0,0,0);
      stddevs[i] = 0;
    }

    // Collect the centroids.
    for( unsigned int i=0; i<n; i++ )
      localCentroids[i % nbins] += (Vector) points[i];

    for( unsigned int i=0; i<nbins; i++ )
      localCentroids[i] /= ((n / nbins) + (i < (n % nbins) ? 1 : 0));

    for( unsigned int i=0; i<n; i++ )
      stddevs[i % nbins] +=
	((Vector) points[i]-localCentroids[i % nbins]).length2();

    for( unsigned int i=0; i<vbins; i++ )
      stddevs[i] = sqrt( stddevs[i] /
			 ((n / nbins) - 1 + (i < (n % nbins) ? 1 : 0)) );

    for( unsigned int i=0; i<nbins; i++ )
      centroid += localCentroids[i];

    for( unsigned int i=0; i<vbins; i++ )
      stddev += stddevs[i];

    centroid /= nbins;
    stddev   /= vbins;

    double angle = 0;
      
    // Determine the number of twists by totalling up the angles
    // between each of the group centroids and dividing by 2 PI.
    for( unsigned int i=1; i<=nbins; i++ ) {
      
      Vector v0 = localCentroids[i-1    ] - centroid;
      Vector v1 = localCentroids[i%nbins] - centroid;
      
      angle += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );
    }

    // Because the total will not be an exact integer add a delta so
    // the rounded value will be correct.
    twist = (unsigned int) ((angle +  M_PI/(double)nbins) / (2.0*M_PI));

  } else {

    stddev = 1.0e12;
    twist = 0;
  }

  unsigned int fact;
  while( fact = factorial( winding, twist ) ) {
    winding /= fact;
    twist /= fact;
  }

  safetyFactor = pair< unsigned int, unsigned int > ( winding, twist);

  return stddev;
}

template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
bool
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
BoundingBoxCheck( vector< Point >& points, unsigned nbins ) {

  unsigned int n = points.size();
  
  Vector minPt(1.0e12,1.0e12,1.0e12), maxPt(-1.0e12,-1.0e12,-1.0e12);

  // Collect the centroid and the bounding box.
  for( unsigned int i=0; i<n; i++ ) {
    if( minPt.x() > points[i].x() ) minPt.x( points[i].x() );
    if( minPt.y() > points[i].y() ) minPt.y( points[i].y() );
    if( minPt.z() > points[i].z() ) minPt.z( points[i].z() );

    if( maxPt.x() < points[i].x() ) maxPt.x( points[i].x() );
    if( maxPt.y() < points[i].y() ) maxPt.y( points[i].y() );
    if( maxPt.z() < points[i].z() ) maxPt.z( points[i].z() );
  }

  // If any section is greater than 33% of the bounding box diagonal
  // discard it.
  double len2 = (maxPt-minPt).length2();

  for( unsigned int i=0; i<n-nbins; i++ ) {
    if( ((Vector) points[i] - (Vector) points[i+nbins]).length2() >
	0.33 * len2 ) {

      return false;
    }
  }

  return true;
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
pair< pair< double, double >, pair< double, double > >
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
FingerCheck( vector< Point >& points, unsigned int nbins ) {

  double dotProdAveLength = 0;
  double dotProdAveAngle  = 0;

  double dotProdStdDevLength = 0;
  double dotProdStdDevAngle  = 0;

  unsigned int n = points.size();

  int cc = 0;

  for( unsigned int i=0; i<n-nbins; i++ ) {
    for( unsigned int j=i+1; j<i+nbins; j++ ) {

      Vector v0 = Vector( points[j] - points[i      ] );
      Vector v1 = Vector( points[j] - points[i+nbins] );
      
      double dotProd = Dot( v0, v1 );
      
      if( dotProd < 0 )
	return pair< pair< double, double >, pair< double, double > >
	  (pair< double, double >(-1,1.0e12),pair< double, double >(-1,1.0e12));
      
      dotProdAveLength += sqrt(dotProd / v0.length2());
      
      dotProd /= (v0.length() * v1.length());
      dotProdAveAngle += dotProd;
      
//     cerr << j
// 	 << " Len Dot Prod " << sqrt(dotProd) << "  "
// 	 << " Ang Dot Prod " << dotProd << endl;

      cc++;
    }
  }

  dotProdAveLength /= cc;
  dotProdAveAngle  /= cc;


  cc = 0;

  for( unsigned int i=0; i<n-nbins; i++ ) {
    for( unsigned int j=i+1; j<i+nbins; j++ ) {

      Vector v0 = Vector( points[j] - points[i      ] );
      Vector v1 = Vector( points[j] - points[i+nbins] );

      double dotProd = Dot( v0, v1 ) / (v0.length2() );

      dotProdStdDevLength += ((sqrt(dotProd) - dotProdAveLength) *
			      (sqrt(dotProd) - dotProdAveLength));

      dotProd *= (v0.length() / v1.length());
      dotProdStdDevAngle  += ((dotProd - dotProdAveAngle) *
			      (dotProd - dotProdAveAngle));

      cc++;
    }
  }

  dotProdStdDevLength = sqrt(dotProdStdDevLength/(cc-1));
  dotProdStdDevAngle  = sqrt(dotProdStdDevAngle /(cc-1));

  return pair< pair< double, double >, pair< double, double > >
    ( pair< double, double >( dotProdAveLength, dotProdStdDevLength ),
      pair< double, double >( dotProdAveAngle,  dotProdStdDevAngle  ) );
}


template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
void
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
execute(FieldHandle& ifield_h,
	FieldHandle& ofield_h,
	FieldHandle& ipccfield_h,
	FieldHandle& opccfield_h,
	FieldHandle& ipcsfield_h,
	FieldHandle& opcsfield_h,
	vector< double > &planes,
	unsigned int color,
	unsigned int showIslands,
	unsigned int overlaps,
	unsigned int maxWindings,
	unsigned int override,
	vector< pair< unsigned int, unsigned int > > &topology)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  typename OFIELD::mesh_type *omesh = scinew typename OFIELD::mesh_type();
  OFIELD *ofield = scinew OFIELD(omesh, ifield->basis_order());

  ofield_h = FieldHandle(ofield);

  bool curveField =
    (ofield->get_type_description(0)->get_name() == "CurveField");


  // Point Cloud Field of centroids.
  vector< vector < Vector > > baseCentroids;

  if( ipccfield_h.get_rep() ) {
    PCFIELD *ipcfield = (PCFIELD *) ipccfield_h.get_rep();
    typename PCFIELD::mesh_handle_type ipcmesh = ipcfield->get_typed_mesh();

    typename PCFIELD::mesh_type::Node::iterator inodeItr, inodeEnd;

    ipcmesh->begin( inodeItr );
    ipcmesh->end( inodeEnd );

    Point pt;

    while (inodeItr != inodeEnd) {
      ipcmesh->get_center(pt, *inodeItr);

      vector < Vector > baseCentroid;
      baseCentroid.push_back( (Vector) pt );
      baseCentroids.push_back( baseCentroid );

//       cerr << "input " << pt << endl;

      ++inodeItr;
    }
  }

  typename PCFIELD::mesh_type *opccmesh = scinew typename PCFIELD::mesh_type();
  PCFIELD *opccfield = scinew PCFIELD(opccmesh, ifield->basis_order());

  opccfield_h = FieldHandle(opccfield);

  // Point Cloud Field of Separatrices.
  vector< vector < Vector > > baseSeparatrices;

  if( ipcsfield_h.get_rep() ) {
    PCFIELD *ipcfield = (PCFIELD *) ipcsfield_h.get_rep();
    typename PCFIELD::mesh_handle_type ipcmesh = ipcfield->get_typed_mesh();

    typename PCFIELD::mesh_type::Node::iterator inodeItr, inodeEnd;

    ipcmesh->begin( inodeItr );
    ipcmesh->end( inodeEnd );

    Point pt;

    while (inodeItr != inodeEnd) {
      ipcmesh->get_center(pt, *inodeItr);

      vector < Vector > baseSeparatrix;
      baseSeparatrix.push_back( (Vector) pt );
      baseSeparatrices.push_back( baseSeparatrix );

//       cerr << "input " << pt << endl;

      ++inodeItr;
    }
  }

  typename PCFIELD::mesh_type *opcsmesh = scinew typename PCFIELD::mesh_type();
  PCFIELD *opcsfield = scinew PCFIELD(opcsmesh, ifield->basis_order());

  opcsfield_h = FieldHandle(opcsfield);


  // Input iterators
  typename IFIELD::fdata_type::iterator in = ifield->fdata().begin();

  typename IFIELD::mesh_type::Node::iterator inodeItr, inodeEnd;
  typename IFIELD::mesh_type::Node::index_type inodeNext;
  vector< typename IFIELD::mesh_type::Node::index_type > inodeGlobalStart;

  imesh->begin( inodeItr );
  imesh->end( inodeEnd );

  if(inodeItr == inodeEnd) return;

  topology.clear();

  vector< unsigned int > windings;
  vector< unsigned int > twists;

  vector< unsigned int > windings2;
  vector< unsigned int > twists2;

  Point lastPt, currPt;
  double lastAng, currAng;

  // Get the direction of the streamline winding.
  imesh->get_center(lastPt, *inodeItr);
  lastAng = atan2( lastPt.y(), lastPt.x() );

  ++inodeItr;
  if(inodeItr == inodeEnd) return;

  imesh->get_center(currPt, *inodeItr);
  currAng = atan2( currPt.y(), currPt.x() );

  bool CCWstreamline = (lastAng < currAng);


  unsigned int count = 0;
  
  // First find out how many windings it takes to get back to the
  // starting point at the Phi = 0 plane. 
  double plane = 0;

  vector< Point > points;

  imesh->begin( inodeItr );
  inodeNext = *inodeItr;

  Vector centroid;

  while (inodeItr != inodeEnd) {

    if( *inodeItr == inodeNext ) {

      points.clear();

//      cerr << "Starting new streamline winding "
//	   << count << "  " << *inodeItr << endl;

      imesh->get_center(lastPt, *inodeItr);
      lastAng = atan2( lastPt.y(), lastPt.x() );      
      ++inodeItr;
      
      ostringstream str;
      str << "Streamline " << count+1 << " Node Index";
      if( !ifield->get_property( str.str(), inodeNext ) )
	 inodeNext = *inodeEnd;
    }
	
    imesh->get_center(currPt, *inodeItr);
    currAng = atan2( currPt.y(), currPt.x() );
      
    // First look at only points that are in the correct plane.
    if( ( CCWstreamline && lastAng <= plane && plane <= currAng) ||
	(!CCWstreamline && currAng <= plane && plane <= lastAng) ) {
      double t;

      if( fabs(currAng-lastAng) > 1.0e-12 )
	t = (plane-lastAng) / (currAng-lastAng);
      else
	t = 0;

      Point point = interpert( lastPt, currPt, t );

      // Save the point found before the zero plane for CCW
      // streamlines or point found after the zero plane for CW
      // streamlines.
      if( points.size() == 0 ) {
	typename IFIELD::mesh_type::Node::iterator inodeFirst = inodeItr;

	--inodeFirst;

	if( CCWstreamline )
	  inodeGlobalStart.push_back( *inodeFirst );
	else
	  inodeGlobalStart.push_back( *inodeItr );
      }

      // If overriding skip all the other points.
      if( override ) {
	while (*inodeItr != inodeNext)
	  ++inodeItr;

	--inodeItr;
      } else
	points.push_back( point );
    }

    lastPt  = currPt;
    lastAng = currAng;
    
    ++inodeItr;

    // If about to start the next streamline figure out the number of
    // windings.
    if( *inodeItr == inodeNext ) {

//      cerr << "Getting last check" << endl;

      if( override ) {
	windings.push_back( override );

      } else if( points.size() < maxWindings/5 ) {
	cerr << "Streamline " << count << " has too few points ("
	     << points.size() << ") to determine the winding"
	     << "!!!!!!!!!!!!!!!!!!!" << endl;

	windings.push_back( 0 );

      } else {

	unsigned int desiredWinding = 0;

	if( baseCentroids.size() ) {

	  desiredWinding = (unsigned int) (*in);

	  in = ifield->fdata().begin();

	  typename IFIELD::mesh_type::Node::iterator inodeTmp;
	  imesh->begin( inodeTmp );

	  while( *inodeTmp != inodeNext ) {
	    ++inodeTmp;
	    ++in;
	  }
	}

	if( points.size() < maxWindings )
	  cerr << "Streamline " << count << " has too few points ("
	       << points.size() << ") to determine the winding accurately"
	       << "!!!!!!!!!!!!!!!!!!!" << endl;

	// Get the centroid for all of the points.
	centroid = Vector(0,0,0);

	for( unsigned int i=0; i<points.size(); i++ )
	  centroid += (Vector) points[i];

	centroid /= (double) points.size();


	// Get the direction of the twisting.
	bool twistCCW = ccw(Vector( (Vector) points[0] - centroid ), 
			    Vector( (Vector) points[1] - centroid ));  

	pair< unsigned int, unsigned int > safetyFactor;

	pair< unsigned int, unsigned int > Centroid2Min, Centroid2Min2;
	pair< unsigned int, unsigned int > CentroidMin,  CentroidMin2;

	pair< unsigned int, double > AngleMin( 0, 1.0e12);
	pair< unsigned int, double > LengthMin( 0, 1.0e12);
      
	pair< unsigned int, double > AngleMin2;
	pair< unsigned int, double > LengthMin2;

	double CentroidMinSD  = 1.0e12;
	double Centroid2MinSD = 1.0e12;
	double tmpMin;

	bool bboxCheck = false;

//	cerr << "Starting checks " << endl;

	// Find the best winding for each test.
	for( unsigned int i=2; i<maxWindings; i++ ) {
	  
	  bool groupCCW = ccw(Vector( (Vector) points[0] - centroid ), 
			      Vector( (Vector) points[i] - centroid ));

	  // If group direction is not the same as the twist direction
	  // skip it.
// 	  if( groupCCW != twistCCW )
// 	    continue;

	  // If any section is greater than 33% of the bounding box
	  // diagonal skip it.
	  if( !BoundingBoxCheck( points, i ) )
	    continue;

	  // Do the finger check.
	  pair< pair< double, double >, pair< double, double > > fingers =
	    FingerCheck( points, i );

	  if( fingers.first.first > 0 &&
	      fingers.first.second > 0 &&
	      fingers.second.first > 0 &&
	      fingers.second.second > 0 ) {

  	    cerr << i << " fingers "
 		 << fingers.first.first << "  "
 		 << fingers.first.second << "  "
 		 << fingers.second.first << "  "
 		 << fingers.second.second << "  "
		 << endl;

	    if( LengthMin.second > fingers.first.second ){
	      LengthMin2 = LengthMin;
	      
	      LengthMin.first  = i;
	      LengthMin.second = fingers.first.second;
	    }
	    
	    if( AngleMin.second > fingers.second.second ){
	      AngleMin2 = AngleMin;
	      
	      AngleMin.first  = i;
	      AngleMin.second = fingers.second.second;
	    }
	  } else
	    // If any of the dot products from the finger check are
	    // negative skip it.
	    continue;

	  tmpMin = CentroidCheck2( points, i, safetyFactor );

	  if( Centroid2MinSD > tmpMin ) {
	    Centroid2MinSD = tmpMin;
	    
	    Centroid2Min2 = Centroid2Min;
	    Centroid2Min  = safetyFactor;
	  }


	  tmpMin = CentroidCheck( points, i, safetyFactor );

	  if( CentroidMinSD > tmpMin ) {
	    CentroidMinSD = tmpMin;

	    CentroidMin2 = CentroidMin;
	    CentroidMin  = safetyFactor;
	  }

	  if( BoundingBoxCheck( points, i ) )
	    bboxCheck = true;
	}

	pair< unsigned int, unsigned int > sfCheck, sfCheck2;

	cerr << "Safety factor Check ";
	cerr << SafetyFactor( points, maxWindings, sfCheck, sfCheck2 );
	cerr << endl;



	bool print = true;
	bool showAll = true;

	// Never passed the bounding box check.
	if( !bboxCheck ) {
	  windings.push_back( 0 );
	  
	  cerr << "FAILED all bounding box checks *********************\n";

	} else if( 1 ) {

	  windings.push_back( sfCheck.first );
	  windings2.push_back( sfCheck2.first );

//	  twists.push_back( sfCheck.second );
	  twists2.push_back( sfCheck.second );

	// All primary checks agree.
	} else if( CentroidMin.first > 0 &&
		   CentroidMin.first == Centroid2Min.first &&
		   CentroidMin.first == LengthMin.first &&
		   CentroidMin.first == AngleMin.first ) {
	  windings.push_back( showAll ? CentroidMin.first : 0 );
	
	// All secondary checks agree.
	} else if( CentroidMin2.first > 0 &&
		   CentroidMin2.first == Centroid2Min2.first &&
		   CentroidMin2.first == LengthMin2.first &&
		   CentroidMin2.first == AngleMin2.first ) {
	  windings.push_back( showAll ? CentroidMin2.first : 0 );

	  if( CentroidMin.first &&
	      !BoundingBoxCheck( points, CentroidMin.first ) )
	    cerr << CentroidMin.first << " FAILED bounding box check -------------------------------\n";
	  if( Centroid2Min.first &&
	      !BoundingBoxCheck( points, Centroid2Min.first ) )
	    cerr << Centroid2Min.first << " FAILED bounding box check -------------------------------\n";
	  if( LengthMin.first &&
	      !BoundingBoxCheck( points, LengthMin.first ) )
	    cerr << LengthMin.first << " FAILED bounding box check -------------------------------\n";
	  if( AngleMin.first &&
	      !BoundingBoxCheck( points, AngleMin.first ) )
	    cerr << AngleMin.first << " FAILED bounding box check -------------------------------\n";

	  print = true;
	  
	// In this case probably an island where a latter point is
	// closer to the first point than a middle point.
	} else if( CentroidMin.first > 0 &&
		   CentroidMin.first < Centroid2Min.first &&
		   Centroid2Min.first % CentroidMin.first == 0 &&
		   CentroidMin.first == LengthMin.first &&
		   CentroidMin.first == AngleMin.first ) {
	  windings.push_back( showAll ? CentroidMin.first : 0 );
	  
	  // In this case the last three checks agree 
	} else if( Centroid2Min.first == LengthMin.first &&
		   Centroid2Min.first == AngleMin.first ) {
	  
	  // If the two bin indexs are factors of one another take the
	  // smallest index as the winding.
	  if( CentroidMin.first > 0 && Centroid2Min.first > 0 &&
	      Centroid2Min.first % CentroidMin.first == 0 )
	    windings.push_back( showAll ? CentroidMin.first : 0 );
	  
	  else if( CentroidMin.first > 0 && Centroid2Min.first > 0 &&
		   CentroidMin.first % Centroid2Min.first == 0 )
	    windings.push_back( showAll ? Centroid2Min.first : 0 );

	  else if( Centroid2Min.first > 0 ) {
	    windings.push_back( showAll ? Centroid2Min.first : 0 );
	    print = true;

	  } else {
	    windings.push_back( 0 );	  
	    print = true;
	  }

	} else {
	  windings.push_back( 0 );
	}

	// This is for island analysis.
	if( desiredWinding && windings[count] != desiredWinding )
	  windings[count] = 0;

	// Make sure the best can pass the bounding box check.
	if( windings[count] && !BoundingBoxCheck( points, windings[count] ) ) {
	    windings[count] = 0;

	    cerr << "Best FAILED bounding box check !!!!!!!!!!!!!!!!!!!!!\n";

	    print = true;
	}

	if( print ) {
	  cerr << " Safety Factor selection "
	       << sfCheck.first << "  "
	       << sfCheck.second << "  "
	       << sfCheck2.first << "  "
	       << sfCheck2.second << "  "
	       << endl;

	  cerr << " Centroid selection      "
	       << CentroidMin.first << "  "
	       << CentroidMin.second << "  "
	       << CentroidMin2.first << "  "
	       << CentroidMin2.second << "  "
	       << endl;

	  cerr << " Centroid2 selection     "
	       << Centroid2Min.first << "  "
	       << Centroid2Min.second << "  "
	       << Centroid2Min2.first << "  "
	       << Centroid2Min2.second << "  "
	       << endl;

	  cerr << " Length selection        "
	       << LengthMin.first << "  "
	       << LengthMin.second << "  "
	       << LengthMin2.first << "  "
	       << LengthMin2.second << "  "
	       << endl;

	  cerr << " Angle selection         "
	       << AngleMin.first << "  "
	       << AngleMin.second << "  "
	       << AngleMin2.first << "  "
	       << AngleMin2.second << "  "
	       << endl;

	  cerr << "End of streamline " << count
	       << "  it has a winding of " << windings[count] << endl;
	}
      }

      count++;
    }
  }


  // Now bin the points.
  for( unsigned int c=0; c<count; c++ ) {

    if( windings[c] == 0 ) {
      twists.push_back( 0 );

      pair< unsigned int, unsigned int > topo( 0, 0 );
      topology.push_back(topo);

      continue;
    }

    vector< vector < pair< Point, double > > > bins[planes.size()];
    
    for( unsigned int p=0; p<planes.size(); p++ ) {

      int bin = 0;

      // Go through the planes in the same direction as the streamline.
      if( CCWstreamline )
	plane = planes[p];
      else
	plane = planes[planes.size()-1-p];
      
      // The plane goes from 0 to 2PI but it is checked against atan2
      // which returns -PI to PI so adjust accordingly.
      if( plane > M_PI )
	plane -= 2.0 * M_PI;
      
      // If the plane is near PI which is where the through zero
      // happens with atan2 adjust the angle so that the through zero
      // is at 0 - 2PI instead.
      bool nearPI = (fabs(plane-M_PI) < 1.0e-2);
      
//      cerr << "Plane " << p << " is " << plane << endl;

      // Ugly but it is necessary to start at the same place each time.
      in = ifield->fdata().begin();
      imesh->begin( inodeItr );

      // Skip all of the points between the beginning and the first
      // point found before (CCW) or after (CW) the zero plane.
      while( *inodeItr != inodeGlobalStart[c] ) {
	++inodeItr;
	++in;
      }

      imesh->get_center(lastPt, *inodeItr);
      lastAng = atan2( lastPt.y(), lastPt.x() );
      // If the plane is near PI which where through zero happens with
      // atan2 adjust the angle so that the through zero is at 0 - 2PI
      // instead.
      if( nearPI && lastAng < 0 ) lastAng += 2 * M_PI;

      ++inodeItr;
      ++in;
      
      ostringstream str;
      str << "Streamline " << c+1 << " Node Index";
      if( !ifield->get_property( str.str(), inodeNext ) )
	inodeNext = *inodeEnd;

//      cerr << "Starting new streamline binning " << c << endl;;

      while (*inodeItr != inodeNext) {
	
	imesh->get_center(currPt, *inodeItr);
	currAng = atan2( currPt.y(), currPt.x() );
	// If the plane is near PI which where through zero happens with
	// atan2 adjust the angle so that the through zero is at 0 - 2PI
	// instead.
	if( nearPI && currAng < 0 ) currAng += 2 * M_PI;

	// First look at only points that are in the correct plane.
	if( ( CCWstreamline && lastAng <= plane && plane <= currAng) ||
	    (!CCWstreamline && currAng <= plane && plane <= lastAng) ) {
	  double t;

	  if( fabs(currAng-lastAng) > 1.0e-12 )
	    t = (plane-lastAng) / (currAng-lastAng);
	  else
	    t = 0;
	  
	  Point point = interpert( lastPt, currPt, t );

	  if( bins[p].size() <= (unsigned int) bin ) {
	    vector< pair<Point, double> > sub_bin;
	  
	    sub_bin.push_back( pair<Point, double>(point, *in) );

	    bins[p].push_back( sub_bin );

	  } else {
	    bins[p][bin].push_back( pair<Point, double>(point, *in) );
	  }

	  bin = (bin + 1) % windings[c];
	}

	lastPt  = currPt;
	lastAng = currAng;
    
	++inodeItr;
	++in;
      }
    }
    
    unsigned int nnodes = (int) 1e8;
    bool VALID = true;

    // Validation check
    for( unsigned int p=0; p<planes.size(); p++ ) {
      for( unsigned int i=0; i<windings[c]; i++ ) {
	if( nnodes > bins[p][i].size() )
	  nnodes = bins[p][i].size();

	if( bins[p][i].size() < 1 ) {
	  cerr << "INVALID - Plane " << p << " bin  " << i
	       << " bin size=" << bins[p][i].size() << endl;

	  VALID = false;
	}
      }
    }

    // Get the rest of the info only from the phi = zero plane.
    unsigned int p;

    if( CCWstreamline )
      p = 0;
    else
      p = planes.size()-1;

    // Get the centroid of each group and all groups.
    Vector centroid(0,0,0);
    vector< Vector > localCentroids;
    vector< Vector > localSeparatrices[2];

    localCentroids.resize(windings[c]);
    localSeparatrices[0].resize(windings[c]);
    localSeparatrices[1].resize(windings[c]);

    unsigned int cc = 0;

    for( unsigned int i=0; i<windings[c]; i++ ) {
      Vector localCentroid(0,0,0);

      for( unsigned int j=0; j<bins[p][i].size(); j++ ) 
	localCentroid += (Vector) bins[p][i][j].first;

      if( bins[p][i].size() ) {
	localCentroids[i] = localCentroid / (double) bins[p][i].size();

	centroid += localCentroids[i];

	cc++;

      } else {
	localCentroids[i] = Vector(0,0,0);
      }
    }

    unsigned int islands2 = 0;
    unsigned int islands = 0;
    unsigned int twist = 0;
    bool completeIslands = true;

    if( cc == windings[c]) {
      centroid /= cc;

      double angle = 0;
      
      // Determine the number of twists by totalling up the angles
      // between each of the group centroids and dividing by 2 PI.
      for( unsigned int i=1; i<=windings[c]; i++ ) {
	
	Vector v0 = localCentroids[i-1          ] - centroid;
	Vector v1 = localCentroids[i%windings[c]] - centroid;
	
	angle += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );
      }

      // Because the total will not be an exact integer add a delta so
      // the rounded value will be correct.
      twist = (unsigned int) ((angle +  M_PI/windings[c]) / (2.0*M_PI));

      // Determine if islands exists. If an island exists there will
      // be both clockwise and counterclockwise sections when compared
      // to the main centroid.
      for( unsigned int i=0; i<windings[c]; i++ ) {
	bool lastCCW = 0;
	
	unsigned int tmpNnodes = 0;
	unsigned int turns = 0;

	unsigned int startIndex,  middleIndex,  endIndex;

	if( bins[p][i].size() >= 3 ) {
	  for( unsigned int j=1; j<bins[p][i].size(); j++ ) {
	    bool CCW =
	      ccw(Vector( (Vector) bins[p][i][j-1].first - centroid ), 
		  Vector( (Vector) bins[p][i][j  ].first - centroid ));
	    
	    if( j > 1 && CCW != lastCCW ) {
	      if( turns == 0 )
		islands++;
	      
	      turns++;

	      if( turns == 1 )
		startIndex = j - 1;
	      else if( turns == 2 )
		middleIndex = j - 1;
	      else if( turns == 3 )
		endIndex = j - 1;
	    }

	    // Count the number of nodes between the first and third
	    // turn. This count gives a complete winding of the island.
	    if( turns > 0 )
	      tmpNnodes++;
	    
	    if( turns == 3 )
	      break;

	    lastCCW = CCW;
	  }

//  	  cerr << i << " turns " << turns
//  	       << "  " << tmpNnodes
//  	       << "  " << nnodes << endl;

	  if( overlaps && turns == 3 && nnodes > tmpNnodes )
	    nnodes = tmpNnodes;

	  if( turns >= 2 ) {
	    localSeparatrices[0][i] = (Vector) bins[p][i][startIndex].first;
	    localSeparatrices[1][i] = (Vector) bins[p][i][middleIndex].first;
	  }

	  if( turns == 3 ) {
	    unsigned int index0 = (middleIndex - startIndex ) / 2;
	    unsigned int index1 = (   endIndex - middleIndex) / 2;

 	    cerr << "Indexes " <<  startIndex << "  "
 		 << middleIndex << "  " << endIndex << endl;

 	    cerr << "Indexes mid " <<  index0 << "  " << index1 << endl;

	    localCentroids[i] =
	      ( (Vector) bins[p][i][ startIndex + index0].first + 
		(Vector) bins[p][i][middleIndex - index0].first + 
		(Vector) bins[p][i][middleIndex + index1].first + 
		(Vector) bins[p][i][   endIndex - index1].first ) / 4.0;
	  } else
	    completeIslands = false;
	}
      }

    } else {
      cerr << "Can not determine the numbers of twists or islands" << endl;
    }

    if( overlaps )
      removeOverlaps( bins[p], nnodes,
		      windings[c], twist, CCWstreamline, centroid );

    // If the twists is a factorial of the winding then rebin the points.
    if( !override && windings[c] && twist != 1 &&
	factorial( windings[c], twist ) ) {

      unsigned int fact;
      while( fact = factorial( windings[c], twist ) ) {
	windings[c] /= fact;
	twist /= fact;
      }

      c--;

    } else {

      cerr << "Surface " << c << " is a "
	   << windings[c] << ":" << twist << " surface ("
	   << (double) windings[c] / (double) twist << ") ";

      if( islands ) 
	cerr << "that contains " << islands << " islands"
	     << (completeIslands ? " (Complete)" : "");

      cerr << " and has " << nnodes << " nodes"
	   << endl;


      if( islands && islands != windings[c] ) {
	cerr << "WARNING - The island count does not match the winding count" 
	     << endl;
      }

      twists.push_back( twist );

      // Record the topology.
      pair< unsigned int, unsigned int > topo( windings[c], twist );
      topology.push_back(topo);

      if( !curveField ) {

	vector< unsigned int > dims;
      
	dims.resize(2);
      
	dims[0] = nnodes;
	dims[1] = (planes.size()+1) * windings[c];
      
	((StructQuadSurfMesh *) omesh)->set_dim( dims );

	ofield->resize_fdata();
      }

      double color_value = 0;

      if( color == 1 )
	color_value = c;
      else if( color == 3 )
	color_value = (islands == 0 ? 0 : c);
      else if( color == 6 )
	color_value = windings[c];
      else if( color == 7 )
	color_value = twists[c];
      else if( color == 8 )
	color_value = (double) windings[c] / (double) twists[c];


      if( islands ) {

	if( baseCentroids.size() ) {

	  unsigned int cc = 0;

	  while( cc < windings.size() ) {
	    if( cc <= c && c < cc+windings[cc] )
	      break;
	    else
	      cc += windings[cc];
	  }

	  cerr << "Searching winding " << cc << endl;

	  for( unsigned int i=0; i<windings[c]; i++ ) {
	    
	    unsigned int index;
	    double mindist = 1.0e12;
	    
	    for( unsigned int j=cc; j<cc+windings[cc]; j++ ) {

	      double dist =
		(localCentroids[i] - baseCentroids[j][0]).length();
	      
	      if( mindist > dist ) {
		mindist = dist;
		index = j;
	      }
	    }

	    cerr << cc << "  " << i << "  "
		 << index << " index " << localCentroids[i] << endl;
	    
	    baseCentroids[index].push_back( localCentroids[i] );	      
	  }

	} else {
	  // In order to find the next group after the first find the
	  // mutual primes (Blankinship Algorithm). In this case we only
	  // care about the first one becuase the second is just the
	  // number of windings done to get there.

	  unsigned int skip;

	  for( skip=1; skip<windings[c]; skip++ )
	    if( skip * twist % windings[c] == 1 )
	      break;

// 	  if( !CCWstreamline )
// 	    skip = windings[c] - skip;


	  typename PCFIELD::mesh_type::Node::index_type n;

	  for( unsigned int i=0; i<windings[c]; i++ ) {
	    // Centroids
	    n = opccmesh->add_point((Point) localCentroids[i]);
	    opccfield->resize_fdata();
	    opccfield->set_value( windings[c], n);

	    // Separatrices
	    unsigned int j = (i+skip) % windings[c];

	    unsigned int ii, jj;

	    if( (localSeparatrices[0][i] - localSeparatrices[0][j]).length() <
		(localSeparatrices[0][i] - localSeparatrices[1][j]).length() )
	      jj = 0;
	    else
	      jj = 1;

	    if( (localSeparatrices[0][i] - localSeparatrices[jj][j]).length() <
		(localSeparatrices[1][i] - localSeparatrices[jj][j]).length() )
	      ii = 0;
	    else
	      ii = 1;

	    n = opcsmesh->add_point((Point) ((localSeparatrices[ii][i] +
					      localSeparatrices[jj][j])/2.0));

 	    opcsfield->resize_fdata();
// 	    opcsfield->set_value( (double) windings[c], n);
 	    opcsfield->set_value( (double) n, n);

//  	    n = opcsmesh->add_point((Point) localSeparatrices[0][i]);
// 	    opcsfield->resize_fdata();
// 	    opcsfield->set_value(0, n);

//  	    n = opcsmesh->add_point((Point) localSeparatrices[1][j]);
// 	    opcsfield->resize_fdata();
// 	    opcsfield->set_value(1, n);

 	    cerr << c << "  Separatrices " << i << "  " << j << endl;
	  }
	}
      }

      if( !showIslands || ( showIslands && islands ) ) {

	// Add the points into the return field.
	for( unsigned int p=0; p<planes.size(); p++ ) {
	  for( unsigned int i=0; i<windings[c]; i++ ) {
	    lock.lock();

	    if( curveField )
	      loadCurve( ofield_h, bins[p][i],
			 planes.size(), windings[c], nnodes, p, i,
			 color, color_value );
	    else
	      loadSurface( ofield_h, bins[p][i],
			   planes.size()+1, windings[c], nnodes, p, i,
			   color, color_value );
	
	    lock.unlock();
	  }
	}

	// For a surface add in the first set first so that the surface is
	// complete.
	if( !curveField ) {
	  for( unsigned int i=0; i<windings[c]; i++ ) {
	    lock.lock();

	    unsigned int j = (i-1 + windings[c]) % windings[c];

	    loadSurface( ofield_h, bins[0][i],
			 planes.size()+1, windings[c], nnodes,
			 c, planes.size(), j, color );
	
	    lock.unlock();
	  }
	}
      }
    }
  }


  if( baseCentroids.size() ) {

    typename PCFIELD::mesh_type::Node::index_type n;

    vector< Vector > newCentroid;
    vector< double > newStdDev;

    newCentroid.resize( baseCentroids.size() );
    newStdDev.resize  ( baseCentroids.size() );

    for( unsigned int i=0; i<baseCentroids.size(); i++ ) {

      if( baseCentroids[i].size() ) {

	newCentroid[i] = Vector(0,0,0);
	for( unsigned int j=1; j<baseCentroids[i].size(); j++ )
	  newCentroid[i] += baseCentroids[i][j];
	
	newCentroid[i] /= (double) (baseCentroids[i].size() - 1);

	newStdDev[i] = 0.0;
	for( unsigned int j=1; j<baseCentroids[i].size(); j++ ) {
	  cerr << "length " << i << "  " << j << "  "
	       << (baseCentroids[i][j]-newCentroid[i]).length2() << endl;
	  newStdDev[i] += (baseCentroids[i][j]-newCentroid[i]).length2();
	}

	newStdDev[i] = sqrt( newStdDev[i] /
			     (double) (baseCentroids[i].size() - 2) );
      }
    }

    unsigned int cc = 0;

    double minSD = 1.0e12;
    unsigned int index;

    while( cc < windings.size() ) {
      cerr << "Searching winding " << cc << endl;

      for( unsigned int i=cc; i<cc+windings[cc]; i++ ) {

	cerr << "Std dev. " << i << "  " << newStdDev[i] << endl;

	if( minSD > newStdDev[i] ) {
	  minSD = newStdDev[i];
	  index = i;
	}
      }

      n = opccmesh->add_point((Point) newCentroid[index]);
      
      opccfield->resize_fdata();

      cerr << "New centroid " << newCentroid[index]
	   << " index " << index << endl;

      opccfield->set_value( windings[index], n );

      cc += windings[cc];

      minSD = 1.0e12;
    }

//     for( unsigned int i=0; i<baseCentroids.size(); i++ ) {

//       if( baseCentroids[i].size() ) {
// 	Vector tmpCentroid(0,0,0);
	
// 	for( unsigned int j=1; j<baseCentroids[i].size(); j++ )
// 	  tmpCentroid += baseCentroids[i][j];
	
// 	tmpCentroid /= (double) (baseCentroids[i].size() - 1);
	
// // 	cerr << tmpCentroid << endl;

// 	n = opcmesh->add_point((Point) tmpCentroid);
//       } else {
// 	n = opcmesh->add_point((Point) baseCentroids[i][0]);
//       }

//       opcfield->resize_fdata();

//       opcfield->set_value( i, n );
//     }
  }
}



template< class IFIELD, class OFIELD, class PCFIELD, class TYPE >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD, TYPE>::
removeOverlaps( vector< vector < pair< Point, double > > > &bins,
		unsigned int &nnodes,
		unsigned int winding,
		unsigned int twist,
		bool CCWstreamline,
		Vector centroid )
 {
  unsigned int islands2;

  unsigned int overlap = 1;
  unsigned int flipped = 0;

  while( overlap || flipped > 1 ) {

    overlap = 0;
    flipped = 0;

    vector< pair< double, double > > angleMinMax;
    angleMinMax.resize( winding );

    Vector v0 = Vector( 1.0, 0, 0 );

    islands2 = 0;

    double offset = 0;

    for( unsigned int i=0; i<winding; i++ ) {
	
      angleMinMax[i].first  =  4.0 * M_PI;
      angleMinMax[i].second = -4.0 * M_PI;

      bool islandCheck = true;

      for( unsigned int j=0; j<nnodes; j++ ) {
	Vector v1 = Vector( (Vector) bins[i][j].first - centroid );

	double angle = acos( Dot( v0, v1 ) / v1.length() );

	if( v1.z() < 0 )
	  angle = 2.0 * M_PI - angle;

	if( offset > 0 ) {
	  angle += offset;

	  if( angle > 2.0* M_PI ) 
	    angle -= (2.0* M_PI);
	}

	bool set = false;

	if( angleMinMax[i].first > angle ) {
	  angleMinMax[i].first = angle;
	  set = true;
	}

	if( angleMinMax[i].second < angle ) {
	  angleMinMax[i].second = angle;
	  set = true;
	}

	if( islandCheck && !set) {
	  islandCheck = false;
	  islands2++;
	}
      }

      // Through zero add PI and redo.
      if( offset == 0 &&
	  angleMinMax[i].second - angleMinMax[i].first > M_PI ) {
	offset = M_PI;

	i--;

	if( !islandCheck )
	  islands2--;
	    
      } else if( offset > 0 ) {
	angleMinMax[i].first  -= M_PI;
	angleMinMax[i].second -= M_PI;

	if( angleMinMax[i].first < 0 )
	  angleMinMax[i].first += 2.0*M_PI;

	if( angleMinMax[i].second < 0 )
	  angleMinMax[i].second += 2.0*M_PI;

	offset =  0;
      }
    }

    // In order to find the next group after the first find the
    // mutual primes (Blankinship Algorithm). In this case we only
    // care about the first one becuase the second is just the
    // number of windings done to get there.

    unsigned int skip;

    for( skip=1; skip<winding; skip++ )
      if( skip * twist % winding == 1 )
	break;

    if( !CCWstreamline )
      skip = winding - skip;

    for( unsigned int i=0, j=skip; i<skip*winding; i+=skip, j+=skip ) {
      unsigned int i0 = i % winding;
      unsigned int j0 = j % winding;

//    cerr << i0 << " angle bounds "
//	   << angleMinMax[i0].first << "  "
// 	   << angleMinMax[i0].second;

      if( angleMinMax[i0].first < angleMinMax[j0].second ) {
	// Skip the through zero angles that go through zero.
	double sectionAngle = 2.0 * M_PI / winding;

	if( angleMinMax[i0].first  > 2.0 * sectionAngle ||
	    angleMinMax[j0].second < 2.0 * M_PI - 2.0 * sectionAngle ) {
	  overlap++;

// 	  cerr << " overlaps " << overlap;

// 	  cerr  << "  " << 2.0 * sectionAngle << "  "
// 		<< 2.0 * M_PI - 2.0 * sectionAngle;
	} else {
// 	  cerr << " near zero "
// 	       << 2.0 * sectionAngle << "  "
// 	       << 2.0 * M_PI - 2.0 * sectionAngle;
	}
      }

      if( angleMinMax[i0].first > angleMinMax[i0].second ) {
	flipped++;
// 	cerr << " through zero ";
      }

//    cerr << endl;
    }

    if( overlap || flipped > 1 ) {

//    cerr << " OVERLAPS " << overlap
//      << " FLIPPED " << flipped << endl;

      if( nnodes > 2 )
	nnodes--;
      else
	overlap = flipped = 0;
    }
  }

  return nnodes;
}

} // end namespace Fusion

#endif // StreamlineAnalyzer_h
