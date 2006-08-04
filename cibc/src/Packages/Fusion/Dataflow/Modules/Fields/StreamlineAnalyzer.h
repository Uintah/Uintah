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

#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>

#include <Core/Containers/FData.h>

#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>

#include <Core/Datatypes/GenericField.h>

#include <sstream>
using std::ostringstream;

namespace Fusion {

using namespace std;
using namespace SCIRun;

class StreamlineAnalyzerAlgo : public DynamicAlgoBase
{
public:

  typedef SCIRun::CurveMesh<CrvLinearLgn<Point> > CMesh;
  typedef SCIRun::CrvLinearLgn<double>            CDatBasis;
  typedef SCIRun::GenericField<CMesh, CDatBasis, vector<double> > CSField;
  typedef SCIRun::GenericField<CMesh, CDatBasis, vector<Vector> > CVField;

  typedef SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> > SQSMesh;
  typedef SCIRun::QuadBilinearLgn<double>                     SQSDatBasis;
  typedef SCIRun::GenericField<SQSMesh, SQSDatBasis, FData2d<double,SQSMesh> > SQSSField;
  typedef SCIRun::GenericField<SQSMesh, SQSDatBasis, FData2d<Vector,SQSMesh> > SQSVField;

  typedef SCIRun::PointCloudMesh<ConstantBasis<Point> > PCMesh;
  typedef SCIRun::ConstantBasis<double>                 PCDatBasis;
  typedef SCIRun::GenericField<PCMesh, PCDatBasis, vector<double> > PCField;  

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
		       unsigned int islandCentroids,
		       unsigned int overlaps,
		       unsigned int maxWindings,
		       unsigned int override,
		       unsigned int order,
		       vector< pair< unsigned int,
		       unsigned int > > &topology ) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *mtd,
					    const TypeDescription *btd,
					    const TypeDescription *dtd);
};


template< class IFIELD, class OFIELD, class PCFIELD >
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
		       unsigned int islandCentroids,
		       unsigned int overlaps,
		       unsigned int maxWindings,
		       unsigned int override,
		       unsigned int order,
		       vector< pair< unsigned int,
		       unsigned int > > &topology );

protected:
  Point interpert( Point lastPt, Point currPt, double t ) {

    return Point( Vector( lastPt ) + Vector( currPt - lastPt ) * t );
  }

  int ccw( Vector v0, Vector v1 ) {
    
    if( v0.x() * v1.z() > v0.z() * v1.x() ) return 1;         //  CCW
    
    if( v0.x() * v1.z() < v0.z() * v1.x() ) return -1;        //  CW
    
    if( v0.x() * v1.x() < 0 || v0.z()*v1.z() < 0 ) return -1; //  CW
    
    if( v0.x()*v0.x()+v0.z()*v0.z() >=
	v1.x()*v1.x()+v1.z()*v1.z() ) return 0;               //  ON LINE
    
    return 1;                                                 //  CCW
  }

  int intersect( Point l0_p0, Point l0_p1,
		 Point l1_p0, Point l1_p1 )
  {
    //  See if the lines intersect.    
    if( ( ccw( Vector(l0_p1-l0_p0), Vector(l1_p0-l0_p0)) *
	  ccw( Vector(l0_p1-l0_p0), Vector(l1_p1-l0_p0) ) <= 0 ) &&
	( ccw( Vector(l1_p1-l1_p0), Vector(l0_p0-l1_p0)) *
	  ccw( Vector(l1_p1-l1_p0), Vector(l0_p1-l1_p0) ) <= 0 ) ) {
	
      //  See if there is not a shared point.
      if( l0_p0 != l1_p0 && l0_p0 != l1_p1 &&
	  l0_p1 != l1_p0 && l0_p1 != l1_p1 )
	return 1;
	
      //  See if there is a shared point.
      else if( l0_p0 == l1_p0 || l0_p0 == l1_p1 ||
	  l0_p1 == l1_p0 || l0_p1 == l1_p1 )
	return 2;
	
      //  There must be a point that is on the other line.
      else
	return 3;
    }
    
    //  Lines do not intersect.
    return 0;
  }


  unsigned int factorial( unsigned int n0, unsigned int n1 ) {

    unsigned int min = n0 < n1 ? n0 : n1;

    for( unsigned int i=min; i>1; i-- )
      if( n0 % i == 0 && n1 % i == 0 )
	return i;

    return 0;
  }

  Point circle(Point &pt1, Point &pt2, Point &pt3)
  {
    if (!IsPerpendicular(pt1, pt2, pt3) )
      return CalcCircle(pt1, pt2, pt3);	
    else if (!IsPerpendicular(pt1, pt3, pt2) )
      return CalcCircle(pt1, pt3, pt2);	
    else if (!IsPerpendicular(pt2, pt1, pt3) )
      return CalcCircle(pt2, pt1, pt3);	
    else if (!IsPerpendicular(pt2, pt3, pt1) )
      return CalcCircle(pt2, pt3, pt1);	
    else if (!IsPerpendicular(pt3, pt2, pt1) )
      return CalcCircle(pt3, pt2, pt1);	
    else if (!IsPerpendicular(pt3, pt1, pt2) )
      return CalcCircle(pt3, pt1, pt2);	
    else
      return Point(-1,-1,-1);
  }

  // Check the given point are perpendicular to x or y axis 
  bool IsPerpendicular(Point &pt1, Point &pt2, Point &pt3)
  {
    double d21z = pt2.z() - pt1.z();
    double d21x = pt2.x() - pt1.x();
    double d32z = pt3.z() - pt2.z();
    double d32x = pt3.x() - pt2.x();
	
    // checking whether the line of the two pts are vertical
    if (fabs(d21x) <= FLT_MIN && fabs(d32z) <= FLT_MIN)
      return false;
    
    else if (fabs(d21z) < FLT_MIN ||
	     fabs(d32z) < FLT_MIN ||
	     fabs(d21x) < FLT_MIN ||
	     fabs(d32x)<= FLT_MIN )
      return true;

    else
      return false;
  }

  Point CalcCircle(Point &pt1, Point &pt2, Point &pt3)
  {
    Point center;
      
    double d21z = pt2.z() - pt1.z();
    double d21x = pt2.x() - pt1.x();
    double d32z = pt3.z() - pt2.z();
    double d32x = pt3.x() - pt2.x();
	
    if (fabs(d21x) < FLT_MIN && fabs(d32z ) < FLT_MIN ) {

      center.x( 0.5*(pt2.x() + pt3.x()) );
      center.y( pt1.y() );
      center.z( 0.5*(pt1.z() + pt2.z()) );

      return center;
    }
	
    // IsPerpendicular() assure that xDelta(s) are not zero
    double aSlope = d21z / d21x;
    double bSlope = d32z / d32x;

    // checking whether the given points are colinear. 	
    if (fabs(aSlope-bSlope) > FLT_MIN) {
    
      // calc center
      center.x( (aSlope*bSlope*(pt1.z() - pt3.z()) +
		 bSlope*(pt1.x() + pt2.x()) -
		 aSlope*(pt2.x() + pt3.x()) ) / (2.0* (bSlope-aSlope) ) );

      center.y( pt1.y() );

      center.z( -(center.x() - (pt1.x()+pt2.x())/2.0) / aSlope +
		(pt1.z()+pt2.z())/2.0 );
    }

    return center;
  }

  bool
  IntersectCheck( vector< Point >& points, unsigned int nbins );

  pair< pair< double, double >, pair< double, double > >
  FingerCheck( vector< Point >& points, unsigned int nbins );

  bool
  basicChecks( vector< Point >& points,
	       Vector centroid,
	       unsigned int &winding,
	       unsigned int &twist,
	       unsigned int &skip,
	       unsigned int &island,
	       float &avenode,
	       bool &groupCCW,
	       unsigned int &windingNextBest );

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
  islandCheck( vector< pair< Point, double > > &bins,
	       Vector centroidGlobal,
	       unsigned int &startIndex,
	       unsigned int &middleIndex,
	       unsigned int &stopIndex );

  unsigned int
  surfaceCheck( vector< vector< pair< Point, double > > > &bins,
		unsigned int winding,
		unsigned int skip,
		Vector centroidGlobal,
		unsigned int &nnodes );

  unsigned int
  surfaceCheck( vector< vector< pair< Point, double > > > &bins,
		unsigned int i,
		unsigned int j,
		unsigned int nnodes );

  unsigned int
  removeOverlap( vector< vector < pair< Point, double > > > &bins,
		 unsigned int &nnodes,
		 unsigned int winding,
		 unsigned int twist,
		 unsigned int skip,
		 unsigned int island );

  unsigned int
  smoothCurve( vector< vector < pair< Point, double > > > &bins,
	       unsigned int &nnodes,
	       unsigned int winding,
	       unsigned int twist,
	       unsigned int skip,
	       unsigned int island );

  unsigned int
  mergeOverlap( vector< vector < pair< Point, double > > > &bins,
		unsigned int &nnodes,
		unsigned int winding,
		unsigned int twist,
		unsigned int skip,
		unsigned int island );

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


template< class IFIELD, class OFIELD, class PCFIELD >
class StreamlineAnalyzerAlgoTScalar :
  public StreamlineAnalyzerAlgoT< IFIELD, OFIELD, PCFIELD >
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

template< class IFIELD, class OFIELD, class PCFIELD >
class StreamlineAnalyzerAlgoTVector :
  public StreamlineAnalyzerAlgoT< IFIELD, OFIELD, PCFIELD >
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


template< class IFIELD, class OFIELD, class PCFIELD >
void
StreamlineAnalyzerAlgoTScalar<IFIELD, OFIELD, PCFIELD>::
loadCurve( FieldHandle &field_h,
	   vector < pair< Point, double > > &nodes,
	   unsigned int nplanes,
	   unsigned int nbins,
	   unsigned int nnodes,
	   unsigned int plane,
	   unsigned int bin,
	   unsigned int color,
	   double color_value ) {

  StreamlineAnalyzerAlgo::CSField *ofield =
    (StreamlineAnalyzerAlgo::CSField *) field_h.get_rep();
  typename StreamlineAnalyzerAlgo::CSField::mesh_handle_type omesh =
    ofield->get_typed_mesh();
  typename StreamlineAnalyzerAlgo::CSField::mesh_type::Node::index_type n1, n2;

  n1 = omesh->add_node(nodes[0].first);
  ofield->resize_fdata();
  if( color == 0 )
    ofield->set_value( nodes[0].second, n1);
  else if( color == 1 )
    ofield->set_value( (double) color_value, n1);
  else if( color == 2 )
    ofield->set_value( (double) (0*nbins+bin), n1);
  else if( color == 3 )
    ofield->set_value( (double) plane, n1);
  else if( color == 4 )
    ofield->set_value( (double) bin, n1);
  else if( color == 5 )
    ofield->set_value( (double) 0, n1);
  else
    ofield->set_value( (double) color_value, n1);
  
  for( unsigned int i=1; i<nnodes; i++ ) {
    n2 = omesh->add_node(nodes[i].first);
    ofield->resize_fdata();

    if( color == 0 )
      ofield->set_value(nodes[i].second, n2);
    else if( color == 1 )
      ofield->set_value( (double) color_value, n2);
    else if( color == 2 )
      ofield->set_value( (double) (i*nbins+bin), n2);
    else if( color == 3 )
      ofield->set_value( (double) plane, n2);
    else if( color == 4 )
      ofield->set_value( (double) bin, n2);
    else if( color == 5 )
      ofield->set_value( (double) i, n2);
    else
      ofield->set_value( (double) color_value, n2);
    
    omesh->add_edge(n1, n2);
	      
    n1 = n2;
  }
}


template< class IFIELD, class OFIELD, class PCFIELD >
void
StreamlineAnalyzerAlgoTScalar<IFIELD, OFIELD, PCFIELD>::
loadSurface( FieldHandle &field_h,
	     vector < pair< Point, double > > &nodes,
	     unsigned int nplanes,
	     unsigned int nbins,
	     unsigned int nnodes,
	     unsigned int plane,
	     unsigned int bin,
	     unsigned int color,
	     double color_value ) {
  
  StreamlineAnalyzerAlgo::SQSSField *ofield =
    (StreamlineAnalyzerAlgo::SQSSField *) field_h.get_rep();
  typename StreamlineAnalyzerAlgo::SQSSField::mesh_handle_type omesh =
    ofield->get_typed_mesh();
  typename StreamlineAnalyzerAlgo::SQSSField::mesh_type::Node::index_type n1;

  n1.mesh_ = omesh.get_rep();

  n1.j_ = nplanes * bin + plane;

  for( unsigned int i=0; i<nnodes; i++ ) {

    n1.i_ = i;

    omesh->set_point(nodes[i].first, n1);

    if( color == 0 )
      ofield->set_value( nodes[i].second, n1);
    else if( color == 1 )
      ofield->set_value( (double) color_value, n1);
    else if( color == 2 )
      ofield->set_value( (double) (i*nbins+bin), n1);
    else if( color == 3 )
      ofield->set_value( (double) plane, n1);
    else if( color == 4 )
      ofield->set_value( (double) bin, n1);
    else if( color == 5 )
      ofield->set_value( (double) i, n1);
    else
      ofield->set_value( (double) color_value, n1);
  }
}

template< class IFIELD, class OFIELD, class PCFIELD >
void
StreamlineAnalyzerAlgoTVector<IFIELD, OFIELD, PCFIELD>::
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

  StreamlineAnalyzerAlgo::CSField *ofield =
    (StreamlineAnalyzerAlgo::CSField *) field_h.get_rep();
  typename StreamlineAnalyzerAlgo::CSField::mesh_handle_type omesh =
    ofield->get_typed_mesh();
  typename StreamlineAnalyzerAlgo::CSField::mesh_type::Node::index_type n1, n2;

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
    tangent *= plane;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (Vector) tangent, n1);
  
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
      tangent *= plane;
    else if( color == 4 )
      tangent *= bin;
    else if( color == 5 )
      tangent *= i;
    else
      tangent *= color_value;
 
    ofield->set_value( (Vector) tangent, n2);
    
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
    tangent *= plane;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (Vector) tangent, n2);
    
  omesh->add_edge(n1, n2);	      
}


template< class IFIELD, class OFIELD, class PCFIELD >
void
StreamlineAnalyzerAlgoTVector<IFIELD, OFIELD, PCFIELD>::
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

  StreamlineAnalyzerAlgo::SQSVField *ofield =
    (StreamlineAnalyzerAlgo::SQSVField *) field_h.get_rep();
  typename StreamlineAnalyzerAlgo::SQSVField::mesh_handle_type omesh =
    ofield->get_typed_mesh();
  typename StreamlineAnalyzerAlgo::SQSVField::mesh_type::Node::index_type n1;

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
    tangent *= plane;
  else if( color == 4 )
    tangent *= bin;
  else if( color == 5 )
    tangent *= i;
  else
    tangent *= color_value;
 
  ofield->set_value( (Vector) tangent, n1);

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
      tangent *= plane;
    else if( color == 4 )
      tangent *= bin;
    else if( color == 5 )
      tangent *= i;
    else
      tangent *= color_value;
 
    ofield->set_value( (Vector) tangent, n1);
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
 
  ofield->set_value( (Vector) tangent, n1);
}


template< class IFIELD, class OFIELD, class PCFIELD >
double
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
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

  bool twistCCW = (ccw(Vector( (Vector) points[0] - centroid ), 
		       Vector( (Vector) points[1] - centroid )) == 1);

  for( unsigned int i=skip, j=1; i<n; i++, j++ ) {
    Vector v0( points[i-1] - centroid );
    Vector v1( points[i  ] - centroid );
  
    angle += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );

    safetyFactorAve += ((2.0*M_PI) / (angle / (double) j));

    double tmp = angle / (2.0*M_PI);

    if( i <= maxWindings ) {
        cerr << i << "  SafetyFactor "
 	    << safetyFactorAve << "  "
 	    << tmp << endl;

      bool groupCCW = (ccw(Vector( (Vector) points[0] - centroid ),
			   Vector( (Vector) points[i] - centroid )) == 1);

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


template< class IFIELD, class OFIELD, class PCFIELD >
double
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
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

template< class IFIELD, class OFIELD, class PCFIELD >
bool
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
IntersectCheck( vector< Point >& points, unsigned int nbins ) {

  for( unsigned int i=0, j=nbins; i<nbins && j<points.size(); i++, j++ ) {
    Point l0_p0 = points[i];
    Point l0_p1 = points[j];

    for( unsigned int k=i+1, l=j+1; k<nbins && l<points.size(); k++, l++ ) {
      Point l1_p0 = points[k];
      Point l1_p1 = points[l];

//      cerr << nbins
// 	   << "   " << i << "  " << j << "  " << k << "  " << l << endl;

      if( j != k && intersect( l0_p0, l0_p1, l1_p0, l1_p1 ) == 1)
	return false;
    }
  }

  return true;
}


template< class IFIELD, class OFIELD, class PCFIELD >
double
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
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

template< class IFIELD, class OFIELD, class PCFIELD >
bool
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
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

  // If any section is greater than 50% of the bounding box diagonal
  // discard it.
  double len2 = (maxPt-minPt).length2();

  for( unsigned int i=0; i<n-nbins; i++ ) {
    if( ((Vector) points[i] - (Vector) points[i+nbins]).length2() >
	0.50 * len2 ) {

      return false;
    }
  }

  return true;
}


template< class IFIELD, class OFIELD, class PCFIELD >
pair< pair< double, double >, pair< double, double > >
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
FingerCheck( vector< Point >& points, unsigned int nbins ) {

  double dotProdAveLength = 0;
  double dotProdAveAngle  = 0;

  double dotProdStdDevLength = 0;
  double dotProdStdDevAngle  = 0;

  unsigned int n = points.size();

  int cc = 0;

  bool violation = false;

  // Check a point and its neighbor from one group with a point from
  // all of the other groups.
  for( unsigned int i=0; i<n-nbins; i++ ) {
    for( unsigned int j=i+1; j<i+nbins; j++ ) {

      Vector v0 = Vector( points[i      ] - points[j] );
      Vector v1 = Vector( points[i+nbins] - points[j] );
      
      v0.safe_normalize();
      v1.safe_normalize();

      double dotProd = Dot( v0, v1 );
      
      cerr << i << "  " << i+nbins << "  " << j
	  << " Len Dot Prod " << sqrt(dotProd) << "  "
 	  << " Ang Dot Prod " << dotProd << endl;

      if( dotProd < 0.0 ) {
	violation = true;

// 	return pair< pair< double, double >, pair< double, double > >
// 	  ( pair< double, double >(-1,1.0e12),
// 	    pair< double, double >(-1,1.0e12) );
      } else {
      
	dotProdAveLength += sqrt(dotProd / v0.length2());
      
	dotProd /= (v0.length() * v1.length());
	dotProdAveAngle += dotProd;
	
	cc++;
      }
    }
  }

  if( violation )
    return pair< pair< double, double >, pair< double, double > >
      ( pair< double, double >(-1,1.0e12),
	pair< double, double >(-1,1.0e12) );

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


template< class IFIELD, class OFIELD, class PCFIELD >
bool
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
basicChecks( vector< Point >& points,
	     Vector centroid,
	     unsigned int &winding,
	     unsigned int &twist,
	     unsigned int &skip,
	     unsigned int &island,
	     float &avenode,
	     bool &groupCCW,
	     unsigned int &windingNextBest ) {

  Vector v0, v1;

  // See if it is posible to find the next best based on the
  // overlap which will rule out possible windings.
  windingNextBest = 0;
  avenode = 0;

  // Do a check for islands. Islands will exists if there is a change
  // in direction of the connected points relative to the centroid of
  // all of the points.

  island = 0;

  bool baseCCW;

  for( unsigned int i=0; i<winding; i++ ) {

    // Get the direction based on the first two points.
    v0 = (Vector) points[i        ] - centroid;
    v1 = (Vector) points[i+winding] - centroid;

    bool lastCCW = (ccw( v0, v1 ) == 1);
    v0 = v1;

    // The islands should all go in the same direction initally.
    if( i == 0 ) {
      baseCCW = lastCCW;
    } else if( baseCCW != lastCCW ) {
      island = 0;
      break;
    }

    // Get the direction based on the remaining points.
    for( unsigned int j=i+2*winding; j<points.size(); j+=winding ) {
      v1 = (Vector) points[j] - centroid;

      bool CCW = (ccw( v0, v1 ) == 1);
      v0 = v1;

      // A switch in direction indicates that an island is
      // present. Stop the check as we will worry about it being
      // complete later.
      if( CCW != lastCCW ) {
	island++;
	break;
      }
    }
  }


  // Get the direction of the points for the first group. Assume that
  // the direction is the same for all of the other groups.
  groupCCW = (ccw( (Vector) points[0      ] - centroid, 
		   (Vector) points[winding] - centroid ) == 1);
  
  // Find the next group that is the same direction of the points that
  // are with in each group. The skip is the index offset from one
  // group to the next.
  skip = 1;

  // Necessary only if the winding is greater than 2.
  if( winding > 2 ) {
    double angleMin = 9999;

    v0 = (Vector) points[0] - centroid;

    for( unsigned int i=1; i<winding; i++ ) {
      v1 = (Vector) points[i] - centroid;

      // Get the direction of the next group and the angle between the
      // first group and next group.
      bool CCW = (ccw( v0, v1 ) == 1);

      double angle = acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );

      // If islands exists it does matter what the direction is
      // because should not be any overlap just find the skip.

      // If no islands the groups will overlap so the directions must
      // match. The skip will be found based on the minimum angle
      // between two groups.
      if( (island || (!island && CCW == groupCCW)) &&
	  angleMin > angle ) {
	angleMin = angle;
	skip = i;
      }
    }
  }

  // In order to find the twist find the mutual primes
  // (Blankinship Algorithm). In this case we only care about
  // the first one becuase the second is just the number of
  // windings done to get there.

  for( twist=1; twist<winding; twist++ )
    if( twist * skip % winding == 1 )
      break;

  if( twist == winding ) {
    cerr << endl
	 << "ERROR in finding the - WINDING - TWIST - SKIP"
	 << " winding " << winding
	 << " twist " << twist
	 << " skip " << skip
	 << endl;

    return false;

  } else {

    // Sanity check for the twist/skip calculation
    double angleSum = 0;

    if( winding > 2 ) {

      for( unsigned int i=0; i<winding; i++ ) {

	unsigned int start = i;
	unsigned int stop  = (start +    1) % winding;
	unsigned int next  = (start + skip) % winding;

	Vector v0 = (Vector) points[start] - centroid;

	// Get the angle traveled from the first group to the second
	// for one winding. This is done by summing the angle starting
	// with the first group and going to the geometrically next
	// group stopping when the logically next group is found.

	do {
	  Vector v1 = (Vector) points[next] - centroid;

	  angleSum += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );

// 	  cerr << " winding " << i
// 	       << " start " << start
// 	       << " next " << next
// 	       << " stop " << stop
// 	       << " twist angle "
// 	       << acos( Dot( v0, v1 ) / (v0.length() * v1.length()) ) << endl;

	  start = next;
	  next  = (start + skip) % winding;

	  v0 = v1;
	} while( start != stop );
      }

      // THe total number of twist should be the same. Account small
      // rounding errors by adding 25% of the distanced traveled in
      // one winding.
      unsigned int twistCheck =
	(unsigned int) (angleSum / (2.0 * M_PI) + M_PI/2.0/winding);

      if( twistCheck != twist ) {
	cerr << endl
	     << "WARNING - TWIST MISMATCH "
	     << " angle sum " << (angleSum / (2.0 * M_PI))
	     << " winding " << winding
	     << " twistCheck " << twistCheck
	     << " twist " << twist
	     << " skip " << skip
	     << endl;
      }
    }
  }

  // Island sanity check make sure no island overlaps another island.
  if( island ) {

    for( unsigned int i=0; i<winding && i+winding<points.size(); i++ ) {
      unsigned int previous = (i - skip + winding) % winding;

      // See if a point overlaps the first section.
      for( unsigned int j=previous; j<points.size(); j+=winding ) {

	v0 = (Vector) points[i        ] - (Vector) points[j];
	v1 = (Vector) points[i+winding] - (Vector) points[j];

	if( Dot( v0, v1 ) < 0 ) {

	  island = false;

	  v0.safe_normalize();
	  v1.safe_normalize();

	  cerr << "FAILED ISLAND SANITY CHECK "
	       << i << "  " << i + winding << "  " << j << "  "
	       << Dot( v0, v1 ) << endl;

	  break;
	}
      }

      // See if the first point overlaps another section.
      if( island ) {
	for( unsigned int j=previous; j<points.size()-winding; j+=winding ) {

	  v0 = (Vector) points[i] - (Vector) points[j];
	  v1 = (Vector) points[i] - (Vector) points[j+winding];

	  if( Dot( v0, v1 ) < 0 ) {
	    island = false;
	  v0.safe_normalize();
	  v1.safe_normalize();

	  cerr << "FAILED ISLAND SANITY CHECK "
	       << i << "  " << j << "  " << j + winding << "  "
	       << Dot( v0, v1 ) << endl;


	    break;
	  }
	}
      }

      unsigned int next = (i + skip) % winding;

      // See if a point overlaps the first section.
      if( island ) {
	for( unsigned int j=next; j<points.size(); j+=winding ) {

	  v0 = (Vector) points[i        ] - (Vector) points[j];
	  v1 = (Vector) points[i+winding] - (Vector) points[j];
	  
	  if( Dot( v0, v1 ) < 0 ) {
	    island = false;
	  v0.safe_normalize();
	  v1.safe_normalize();

	  cerr << "FAILED ISLAND SANITY CHECK "
	       << i << "  " << i + winding << "  " << j << "  "
	       << Dot( v0, v1 ) << endl;

	    break;
	  }
	}
      }

      // See if the first point overlaps another section.
      if( island ) {
	for( unsigned int j=next; j<points.size()-winding; j+=winding ) {

	  v0 = (Vector) points[i] - (Vector) points[j];
	  v1 = (Vector) points[i] - (Vector) points[j+winding];

	  if( Dot( v0, v1 ) < 0 ) {
	    island = false;
	  v0.safe_normalize();
	  v1.safe_normalize();

	  cerr << "FAILED ISLAND SANITY CHECK "
	       << i << "  " << j << "  " << j + winding << "  "
	       << Dot( v0, v1 ) << endl;

	    break;
	  }
	}
      }
    }

    if( island ) {
      // In this case there is no overlap so the number of nodes is
      // just use the average.
      avenode = (float) points.size() / (float) winding;
    }
  }

  else {
    unsigned int previous = winding - skip;

    // See if a point overlaps the first section.
    for( unsigned int j=previous; j<points.size(); j+=winding ) {

      v0 = (Vector) points[      0] - (Vector) points[j];
      v1 = (Vector) points[winding] - (Vector) points[j];

      if( Dot( v0, v1 ) < 0 ) {
	windingNextBest = j - winding;
	break;
      }
    }

    // See if the first point overlaps another section.
    if( windingNextBest == 0 ) {
      for( unsigned int j=previous; j<points.size()-winding; j+=winding ) {

	v0 = (Vector) points[0] - (Vector) points[j];
	v1 = (Vector) points[0] - (Vector) points[j+winding];

	if( Dot( v0, v1 ) < 0 ) {
	  windingNextBest = j;
	  break;
	}
      }
    }

    // If next best winding is set then there is an overlap. As such,
    // check to see if the number of points that would be in each
    // group is the same.
    if( windingNextBest ) {

      unsigned int nnodes = windingNextBest / winding + 1;

      // Search each group
      for( unsigned int i=0; i<winding; i++ ) {

	// and check for overlap with the next group.
	unsigned int j = (i + skip) % winding;

	unsigned int nodes = nnodes;

	while( i + nodes * winding < points.size() ) {
	  // Check to see if the first overlapping point is really a
	  // fill-in point. This happens because the spacing between
	  // winding groups varries between groups.

	  unsigned int k = i + nodes * winding;

	  v0 = (Vector) points[j          ] - (Vector) points[k];
	  v1 = (Vector) points[k - winding] - (Vector) points[k];

	  if( Dot( v0, v1 ) < 0.0 )
	    nodes++;
	  else
	    break;
	}

	avenode += nodes;
      }

      avenode /= winding;

    } else {

      // In this case there is no overlap so the number of nodes can
      // not be determined correctly so just use the average.
      avenode = (float) points.size() / (float) winding;
    }
  }

  return true;
}


template< class IFIELD, class OFIELD, class PCFIELD >
void
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
execute(FieldHandle& ifield_h,
	FieldHandle& ofield_h,
	FieldHandle& ipccfield_h,
	FieldHandle& opccfield_h,
	FieldHandle& ipcsfield_h,
	FieldHandle& opcsfield_h,
	vector< double > &planes,
	unsigned int color,
	unsigned int showIslands,
	unsigned int islandCentroids,
	unsigned int overlaps,
	unsigned int maxWindings,
	unsigned int override,
	unsigned int order,
	vector< pair< unsigned int, unsigned int > > &topology)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  typename OFIELD::mesh_type *omesh = scinew typename OFIELD::mesh_type();
  OFIELD *ofield = scinew OFIELD(omesh);

  ofield_h = FieldHandle(ofield);

  bool curveField =
    (ofield->get_type_description(Field::MESH_TD_E)->get_name().find("Curve") !=
     string::npos );


  // Point Cloud Field of centroids.
  vector< vector < Vector > > baseCentroids;
  vector < unsigned int > baseCentroidsWinding;

  if( ipccfield_h.get_rep() ) {
    PCFIELD *ipcfield = (PCFIELD *) ipccfield_h.get_rep();
    typename PCFIELD::mesh_handle_type ipcmesh = ipcfield->get_typed_mesh();

    typename PCFIELD::fdata_type::iterator dataItr = ipcfield->fdata().begin();
    typename PCFIELD::mesh_type::Node::iterator inodeItr, inodeEnd;

    ipcmesh->begin( inodeItr );
    ipcmesh->end( inodeEnd );

    Point pt;

    while (inodeItr != inodeEnd) {
      ipcmesh->get_center(pt, *inodeItr);

      // Store the base centroid for each point.
      vector < Vector > baseCentroid;
      baseCentroid.push_back( (Vector) pt );
      baseCentroids.push_back( baseCentroid );
      baseCentroidsWinding.push_back( (unsigned int) *dataItr );

//       cerr << "input " << pt << endl;

      ++inodeItr;
      ++dataItr;
    }
  }

  typename PCFIELD::mesh_type *opccmesh = scinew typename PCFIELD::mesh_type();
  PCFIELD *opccfield = scinew PCFIELD(opccmesh);

  opccfield_h = FieldHandle(opccfield);

  // Point Cloud Field of Separatrices.
  vector< vector < Vector > > baseSeparatrices;
  vector < unsigned int > baseSeparatricesWinding;

  if( ipcsfield_h.get_rep() ) {
    PCFIELD *ipcfield = (PCFIELD *) ipcsfield_h.get_rep();
    typename PCFIELD::mesh_handle_type ipcmesh = ipcfield->get_typed_mesh();

    typename PCFIELD::fdata_type::iterator dataItr = ipcfield->fdata().begin();
    typename PCFIELD::mesh_type::Node::iterator inodeItr, inodeEnd;

    ipcmesh->begin( inodeItr );
    ipcmesh->end( inodeEnd );

    Point pt;

    while (inodeItr != inodeEnd) {
      ipcmesh->get_center(pt, *inodeItr);

      vector < Vector > baseSeparatrix;
      baseSeparatrix.push_back( (Vector) pt );
      baseSeparatrices.push_back( baseSeparatrix );
      baseSeparatricesWinding.push_back( (unsigned int) *dataItr );

//       cerr << "input " << pt << endl;

      ++inodeItr;
      ++dataItr;
    }
  }

  typename PCFIELD::mesh_type *opcsmesh = scinew typename PCFIELD::mesh_type();
  PCFIELD *opcsfield = scinew PCFIELD(opcsmesh);

  opcsfield_h = FieldHandle(opcsfield);


  // Input iterators
  typename IFIELD::fdata_type::iterator dataItr = ifield->fdata().begin();

  typename IFIELD::mesh_type::Node::iterator inodeItr, inodeEnd;
  typename IFIELD::mesh_type::Node::index_type inodeNext;
  vector< typename IFIELD::mesh_type::Node::index_type > inodeGlobalStart;

  imesh->begin( inodeItr );
  imesh->end( inodeEnd );

  if(inodeItr == inodeEnd) return;

  topology.clear();

  vector< unsigned int > windings;
  vector< unsigned int > twists;
  vector< unsigned int > skips;
  vector< unsigned int > islands;
  vector< float > avenodes;

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

  while (inodeItr != inodeEnd) {

    if( *inodeItr == inodeNext ) {

//      cerr << "Starting new streamline winding "
//	   << count << "  " << *inodeItr << endl;

      points.clear();

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
    if( ( CCWstreamline && lastAng < plane && plane <= currAng) ||
	(!CCWstreamline && currAng < plane && plane <= lastAng) ) {
      double t;

      if( fabs(currAng-lastAng) > 1.0e-12 )
	t = (plane-lastAng) / (currAng-lastAng);
      else
	t = 0;

      Point point = interpert( lastPt, currPt, t );

      // Save the point found before the zero plane.
      if( points.size() == 0 ) {
	--inodeItr;

	inodeGlobalStart.push_back( *inodeItr );

	++inodeItr;
      }

      points.push_back( point );
    }

    lastPt  = currPt;
    lastAng = currAng;
    
    ++inodeItr;

    // If about to start the next streamline figure out the number of
    // windings, twist, skips, islands, etc.
    if( *inodeItr == inodeNext ) {

      // Get the centroid for all of the points.
      Vector centroid = Vector(0,0,0);

      for( unsigned int i=0; i<points.size(); i++ )
	centroid += (Vector) points[i];

      centroid /= (double) points.size();

      unsigned int winding, twist, skip, island, windingNextBest;
      float avenode;
      bool groupCCW;

      float bestNodes = 1;

//      if( 0 && (windings.size() == 0 || override) ) {
      if( override ) {
	winding = override;

//	if( windings.size() == 0 && override == 0 )
//	  winding = 4;

	// Do this to get the basic values of the fieldline.
	basicChecks( points, centroid,
		     winding, twist, skip,
		     island, avenode, groupCCW, windingNextBest );

	windings.push_back( winding );
	twists.push_back( twist );
	skips.push_back( skip );
	islands.push_back( island );
	avenodes.push_back( avenode );

      } else {

	unsigned int desiredWinding = 0;

	// Iterative processing of the centroids so the desired
	// winding value is from the base centroid data. 

	if( baseCentroidsWinding.size() )
	  desiredWinding = baseCentroidsWinding[windings.size()];

	pair< unsigned int, double > AngleMin ( 0, 1.0e12);
	pair< unsigned int, double > LengthMin( 0, 1.0e12);
      
	pair< unsigned int, double > AngleMin2 ( 0, 1.0e12);
	pair< unsigned int, double > LengthMin2( 0, 1.0e12);

	vector< unsigned int > windingList;

	// Find the best winding for each test.
	for( winding=2; winding<=maxWindings; winding++ ) {
	  
	  if( points.size() < 2 * winding ) {
	    cerr << "Streamline " << count << " has too few points ("
		 << points.size() << ") to determine the winding accurately"
		 << " past " << winding << " windings " << endl;
	    break;
	  }
	  
	  // If the first two connections of any group crosses another
	  // skip it.
	  if( !IntersectCheck( points, winding ) )
	    continue;

	  cerr << winding << " Passed IntersectCheck\n";

	  // If any finger check results in a negative dot product
	  // skip it.
// 	  pair< pair< double, double >, pair< double, double > > fingers =
// 	    FingerCheck( points, winding );

// 	  if( fingers.first.first   < 0 ||
// 	      fingers.first.second  < 0 ||
// 	      fingers.second.first  < 0 ||
// 	      fingers.second.second < 0 )
// 	    continue;

// 	  cerr << winding << " Passed FingerCheck\n";

	  // Passed all checks so add it to the possibility list.
	  windingList.push_back( winding );
	}

	for( unsigned int i=0; i<windingList.size(); i++ ) {
	  winding = windingList[i];

	  // Do the basic checks
	  if ( basicChecks( points, centroid,
			    winding, twist, skip,
			    island, avenode, groupCCW, windingNextBest ) ) {
	       
	    cerr << "windings " << winding 
		 << "  twists " << twist
		 << "  skip "   << skip
		 << "  groupCCW " << groupCCW
		 << "  islands " << island
		 << "  avenodes " << avenode
		 << "  next best winding " << windingNextBest;

	    if( order == 2 && windingList.size() > 1 && island > 0 ) {

	      if( i > 0 ) 
		cerr << " REPLACED - Islands" << endl;
	      else
		cerr << " START - Islands" << endl;

	      while ( i > 0 ) {
		windingList.erase( windingList.begin() );
		i--;
	      }

	      while ( windingList.size() > 1 ) {
		windingList.erase( windingList.begin()+1 );
	      }

	      continue;

	    } else if( order != 2 && (windingNextBest == 0 || island > 0) ) {
	      cerr << endl;
	      continue;

	    // If very low order and has less than three points in
	    // each group skip it.
	    } else if( order != 2 && 2*winding - skip == windingNextBest &&
		winding < 5 && 
		windingNextBest/winding < 2 ) {
	      
	      vector< unsigned int >::iterator inList =
		find( windingList.begin(),
		      windingList.end(), windingNextBest );
	    
	      // Found the next best in the list so delete the current one.
	      if( inList != windingList.end() ) {
		cerr << " REMOVED - Too few points" << endl;
		
		windingList.erase( windingList.begin()+i );

		i--;
		
		continue;
	      }
	    } 

	    if( order == 0 ) {

	      cerr << " KEEP - low" << endl;

	      // Take the lower ordered surface.
	      unsigned int windingNextBestTmp = windingNextBest;
	      unsigned int islandTmp = island;

	      while( windingNextBestTmp && islandTmp == 0 ) {
		vector< unsigned int >::iterator inList =
		  find( windingList.begin(),
			windingList.end(), windingNextBestTmp );
	    
		if( inList != windingList.end() ) {

		  windingList.erase( inList );

		  unsigned int windingTmp = windingNextBestTmp;
		  unsigned int twistTmp, skipTmp;
		  float avenodeTmp;
		  bool groupCCWTmp;
		  
		  if( basicChecks( points, centroid,
				   windingTmp, twistTmp, skipTmp,
				   islandTmp, avenodeTmp, groupCCWTmp,
				   windingNextBestTmp ) ) {
		    
		    cerr << "windings " << windingTmp
			 << "  twists " << twistTmp
			 << "  skip "   << skipTmp
			 << "  groupCCW " << groupCCWTmp
			 << "  islands " << islandTmp
			 << "  avenodes " << avenodeTmp
			 << "  next best winding " << windingNextBestTmp
			 << " REMOVED - low" << endl;		  
		  }

		} else {
		  windingNextBestTmp = 0;
		}
	      }

	    } else if( order == 1 ) {

	      unsigned int windingTmp = windingNextBest;
	      unsigned int twistTmp, skipTmp, islandTmp, windingNextBestTmp;
	      float avenodeTmp;
	      bool groupCCWTmp;
		  
	      if( basicChecks( points, centroid,
			       windingTmp, twistTmp, skipTmp,
			       islandTmp, avenodeTmp, groupCCWTmp,
			       windingNextBestTmp ) ) {
		    
		if( 2*windingTmp - skipTmp != windingNextBestTmp ) {
		  // Basic philosophy - take the higher ordered surface
		  // winding which will give a smoother curve.	      
		  vector< unsigned int >::iterator inList =
		    find( windingList.begin(),
			  windingList.end(), windingNextBest );
	    
		  if( inList != windingList.end() ) {
		    cerr << " REMOVED - high" << endl;
		
		    windingList.erase( windingList.begin()+i );

		    i--;
		
		    continue;

		  } else {
		    cerr << " KEEP - high" << endl;
		  }
		}
	      }
	    } else if( order == 2 ) {

	      // Keep the winding where the number of nodes for each
	      // group is closest to being the same.
// 	      float diff =
// 		(avenode - floor(avenode)) < (ceil(avenode)-avenode) ?
// 		(avenode - floor(avenode)) : (ceil(avenode)-avenode);

	      // For the first keep is as the base winding.
	      if( i == 0 ) {
		bestNodes = avenode;

		cerr << " START " << endl;

		
	      } else if( windingList[0] <= 3 && bestNodes < 8 ) {
		bestNodes = avenode;
		windingList.erase( windingList.begin() );
		
		i--;

		cerr << " REPLACED " << endl;

		// The current winding is the best so erase the first.
	      } else if( bestNodes < avenode ) {
		bestNodes = avenode;
		windingList.erase( windingList.begin() );
		
		i--;

		cerr << " REPLACED " << endl;

		continue;

		// The first winding is the best so erase the current.
	      } else {
		windingList.erase( windingList.begin()+i );
		
		i--;
		
		cerr << " REMOVED " << endl;

		continue;
	      }
	    }
	  } else {
	    windingList.erase( windingList.begin()+i );

	    i--;
		
	    cerr << "windings " << winding
		 << " FAILED Basic checks" << endl;

	    continue;
	  }

	  // Do the finger check again to get the best.
// 	  pair< pair< double, double >, pair< double, double > > fingers =
// 	    FingerCheck( points, winding );

// 	  if( fingers.first.first > 0 &&
// 	      fingers.first.second > 0 &&
// 	      fingers.second.first > 0 &&
// 	      fingers.second.second > 0 ) {
//	  {
//   	    cerr << winding << " fingers "
//  		 << fingers.first.first << "  "
//  		 << fingers.first.second << "  "
//  		 << fingers.second.first << "  "
//  		 << fingers.second.second << "  "
// 		 << endl;

// 	    if( LengthMin.second > fingers.first.second ){
// 	      LengthMin2 = LengthMin;
	      
// 	      LengthMin.first  = winding;
// 	      LengthMin.second = fingers.first.second;

// 	    } else if( LengthMin2.second > fingers.first.second ){

// 	      LengthMin2.first  = winding;
// 	      LengthMin2.second = fingers.first.second;
// 	    }
	    
// 	    if( AngleMin.second > fingers.second.second ){
// 	      AngleMin2 = AngleMin;
	      
// 	      AngleMin.first  = winding;
// 	      AngleMin.second = fingers.second.second;

// 	    } else if( AngleMin2.second > fingers.second.second ){
// 	      AngleMin2.first  = winding;
// 	      AngleMin2.second = fingers.second.second;
// 	    }
// 	  }
 	}

// 	cerr << endl;

	if( windingList.size() == 0 ) {
	  windings.push_back( 0 );
	  twists.push_back( 0 );
	  skips.push_back( 0 );
	  islands.push_back( 0 );

	} else if( windingList.size() == 1 ) {

	  windings.push_back( windingList[0] );

	// If two possible windings take based on the order preference.
	} else if( windingList.size() == 2 ) {

	  windings.push_back( windingList[order] );

	  // Top two windings are the same ...
	  // take based on the order preference.
	} else if( LengthMin.first  == AngleMin.first &&
		   LengthMin2.first == AngleMin2.first )  {
	  
	  if( (order == 0 && LengthMin.first < LengthMin2.first) ||
	      (order == 1 && LengthMin.first > LengthMin2.first) )
	    windings.push_back( LengthMin.first );
	  else
	    windings.push_back( LengthMin2.first );

	  // Top two windings are the same but mixed ...
	  // take based on the order preference.
	} else if( LengthMin.first == AngleMin2.first &&
		   LengthMin2.first == AngleMin.first ) {

	  if( (order == 0 && LengthMin.first < LengthMin2.first) ||
	      (order == 1 && LengthMin.first > LengthMin2.first) )
	    windings.push_back( LengthMin.first );
	  else
	    windings.push_back( AngleMin.first );

	} else if( LengthMin.first == AngleMin.first ) {

	  windings.push_back( LengthMin.first );

	} else {

	  if( (order == 0 && LengthMin.first < LengthMin2.first) ||
	      (order == 1 && LengthMin.first > LengthMin2.first) )
	    windings.push_back( LengthMin.first );
	  else
	    windings.push_back( AngleMin.first );
	}

	if( windings[count] != 0 ) {

	  winding = windings[count];

	  // Do this again to get the values back.
	  basicChecks( points, centroid,
		       winding, twist, skip,
		       island, avenode, groupCCW, windingNextBest );
	  twists.push_back( twist );
	  skips.push_back( skip );
	  islands.push_back( island );
	  avenodes.push_back( avenode );
	}

// 	cerr << " Length selection        "
// 	     << LengthMin.first << "  "
// 	     << LengthMin.second << "  "
// 	     << LengthMin2.first << "  "
// 	     << LengthMin2.second << "  "
// 	     << endl;

// 	cerr << " Angle selection         "
// 	     << AngleMin.first << "  "
// 	     << AngleMin.second << "  "
// 	     << AngleMin2.first << "  "
// 	     << AngleMin2.second << "  "
// 	     << endl;

	// This is for island analysis - if the initial centroid
	// locations are provided which have the winding value from
	// the island then make sure the new analysis gives the same
	// winding.

	if( desiredWinding && windings[count] != desiredWinding )
	  windings[count] = 0;
      }

      cerr << "End of streamline " << count
	   << "  windings " << winding 
	   << "  twists " << twist
	   << "  skip "   << skip
	   << "  groupCCW " << groupCCW
	   << "  islands " << island
	   << "  windingNextBest " << (windingNextBest>0 ? windingNextBest : 0)
	   << endl;

    // If the twists is a factorial of the winding then rebin the points.
      if( 0 && !override && winding && twist != 1 &&
	  factorial( winding, twist ) ) {
	
	unsigned int fact;
	while( fact = factorial( winding, twist ) ) {
	  winding /= fact;
	twist /= fact;
	}
	
	windings[count] = winding;
	twists[count]   = twist;
      }
      count++;
    }
  }


  // Now bin the points.
  for( unsigned int c=0; c<count; c++ ) {

    cerr << c << " STARTING" << endl;

    unsigned int winding = windings[c];
    unsigned int twist   = twists[c];
    unsigned int skip    = skips[c];
    unsigned int island  = islands[c];
    bool groupCCW;
    bool completeIslands = true;

    if( winding == 0 ) {
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
      dataItr = ifield->fdata().begin();
      imesh->begin( inodeItr );

      // Skip all of the points between the beginning and the first
      // point found before (CCW) or after (CW) the zero plane.
      while( *inodeItr != inodeGlobalStart[c] ) {
	++inodeItr;
	++dataItr;
      }

      imesh->get_center(lastPt, *inodeItr);
      lastAng = atan2( lastPt.y(), lastPt.x() );

      // If the plane is near PI which where through zero happens with
      // atan2 adjust the angle so that the through zero is at 0 - 2PI
      // instead.
      if( nearPI && lastAng < 0 ) lastAng += 2 * M_PI;

      ++inodeItr;
      ++dataItr;
      
      ostringstream str;
      str << "Streamline " << c+1 << " Node Index";
      if( !ifield->get_property( str.str(), inodeNext ) )
	inodeNext = *inodeEnd;

//      cerr << "Starting new streamline binning " << c << endl;

      while (*inodeItr != inodeNext) {
	
	imesh->get_center(currPt, *inodeItr);
	currAng = atan2( currPt.y(), currPt.x() );
	// If the plane is near PI which where through zero happens with
	// atan2 adjust the angle so that the through zero is at 0 - 2PI
	// instead.
	if( nearPI && currAng < 0 ) currAng += 2 * M_PI;

	// First look at only points that are in the correct plane.
	if( ( CCWstreamline && lastAng < plane && plane <= currAng) ||
	    (!CCWstreamline && currAng < plane && plane <= lastAng) ) {
	  double t;

	  if( fabs(currAng-lastAng) > 1.0e-12 )
	    t = (plane-lastAng) / (currAng-lastAng);
	  else
	    t = 0;
	  
	  Point point = interpert( lastPt, currPt, t );

	  if( bins[p].size() <= (unsigned int) bin ) {
	    vector< pair<Point, double> > sub_bin;
	  
	    sub_bin.push_back( pair<Point, double>(point, *dataItr) );

	    bins[p].push_back( sub_bin );

	  } else {
	    bins[p][bin].push_back( pair<Point, double>(point, *dataItr) );
	  }

	  bin = (bin + 1) % winding;
	}

	lastPt  = currPt;
	lastAng = currAng;
    
	++inodeItr;
	++dataItr;
      }
    }
    
    unsigned int nnodes = (int) 1e8;
    bool VALID = true;

    // Sanity check
    for( unsigned int p=0; p<planes.size(); p++ ) {
      for( unsigned int i=0; i<winding; i++ ) {

// 	for( unsigned int j=0, k=bins[p][i].size()-1;
// 	     j<bins[p][i].size()/2; j++, k-- ) {

// 	  pair<Point, double> tmp_bin = bins[p][i][j];
// 	  bins[p][i][j] = bins[p][i][k];
// 	  bins[p][i][k] = tmp_bin;
// 	}

	if( nnodes > bins[p][i].size() )
	  nnodes = bins[p][i].size();

	if( bins[p][i].size() < 1 ) {
	  cerr << "INVALID - Plane " << p
	       << " bin  " << i
	       << " number of points " << bins[p][i].size()
	       << endl;

	  VALID = false;

	  return;
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
    Vector centroidGlobal(0,0,0);
    vector< Vector > localCentroids;
    vector< Vector > localSeparatrices[2];

    localCentroids.resize(winding);
    localSeparatrices[0].resize(winding);
    localSeparatrices[1].resize(winding);

    for( unsigned int i=0; i<winding; i++ ) {
      localCentroids[i] = Vector(0,0,0);
      
      for( unsigned int j=0; j<bins[p][i].size(); j++ ) 
	localCentroids[i] += (Vector) bins[p][i][j].first;

      if( bins[p][i].size() ) {
	localCentroids[i] /= (double) bins[p][i].size();

	centroidGlobal += localCentroids[i];
      }
    }

    centroidGlobal /= winding;

    // Get the direction of the points within a group.
    Vector v0 = (Vector) bins[p][0][0].first - centroidGlobal;
    Vector v1 = (Vector) bins[p][0][1].first - centroidGlobal;

    groupCCW = (ccw( v0, v1 ) == 1);

//    cerr << 0 << "  " << groupCCW << endl;

    if( island ) {
      for( unsigned int i=0; i<winding; i++ ) {

	unsigned int startIndex;
	unsigned int middleIndex;
	unsigned int stopIndex;
	
	unsigned int turns = islandCheck( bins[p][i], centroidGlobal,
					  startIndex, middleIndex, stopIndex );

	if( turns < 3 )
	  completeIslands = false;

	if( turns >= 2 ) {
// 	  localSeparatrices[0][i] = (Vector) bins[p][i][startIndex].first;
// 	  localSeparatrices[1][i] = (Vector) bins[p][i][middleIndex].first;
	}

	if( turns == 3 ) {
 	  unsigned int index0 = (middleIndex - startIndex ) / 2;
 	  unsigned int index1 = (  stopIndex - middleIndex) / 2;

	  unsigned int nodes = stopIndex - startIndex + 1;

 	  cerr << "Indexes mid " << nodes << " nodes "
	       << "  " << ( startIndex + index0)%nodes 
	       << "  " << (middleIndex - index0)%nodes
	       << "  " << (middleIndex + index1)%nodes
	       << "  " << (  stopIndex - index1)%nodes << endl;

 	  localCentroids[i] =
 	    ( (Vector) bins[p][i][( startIndex + index0)%nodes].first + 
 	      (Vector) bins[p][i][(middleIndex - index0)%nodes].first +

 	      (Vector) bins[p][i][(middleIndex + index1)%nodes].first + 
 	      (Vector) bins[p][i][(  stopIndex - index1)%nodes].first ) / 4.0;
	}
      }
    }


    for( unsigned int p=0; p<planes.size(); p++ ) {
      if( overlaps == 1 || overlaps == 3 )
	removeOverlap( bins[p], nnodes, winding, twist, skip, island );
      
      if( overlaps == 2 )
	mergeOverlap( bins[p], nnodes, winding, twist, skip, island );
      else if( overlaps == 3 )
	smoothCurve( bins[p], nnodes, winding, twist, skip, island );

      // Sanity check
      for( unsigned int i=0; i<winding; i++ ) {

	if( nnodes > bins[p][i].size() )
	  nnodes = bins[p][i].size();

	if( bins[p][i].size() < 1 ) {
	  cerr << "INVALID - Plane " << p
	       << " bin  " << i
	       << " number of points " << bins[p][i].size()
	       << endl;

	  VALID = false;

	  return;
	}
	  
	cerr << "Surface " << c
	     << " plane " << p
	     << " bin " << i
	     << " base number of nodes " << nnodes
	     << " number of points " << bins[p][i].size()
	     << endl;
      }
    }

    cerr << "Surface " << c << " is a "
	 << winding << ":" << twist << " surface ("
	 << (double) winding / (double) twist << ") ";
    
    if( island > 0 ) 
      cerr << "that contains " << island << " islands"
	   << (completeIslands ? " (Complete)" : "");
    
    cerr << " and has " << nnodes << " nodes"
	 << endl;
        
    if( island && island != winding ) {
      cerr << "WARNING - The island count does not match the winding count" 
	   << endl;
    }
    
    // Record the topology.
    pair< unsigned int, unsigned int > topo( winding, twist );
    topology.push_back(topo);

    if( !curveField ) {

      vector< unsigned int > dims;
      
      dims.resize(2);
      
      dims[0] = nnodes;
      dims[1] = (planes.size()+1) * winding;
      
      ((SQSMesh *) omesh)->set_dim( dims );

      ofield->resize_fdata();

      cerr << "Creating a surface of " << dims[0] << "  " << dims[1] << endl;
    }

    double color_value = 0;

    if( color == 1 )
      color_value = c;
    else if( color == 6 )
      color_value = winding;
    else if( color == 7 )
      color_value = twist;
    else if( color == 8 )
      color_value = (double) winding / (double) twist;


    if( island ) {

      if( completeIslands ) {

 	if( baseCentroids.size() ) {

	  unsigned int cc = 0;

	  // Find the bounds where the islands are for this winding.
	  while( cc < windings.size() ) {
	    if( cc <= c && c < cc+baseCentroidsWinding[cc] )
	      break;
	    else
	      cc += baseCentroidsWinding[cc];
	  }

	  cerr << "Searching winding " << baseCentroidsWinding[cc] << endl;

	  for( unsigned int i=0; i<winding; i++ ) {
	    
	    unsigned int index;
	    double mindist = 1.0e12;
	    
	    for( unsigned int j=cc; j<cc+winding; j++ ) {

	      double dist = (localCentroids[i] - baseCentroids[j][0]).length();
	      
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

	  for( unsigned int i=0; i<winding; i++ ) {
	    // Centroids
	    cerr << i << "               " << localCentroids[i] << endl;

	    typename PCFIELD::mesh_type::Node::index_type n =
	      opccmesh->add_node((Point) localCentroids[i]);
	    opccfield->resize_fdata();
	    opccfield->set_value( (double) i, n );

	    // Separatrices
	    unsigned int j = (i+skip) % winding;

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

	    n = opcsmesh->add_node((Point) ((localSeparatrices[ii][i] +
					     localSeparatrices[jj][j])/2.0));

	    opcsfield->resize_fdata();
	    opcsfield->set_value( (double) i, n);

//  	    n = opcsmesh->add_node((Point) localSeparatrices[0][i]);
// 	    opcsfield->resize_fdata();
// 	    opcsfield->set_value(0, n);

//  	    n = opcsmesh->add_node((Point) localSeparatrices[1][j]);
// 	    opcsfield->resize_fdata();
// 	    opcsfield->set_value(1, n);

	    cerr << c << "  Separatrices " << i << "  " << j << endl;
	  }
	}
      }
    } else { // Surface
    }

    if( !showIslands || ( showIslands && island ) ) {

      // Add the points into the return field.
      for( unsigned int p=0; p<planes.size(); p++ ) {
	for( unsigned int i=0; i<winding; i++ ) {

	  lock.lock();

	  if( curveField ) {
// 	    cerr << "Loading curve " << p * winding + i
// 		 << " plane " << planes.size() << " winding " << i;

	    loadCurve( ofield_h, bins[p][i],
		       planes.size(), winding,
		       bins[p][i].size(), p, i,
		       color, color_value );

// 	    cerr << " done";

	  } else {
// 	    cerr << "Loading surface " << p * winding + i
// 		 << " plane " << planes.size() << " winding " << i;

	    if( p == planes.size()-1 ) {
	      unsigned int j = (i-1 + winding) % winding;

	      loadSurface( ofield_h, bins[p][i],
			   planes.size()+1, winding, nnodes, p, j,
			   color, color_value );
	    } else {
	      loadSurface( ofield_h, bins[p][i],
			   planes.size()+1, winding, nnodes, p, i,
			   color, color_value );
	    }

// 	    cerr << " done";
	  }

	  lock.unlock();

//	  cerr << " unlocked " << endl;
	}
      }

      // For a surface add in the first set again so that the surface
      // is complete. However because the complete surface has
      if( !curveField ) {
	for( unsigned int i=0; i<winding; i++ ) {
	  lock.lock();

	  unsigned int j = (i-1 + winding) % winding;

// 	  cerr << "Loading surface " << planes.size() * winding + i
// 	       << " plane " << planes.size() << " winding " << j;

	  loadSurface( ofield_h, bins[0][i],
		       planes.size()+1, winding, nnodes,
		       planes.size(), j, color, color_value );
	
// 	  cerr << " done" << endl;

	  lock.unlock();
	}
      }
    }

    for( unsigned int p=0; p<planes.size(); p++ ) {
	for( unsigned int i=0; i<winding; i++ ) {
	  bins[p][i].clear();
	}

	bins[p].clear();
    }

//     cerr << c << " DONE" << endl;
  }

//   cerr << "DONE DONE" << endl;

  if( baseCentroids.size() ) {

    if( islandCentroids ) {

      typename PCFIELD::mesh_type::Node::index_type n;

      for( unsigned int i=0; i<baseCentroids.size(); i++ ) {

	if( baseCentroids[i].size() ) {
	  Vector tmpCentroid(0,0,0);
	
	  for( unsigned int j=1; j<baseCentroids[i].size(); j++ )
	    tmpCentroid += baseCentroids[i][j];
	
	  tmpCentroid /= (double) (baseCentroids[i].size() - 1);
	
	  // 	cerr << tmpCentroid << endl;

	  n = opccmesh->add_node((Point) tmpCentroid);
	} else {
	  n = opccmesh->add_node((Point) baseCentroids[i][0]);
	}

	opccfield->resize_fdata();
//	opccfield->set_value( (double) windings[index], n );
      }
    } else { // Produce seed to show the NULL fieldline.
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

	cerr << "New centroid " << newCentroid[index]
	     << " index " << index << endl;

	typename PCFIELD::mesh_type::Node::index_type n =
	  opccmesh->add_node((Point) newCentroid[index]);
	opccfield->resize_fdata();
	opccfield->set_value( (double) windings[index], n );

	cc += windings[cc];

	minSD = 1.0e12;
      }
    }
  }
}

template< class IFIELD, class OFIELD, class PCFIELD >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
islandCheck( vector< pair< Point, double > > &bins,
	     Vector centroidGlobal,
	     unsigned int &startIndex,
	     unsigned int &middleIndex,
	     unsigned int &stopIndex )
{
  // Determine if islands exists. If an island exists there will be
  // both clockwise and counterclockwise sections when compared to the
  // main centroid.
  unsigned int turns = 0;

  startIndex = middleIndex = stopIndex = 0;

  Vector v0 = (Vector) bins[0].first - centroidGlobal;
  Vector v1 = (Vector) bins[1].first - centroidGlobal;

  bool lastCCW = (ccw(v0, v1) == 1);
  v0 = v1;

  for( unsigned int j=2; j<bins.size(); j++ ) {

    v1 = (Vector) bins[j].first - centroidGlobal;

    bool CCW = (ccw(v0, v1) == 1);
    v0 = v1;

    if( CCW != lastCCW ) {
      turns++;

      if( turns == 1 )      startIndex  = j - 1;
      else if( turns == 2 ) middleIndex = j - 1;
      else if( turns == 3 ) stopIndex   = j - 1;

      if( turns == 3 )
	break;

      lastCCW = CCW;
    }
  }

  if( turns == 3 ) {

    // Check for a negative epsilon.
    unsigned int nodes = stopIndex - startIndex + 1;

    double length0 = 0;
    double length1 = 0;

    for( unsigned j=0, k=nodes, l=nodes+1; l<bins.size(); j++, k++, l++ ) {
      length0 += ((Vector) bins[j].first - (Vector) bins[k].first).length();
      length1 += ((Vector) bins[j].first - (Vector) bins[l].first).length();
    }

    if( length0 < length1 )
      stopIndex--;

    cerr << "complete 3 turns\n";

    if( 2*startIndex == middleIndex + 1 ) {
      // First point is actually the start point.
      cerr << "First point is actually the start point.\n";

      stopIndex   = middleIndex;
      middleIndex = startIndex;
      startIndex  = 0;
    }

  } else if( turns == 2 ) {

    if( 2*startIndex == middleIndex + 1 ) {
      // First point is actually the start point.
      cerr << "First point is actually the start point.\n";

      stopIndex   = middleIndex;
      middleIndex = startIndex;
      startIndex  = 0;
      
      turns = 3;
    } else if( bins.size() < 2 * (middleIndex - startIndex) - 1 ) {
      // No possible over lap.
      cerr <<  "islandCheck - No possible over lap.\n";

      stopIndex = startIndex + bins.size() - 1;

    } else {	  
      // See if the first point overlaps another section.
      for( unsigned int  j=middleIndex+1; j<bins.size(); j++ ) {
	if( Dot( (Vector) bins[j  ].first - (Vector) bins[0].first,
		 (Vector) bins[j-1].first - (Vector) bins[0].first )
	    < 0 ) {
	  stopIndex = startIndex + (j-1) + 1; // Add one for the zeroth 
	  turns = 3;
	  cerr <<  "islandCheck - First point overlaps another section after " << j-1 << endl;
	  break;
	}
      }
      
      // See if a point overlaps the first section.
      if( turns == 2 )
	for( unsigned int j=middleIndex; j<bins.size(); j++ ) {
	  if( Dot( (Vector) bins[0].first - (Vector) bins[j].first,
		   (Vector) bins[1].first - (Vector) bins[j].first )
	      < 0 ) {
	    stopIndex = startIndex + (j-1) + 1; // Add one for the zeroth 
	    turns = 3;
	    cerr << "islandCheck - A point overlaps the first section at " << j-1 << endl;
	    break;
	  }
	}

      // No overlap found
      if( turns == 2 ) {
	stopIndex = startIndex + bins.size() - 1;
	cerr << "islandCheck - No overlap found\n";
      }
    }
  } else {
    startIndex  = 0;
    middleIndex = 0;
    stopIndex   = bins.size() - 1;
  }

  cerr << "Turns "
       << turns  << "  "
       << "Indexes "
       << startIndex  << "  "
       << middleIndex << "  "
       << stopIndex   << endl;

  return turns;
}


template< class IFIELD, class OFIELD, class PCFIELD >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
surfaceCheck( vector< vector< pair< Point, double > > > &bins,
	      unsigned int winding,
	      unsigned int skip,
	      Vector centroidGlobal,
	      unsigned int &nnodes )
{
  for( unsigned int i=0; i<winding; i++ ) {
    // First make sure none of the groups overlap themselves.
    Vector v0 = Vector( (Vector) bins[i][0].first - centroidGlobal );
    
    double angleSum = 0;
    
    for( unsigned int j=1; j<nnodes; j++ ) {
      Vector v1 = Vector( (Vector) bins[i][j].first - centroidGlobal );
      
      angleSum += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );
      
      if( angleSum > 2.0 * M_PI ) {
	nnodes = j;
	break;
      } else
	v0 = v1;
    }
  }

  // Second make sure none of the groups overlap each other.
  double angleSum = 4.0 * M_PI;

  while ( angleSum > 2.0 * M_PI ) {
    angleSum = 0.0;

    for( unsigned int i=0; i<winding && angleSum <= 2.0 * M_PI; i++ ) {
      
      Vector v0 = Vector( (Vector) bins[i][0].first - centroidGlobal );
      
      for( unsigned int j=1; j<nnodes && angleSum <= 2.0 * M_PI; j++ ) {
	Vector v1 = Vector( (Vector) bins[i][j].first - centroidGlobal );
	
	angleSum += acos( Dot( v0, v1 ) / (v0.length() * v1.length()) );
	
	v0 = v1;
      }
    }

    if( angleSum > 2.0 * M_PI ) {
      --nnodes;
    }
  }

  // Because of gaps between groups it is possible to still have over
  // laps so check between the groups.
  for( unsigned int i=0, j=skip; i<skip*winding; i+=skip, j+=skip ) {
    unsigned int i0 = i % winding;
    unsigned int j0 = j % winding;

    for( unsigned int k=nnodes-1; k>0; k-- ) {
      Vector v0 = (Vector) bins[j0][0].first - (Vector) bins[i0][k].first;

      for( unsigned int l=1; l<nnodes; l++ ) {

	Vector v1 = (Vector) bins[j0][l].first - (Vector) bins[i0][k].first;
	
	if( Dot( v0, v1 ) < 0.0 ) {
	  nnodes = k;
	  break;
	} else
	  v0 = v1;
      }
    }
  }

  return nnodes;
}


template< class IFIELD, class OFIELD, class PCFIELD >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
surfaceCheck( vector< vector< pair< Point, double > > > &bins,
	      unsigned int i,
	      unsigned int j,
	      unsigned int nnodes ) {

  unsigned int nodes = nnodes;

  while( nodes < bins[i].size() ) {
    // Check to see if the first overlapping point is really a
    // fill-in point. This happens because the spacing between
    // winding groups varries between groups.
    Vector v0 = (Vector) bins[j][0      ].first - (Vector) bins[i][nodes].first;
    Vector v1 = (Vector) bins[i][nodes-1].first - (Vector) bins[i][nodes].first;
    
    if( Dot( v0, v1 ) < 0.0 )
      nodes++;
    else
      break;
  }

  return nodes;
}


template< class IFIELD, class OFIELD, class PCFIELD >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
removeOverlap( vector< vector < pair< Point, double > > > &bins,
	       unsigned int &nnodes,
	       unsigned int winding,
	       unsigned int twist,
	       unsigned int skip,
	       unsigned int island )
{
  Vector centroidGlobal = Vector(0,0,0);;

  for( unsigned int i=0; i<winding; i++ )
    for( unsigned int j=0; j<nnodes; j++ )
      centroidGlobal += (Vector) bins[i][j].first;
  
  centroidGlobal /= (winding*nnodes);
    
  unsigned int avennodes = 0;

  if( island ) {

    for( unsigned int i=0; i<winding; i++ ) {

      unsigned int startIndex;
      unsigned int middleIndex;
      unsigned int stopIndex;

      unsigned int turns = islandCheck( bins[i], centroidGlobal,
					startIndex, middleIndex, stopIndex );

      unsigned int nodes = 0;
      bool completeIsland = false;

      if( turns <= 1 ) {
	nodes = bins[i].size();

      } else {
	if( 2*startIndex == middleIndex + 1 ) {
	  // First point is actually the start point.
	  cerr << "removeOverlap - First point is actually the start point.\n";

	  nodes = middleIndex;
	  completeIsland = true;

	} else if( bins[i].size() < 2 * (middleIndex - startIndex) - 1 ) {
	  // No possible over lap.
	  cerr <<  "removeOverlap - No possible over lap.\n";

	  nodes = bins[i].size();

	} else {
	  // See if the first point overlaps another section.
	  for( unsigned int  j=middleIndex+1; j<bins[i].size(); j++ ) {
	    if( Dot( (Vector) bins[i][j  ].first - (Vector) bins[i][0].first,
		     (Vector) bins[i][j-1].first - (Vector) bins[i][0].first )
		< 0 ) {

	      // Check for bogus overlaps.
	      if( turns == 3 && j > stopIndex - startIndex + 2 ) {
		cerr << "removeOverlap - Bogus overlap section at " << j-1 << endl;
		nodes = stopIndex - startIndex + 1;
	      } else {

		cerr <<  "removeOverlap - First point overlaps another section after " << j-1 << endl;
		nodes = j;
	      }

	      completeIsland = true;
	      break;
	    }
	  }
      
	  // See if a point overlaps the first section.
	  if( nodes == 0 )
	    for( unsigned int j=middleIndex; j<bins[i].size(); j++ ) {
	      if( Dot( (Vector) bins[i][0].first - (Vector) bins[i][j].first,
		       (Vector) bins[i][1].first - (Vector) bins[i][j].first )
		  < 0 ) {
		// Check for bogus overlaps.
		if( turns == 3 && j > stopIndex - startIndex + 2 ) {
		  cerr << "removeOverlap - Bogus overlap section at " << j-1 << endl;
		  nodes = stopIndex - startIndex + 1;
		} else {
		  cerr << "removeOverlap - A point overlaps the first section at " << j-1 << endl;
		  nodes = j;
		}

		completeIsland = true;
		break;
	      }
	    }

	  // No overlap found
	  if( nodes == 0 ) {
	    if( turns == 3 ) {
	      nodes = stopIndex - startIndex + 1;
	      cerr << "removeOverlap - No overlap found using the indexes instead\n";
	      if( nodes <= bins[i].size() )
		completeIsland = true;
	      else
		nodes = bins[i].size();
	    } else {
	      nodes = bins[i].size();
	      cerr << "No overlap found\n";
	    }
	  }
	}
      }

      if( nodes * i != avennodes )
	cerr << i << " nnodes mismatch ";

      avennodes += nodes;

      // Erase all of the overlapping points.
      bins[i].erase( bins[i].begin()+nodes, bins[i].end() );

      // Close the island if it is complete
      if( completeIsland ) {
	bins[i].push_back( bins[i][0] );

	avennodes += 1;
      }
    }

  } else {  // Surface

    // This gives the total number of node for each group.
    surfaceCheck( bins, winding, skip, centroidGlobal, nnodes );

    for( unsigned int i=0; i<winding; i++ ) {
      unsigned int nodes = surfaceCheck( bins, i, (i+skip)%winding, nnodes );
      
      avennodes += nodes;

      // Erase all of the overlapping points.
      bins[i].erase( bins[i].begin()+nodes, bins[i].end() );
    }

    if( (unsigned int) (avennodes / winding) * winding != avennodes )
      cerr << "nnodes mismatch "
	   << " nodes " << avennodes / winding
	   << " avennodes " << (float) avennodes / (float) winding
	   << endl;


  }

  nnodes = (unsigned int) ((double) avennodes / (double) winding + 0.5);

  if( nnodes * winding != avennodes )
    cerr << " overall nnodes mismatch ";

  cerr << endl;

  return nnodes;
}


template< class IFIELD, class OFIELD, class PCFIELD >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
smoothCurve( vector< vector < pair< Point, double > > > &bins,
	     unsigned int &nnodes,
	     unsigned int winding,
	     unsigned int twist,
	     unsigned int skip,
	     unsigned int island )
{
  Vector centroidGlobal = Vector(0,0,0);;

  for( unsigned int i=0; i<winding; i++ )
    for( unsigned int j=0; j<nnodes; j++ )
      centroidGlobal += (Vector) bins[i][j].first;
  
  centroidGlobal /= (winding*nnodes);

  unsigned int add = 2;

  if( island ) {

    for( unsigned int i=0; i<winding; i++ ) {
//      for( unsigned int s=0; s<add; s++ )
      {
	pair< Point, double > newPts[add*nnodes];

	for( unsigned int j=0; j<add*nnodes; j++ )
	  newPts[j] = pair< Point, double > (Point(0,0,0), 0 );
	
	for( unsigned int j=1; j<nnodes-1; j++ ) {

	  unsigned int j_1 = (j-1+nnodes) % nnodes;
	  unsigned int j1  = (j+1+nnodes) % nnodes;

	  Vector v0 = (Vector) bins[i][j1].first - (Vector) bins[i][j  ].first;
	  Vector v1 = (Vector) bins[i][j ].first - (Vector) bins[i][j_1].first;

 	  cerr << i << " smooth " << j_1 << " "  << j << " "  << j1 << "  "
 	       << ( v0.length() > v1.length() ?
 		    v0.length() / v1.length() :
 		    v1.length() / v0.length() ) << endl;

	  if( Dot( v0, v1 ) > 0 &&
	      ( v0.length() > v1.length() ?
		v0.length() / v1.length() :
		v1.length() / v0.length() ) < 10.0 ) {

	    Vector center = (Vector) circle( bins[i][j_1].first,
					     bins[i][j  ].first,
					     bins[i][j1 ].first );

	    double rad = ((Vector) bins[i][j].first - center).length();


	    for( unsigned int s=0; s<add; s++ ) {
	      Vector midPt = (Vector) bins[i][j_1].first +
		(double) (add-s) / (double) (add+1) *
		((Vector) bins[i][j].first - (Vector) bins[i][j_1].first );
		

	      Vector midVec = midPt - center;

	      midVec.safe_normalize();

	      newPts[add*j+s].first += center + midVec * rad;
	      newPts[add*j+s].second += 1.0;

	      midPt = (Vector) bins[i][j].first +
		(double) (add-s) / (double) (add+1) *
		((Vector) bins[i][j1].first - (Vector) bins[i][j].first );

	      midVec = midPt - center;

	      midVec.safe_normalize();

	      newPts[add*j1+s].first += center + midVec * rad;
	      newPts[add*j1+s].second += 1.0;
	    }
	  }
	}

	for( unsigned int j=nnodes-1; j>0; j-- ) {

	  for( unsigned int s=0; s<add; s++ ) {

	    unsigned int k = add * j + s;

	    if( newPts[k].second > 0 ) {
	      
	      newPts[k].first /= newPts[k].second;
	      
//	      cerr << i << " insert " << j << "  " << newPts[k].first << endl;
	      
	      bins[i].insert( bins[i].begin()+j, newPts[k] );
	    }
	  }
	}

	for( unsigned int s=0; s<add; s++ ) {

	  unsigned int k = add - 1 - s;

	  if( newPts[k].second > 0 ) {
	      
	    newPts[k].first /= newPts[k].second;
	      
//	      cerr << i << " insert " << 0 << "  " << newPts[k].first << endl;
	      
	    bins[i].push_back( newPts[k] );
	  }
	}
      }
    }

  } else {

    for( unsigned int i=0; i<winding; i++ ) {

      if( bins[i].size() < 2 )
	continue;

      // Index of the next winding group
      unsigned int j = (i+skip)%winding;

      // Insert the first point from the next winding so the curve
      // is contiguous.
      bins[i].push_back( bins[j][0] );

      //for( unsigned int s=0; s<add; s++ )
      {
	unsigned int nodes = bins[i].size();

	pair< Point, double > newPts[add*nodes];

	for( unsigned int j=0; j<add*nodes; j++ )
	  newPts[j] = pair< Point, double > (Point(0,0,0), 0 );
	
	for( unsigned int j=1; j<nodes-1; j++ ) {

	  unsigned int j_1 = j - 1;
	  unsigned int j1  = j + 1;

	  Vector v0 = (Vector) bins[i][j1].first - (Vector) bins[i][j  ].first;
	  Vector v1 = (Vector) bins[i][j ].first - (Vector) bins[i][j_1].first;

// 	  cerr << i << " smooth " << j_1 << " "  << j << " "  << j1 << "  "
// 	       << ( v0.length() > v1.length() ?
// 		    v0.length() / v1.length() :
// 		    v1.length() / v0.length() ) << endl;

	  if( Dot( v0, v1 ) > 0 &&
	      ( v0.length() > v1.length() ?
		v0.length() / v1.length() :
		v1.length() / v0.length() ) < 10.0 ) {

	    Vector center = (Vector) circle( bins[i][j_1].first,
					     bins[i][j  ].first,
					     bins[i][j1 ].first );

	    double rad = ((Vector) bins[i][j].first - center).length();


	    for( unsigned int s=0; s<add; s++ ) {
	      Vector midPt = (Vector) bins[i][j_1].first +
		(double) (add-s) / (double) (add+1) *
		((Vector) bins[i][j].first - (Vector) bins[i][j_1].first );
		

	      Vector midVec = midPt - center;

	      midVec.safe_normalize();

	      newPts[add*j+s].first += center + midVec * rad;
	      newPts[add*j+s].second += 1.0;

	      midPt = (Vector) bins[i][j].first +
		(double) (add-s) / (double) (add+1) *
		((Vector) bins[i][j1].first - (Vector) bins[i][j].first );

	      midVec = midPt - center;

	      midVec.safe_normalize();

	      newPts[add*j1+s].first += center + midVec * rad;
	      newPts[add*j1+s].second += 1.0;
	    }
	  }
	}

	for( int j=nodes-1; j>=0; j-- ) {

	  for( unsigned int s=0; s<add; s++ ) {

	    unsigned int k = add * j + s;

	    if( newPts[k].second > 0 ) {
	      
	      newPts[k].first /= newPts[k].second;
	      
//	      cerr << i << " insert " << j << "  " << newPts[k].first << endl;
	      
	      bins[i].insert( bins[i].begin()+j, newPts[k] );
	    }
	  }
	}
      }

      // Remove the last point so it is possilble to see the groups.
      bins[i].erase( bins[i].end() );
    }
  }

  return winding*(add+1)*nnodes;
}


template< class IFIELD, class OFIELD, class PCFIELD >
unsigned int
StreamlineAnalyzerAlgoT<IFIELD, OFIELD, PCFIELD>::
mergeOverlap( vector< vector < pair< Point, double > > > &bins,
	      unsigned int &nnodes,
	      unsigned int winding,
	      unsigned int twist,
	      unsigned int skip,
	      unsigned int island )
{
  Vector centroidGlobal = Vector(0,0,0);;

  for( unsigned int i=0; i<winding; i++ )
    for( unsigned int j=0; j<nnodes; j++ )
      centroidGlobal += (Vector) bins[i][j].first;
  
  centroidGlobal /= (winding*nnodes);
    
  if( island ) {

    vector < pair< Point, double > > tmp_bins[winding];

    for( unsigned int i=0; i<winding; i++ ) {
      
      unsigned int startIndex;
      unsigned int middleIndex;
      unsigned int stopIndex;

      // Merge only if there are overlapping points.
      if( islandCheck( bins[i], centroidGlobal,
		       startIndex, middleIndex, stopIndex ) == 3 ) {
	nnodes = stopIndex - startIndex + 1;

	if( nnodes == bins[i].size() )
	  continue;

	// Store the overlapping points.
	for( unsigned int j=nnodes; j<bins[i].size(); j++ )
	  tmp_bins[i].push_back( bins[i][j] );

	cerr << i << " stored extra points " << tmp_bins[i].size() << endl;

	// Erase all of the overlapping points.
	bins[i].erase( bins[i].begin()+nnodes, bins[i].end() );
	  
	// Insert the first point so the curve is contiguous.
	bins[i].insert( bins[i].begin()+nnodes, bins[i][0] );

	unsigned int index_prediction = 1;
	unsigned int prediction_true  = 0;
	unsigned int prediction_false = 0;

	unsigned int modulo = bins[i].size() - 1;
 
	// Insert the remaining points.
	for( unsigned int j=0; j<tmp_bins[i].size(); j++ ) {

	  Vector v0 = (Vector) bins[i][0].first -
	    (Vector) tmp_bins[i][j].first;

	  double angle = 0;
	  double length = 99999;
	  unsigned int angleIndex = 0;
	  unsigned int lengthIndex = 0;

	  for( unsigned int k=1; k<bins[i].size(); k++ ) {

	    Vector v1 = (Vector) bins[i][k].first -
	      (Vector) tmp_bins[i][j].first;

	    double ang = acos( Dot(v0, v1) / (v0.length() * v1.length()) );

	    if( angle < ang ) {
	      angle = ang;
	      angleIndex = k;
	    }

	    if( length < v1.length() ) {
	      length = v1.length();
	      lengthIndex = k;
	    }

	    // Go on.
	    v0 = v1;
	  }

	  // Insert it between the other two.
	  if( angle > M_PI / 3.0 )
	    bins[i].insert( bins[i].begin()+angleIndex, tmp_bins[i][j] );

	  cerr << i << "  " << modulo << "  " << j + nnodes
	       << "  Prediction " << index_prediction
	       << " actual " << angleIndex << "  "
	       << (angleIndex == index_prediction) << endl;

	  // Check to see if the prediction and the actual index are
	  // the same.
	  if( angleIndex == index_prediction )
	    prediction_true++;
	  else // if( angleIndex != index_prediction )
	    prediction_false++;

	  // Predict where the next insertion will take place.
	  if( (j+1) % modulo == 0 )
	    index_prediction = 1 + (j+1) / modulo;
	  else
	    index_prediction = angleIndex + (j+1) / modulo + 2;
	}

	cerr << "Winding " << i << " inserted "
	     << prediction_true+prediction_false << " nodes "
	     << " True " << prediction_true
	     << " False " << prediction_false << endl;

	// If more of the predictions are incorrect than correct
	// insert based on the predictions.
	if( 0 && prediction_true < prediction_false ) {

	  cerr << "Winding " << i << " bad predicted insertion ";

	  unsigned int cc = 0;

	  for( unsigned int j=0; j<tmp_bins[i].size(); j++ ) {

 	    vector< pair< Point, double > >::iterator inList =
	      find( bins[i].begin(), bins[i].end(), tmp_bins[i][j] );
	      
	    if( inList != bins[i].end() ) {
	      bins[i].erase( inList );

	      cc++;
	    }
	  }

	  cerr << "removed " << cc << " points" << endl;

	  unsigned int index = 1;
	    
	  for( unsigned int j=0; j<tmp_bins[i].size(); j++ ) {
	    
	    // Insert it between the other two.
	    bins[i].insert( bins[i].begin()+index, tmp_bins[i][j] );

	    cerr << i << "  " << modulo << "  " << j + nnodes
		 << " actual " << index << endl;

	    // Predict where the next insertion will take place.
	    if( (j+1) % modulo == 0 )
	      index = 1 + (j+1) / modulo;
	    else
	      index += (j+1) / modulo + 2;
	  }
	}

	unsigned int start0  = 0;
	unsigned int end0    = 0;
	unsigned int start1  = 0;
	unsigned int end1    = 0;

	if( prediction_true > prediction_false ) {
	  // See if any of the segments cross.
	  for( unsigned int j=0; 0 && j<bins[i].size()-1; j++ ) {
	      
	    Point l0_p0 = bins[i][j].first;
	    Point l0_p1 = bins[i][j+1].first;
	      
	    for( unsigned int k=j+2; k<bins[i].size()-1; k++ ) {
		
	      Point l1_p0 = bins[i][k].first;
	      Point l1_p1 = bins[i][k+1].first;
		
	      if( intersect( l0_p0, l0_p1, l1_p0, l1_p1 ) == 1 ) {
		if( start0 == 0 ) {
		  start0  = j + 1;
		  end1    = k + 1;
		} else {
		  end0   = j + 1;
		  start1 = k + 1;

		  cerr << " merge self intersection " 
		       << start0 << "  " << end0 << "  "
		       << start1 << "  " << end1 << endl;

		  if( 0 ) {
		    vector < pair< Point, double > > tmp_bins[2];

		    for( unsigned int l=start0; l<end0; l++ )
		      tmp_bins[0].push_back( bins[i][l] );

		    for( unsigned int l=start1; l<end1; l++ )
		      tmp_bins[1].push_back( bins[i][l] );

		    bins[i].erase( bins[i].begin()+start1,
				   bins[i].begin()+end1 );

		    bins[i].erase( bins[i].begin()+start0,
				   bins[i].begin()+end0 );

		    for( unsigned int l=0; l<tmp_bins[1].size(); l++ )
		      bins[i].insert( bins[i].begin()+start0,
				      tmp_bins[1][l] );

		    for( unsigned int l=0; l<tmp_bins[0].size(); l++ )
		      bins[i].insert( bins[i].begin() + start1 -
				      tmp_bins[0].size() +
				      tmp_bins[1].size(),
				      tmp_bins[0][l] );
		  }

		  start0 = 0;
		}
	      }
	    }
	  }
	}
      }
    }
  } else {

    vector < pair< Point, double > > tmp_bins[winding];

    surfaceCheck( bins, winding, skip, centroidGlobal, nnodes );

    for( unsigned int i=0; i<winding; i++ ) {
      unsigned int j = (i+skip) % winding;

      unsigned int nodes = surfaceCheck( bins, i, j, nnodes );
      
      // Store the overlapping points.
      for( unsigned int j=nodes; j<bins[i].size(); j++ )
	tmp_bins[i].push_back( bins[i][j] );

      cerr << i << " stored extra points " << tmp_bins[i].size() << endl;

      // Erase all of the overlapping points.
      bins[i].erase( bins[i].begin()+nodes, bins[i].end() );

      // Insert the first point from the next winding so the curve
      // is contiguous.
      bins[i].push_back( bins[j][0] );
    }

    for( unsigned int i=0; i<winding; i++ ) {
      
      unsigned int winding_prediction = (i+skip)%winding;
      unsigned int index_prediction = 1;
      unsigned int prediction_true  = 0;
      unsigned int prediction_false = 0;

      for( unsigned int i0=0; i0<tmp_bins[i].size(); i0++ ) {

	double angle = 0;
	unsigned int index_wd = 0;
	unsigned int index_pt = 0;

	for( unsigned int j=0; j<winding; j++ ) {

	  Vector v0 = (Vector) bins[j][0].first -
	    (Vector) tmp_bins[i][i0].first;

	  for( unsigned int j0=1; j0<bins[j].size(); j0++ ) {
	    Vector v1 = (Vector) bins[j][j0].first -
	      (Vector) tmp_bins[i][i0].first;
	
	    double ang = acos( Dot(v0, v1) / (v0.length() * v1.length()) );

	    if( angle < ang ) {

	      angle = ang;
	      index_wd = j;
	      index_pt = j0;
	    }

	    // Go on.
	    v0 = v1;
	  }
	}

	// Insert it between the other two.
	bins[index_wd].insert( bins[index_wd].begin()+index_pt,
			       tmp_bins[i][i0] );

	cerr << "Winding prediction " << winding_prediction
	     << " actual " << index_wd
	     << "  Index prediction  " << index_prediction
	     << " actual " << index_pt << "  "
	     << (index_wd == winding_prediction &&
		 index_pt == index_prediction) << endl;

	// Check to see if the prediction of where the point was inserted
	// is correct;
	if( index_wd == winding_prediction && index_pt == index_prediction )
	  prediction_true++;
	else 
	  prediction_false++;

	// Prediction of where the next insertion will take place.
	index_prediction = index_pt + 2;
	winding_prediction = index_wd;
      }

      cerr << "Winding " << i << " inserted "
	   << prediction_true+prediction_false << " nodes "
	   << " True " << prediction_true
	   << " False " << prediction_false << endl;
    }

    // Remove the last point so it is possilble to see the groups.
    for( unsigned int i=0; i<winding; i++ )
      bins[i].erase( bins[i].end() );
  }

  // Update the approximate node count.
  nnodes = 9999;

  for( unsigned int i=0; i<winding; i++ )
    if( nnodes > bins[i].size() )
      nnodes = bins[i].size();

  return nnodes;
}

} // end namespace Fusion

#endif // StreamlineAnalyzer_h
