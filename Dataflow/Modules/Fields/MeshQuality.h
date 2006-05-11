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


//    File   : MeshQuality.h
//    Author : Jason Shepherd
//    Date   : January 2006

#if !defined(MeshQuality_h)
#define MeshQuality_h

#include <verdict.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/Constant.h>
#include <Dataflow/Network/Module.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <algorithm>
#include <set>

namespace SCIRun {

using std::copy;

class GuiInterface;

class MeshQualityAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};


//For Tets...
template <class FIELD>
class MeshQualityAlgoTet : public MeshQualityAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshQualityAlgoTet<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
//   enum tetMetrics {ASPECT_RATIO,   // this is aspect ratio 'beta'
//                    ASPECT_RATIO_GAMMA, 
//                    VOLUME, 
//                    CONDITION_NUMBER,
//                    JACOBIAN, NORM_JACOBIAN, 
//                    SHAPE, RELSIZE, SHAPE_SIZE,DISTORTION,
//                    ALLMETRICS, ALGEBRAIC, TRADITIONAL,                  
//                    NUM_TET_METRICS};
// // Note: if you want to add a new metric, do it immediately before 
// //       "ALLMETRICS" so that grouping capability is not broken

//   const int metricBitFlags[NUM_TET_METRICS] = 
//       {
//           V_TET_ASPECT_BETA,
//           V_TET_ASPECT_GAMMA,
//           V_TET_VOLUME,
//           V_TET_CONDITION,
//           V_TET_JACOBIAN,
//           V_TET_SCALED_JACOBIAN,
//           V_TET_SHAPE,
//           V_TET_RELATIVE_SIZE_SQUARED,
//           V_TET_SHAPE_AND_SIZE,
//           V_TET_DISTORTION,
//           V_TET_ALL,
//           V_TET_ALGEBRAIC,
//           V_TET_TRADITIONAL
//       };

  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());

  double node_pos[4][3];
  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);

  mesh->synchronize( Mesh::ALL_ELEMENTS_E );
  
  typedef GenericField<typename FIELD::mesh_type, ConstantBasis<double>, vector<double> > out_fld;
  out_fld *ofield = scinew out_fld( field->get_typed_mesh() );
  FieldHandle output = dynamic_cast<Field*>(ofield);
  if (output.get_rep() == 0)
  {
    return (fieldh); 
  }

  int total_elements = 0;
  double aspect_high = 0, aspect_low = 0, aspect_ave = 0;
  double aspect_gamma_high = 0, aspect_gamma_low = 0, aspect_gamma_ave = 0;
  double volume_high = 0, volume_low = 0, volume_ave = 0;
  double condition_high = 0, condition_low = 0, condition_ave = 0;
  double jacobian_high = 0, jacobian_low = 0, jacobian_ave = 0;
  double scaled_jacobian_high = 0, scaled_jacobian_low = 0, scaled_jacobian_ave = 0;
  double shape_high = 0, shape_low = 0, shape_ave = 0;
  double shape_size_high = 0, shape_size_low = 0, shape_size_ave = 0;
  double distortion_high = 0, distortion_low = 0, distortion_ave = 0;
  
  int inversions = 0;
  int first_time_thru = 1;

  while (bi != ei)
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    mesh->get_nodes(onodes, *bi);

    int i;
    for( i = 0; i < 4; i++ )
    {
      Point p;
      mesh->get_center( p, onodes[i] );
      node_pos[i][0] = p.x(); 
      node_pos[i][1] = p.y(); 
      node_pos[i][2] = p.z(); 
    }
    
    TetMetricVals values;
//    int verdict_metric = metricBitFlags[V_TET_ALL];
    int verdict_metric = V_TET_ALL;
    v_tet_quality(4, node_pos, verdict_metric, &values);

    double aspect = values.aspect_beta;
    double aspect_gamma = values.aspect_gamma;
    double volume = values.volume;
    double condition = values.condition;
    double jacobian = values.jacobian;
    double scaled_jacobian = values.scaled_jacobian;
    double shape = values.shape;
    double shape_size = values.shape_and_size;
    double distortion = values.distortion;

    if( first_time_thru )
    {
      aspect_high = aspect;
      aspect_low = aspect;
      aspect_gamma_high = aspect_gamma;
      aspect_gamma_low = aspect_gamma;
      volume_high = volume;
      volume_low = volume;
      condition_high = condition;
      condition_low = condition;
      jacobian_high = jacobian;
      jacobian_low = jacobian;
      scaled_jacobian_high = scaled_jacobian;
      scaled_jacobian_low = scaled_jacobian;
      shape_high = shape;
      shape_low = shape;
      shape_size_high = shape_size;
      shape_size_low = shape_size;
      distortion_high = distortion;
      distortion_low = distortion;
      first_time_thru = 0;
    }

    if( aspect > aspect_high )
        aspect_high = aspect;
    else if( aspect < aspect_low )
        aspect_low = aspect;
    aspect_ave += aspect;

    if( aspect_gamma > aspect_gamma_high )
        aspect_gamma_high = aspect_gamma;
    else if( aspect_gamma < aspect_gamma_low )
        aspect_gamma_low = aspect_gamma;
    aspect_gamma_ave += aspect_gamma;
    
    if( volume > volume_high )
        volume_high = volume;
    else if( volume < volume_low )
        volume_low = volume;
    volume_ave += volume;

    if( condition > condition_high )
        condition_high = condition;
    else if( condition < condition_low )
        condition_low = condition;
    condition_ave += condition;

    if( jacobian > jacobian_high )
        jacobian_high = jacobian;
    else if( jacobian < jacobian_low )
        jacobian_low = jacobian;
    jacobian_ave += jacobian;

    if( scaled_jacobian > scaled_jacobian_high )
        scaled_jacobian_high = scaled_jacobian;
    else if( scaled_jacobian < scaled_jacobian_low )
        scaled_jacobian_low = scaled_jacobian;
    scaled_jacobian_ave += scaled_jacobian;

    if( shape > shape_high )
        shape_high = shape;
    else if( shape < shape_low )
        shape_low = shape;
    shape_ave += shape;

    if( shape_size > shape_size_high )
        shape_size_high = shape_size;
    else if( shape_size < shape_size_low )
        shape_size_low = shape_size;
    shape_size_ave += shape_size;

    if( distortion > distortion_high )
        distortion_high = distortion;
    else if( distortion < distortion_low )
        distortion_low = distortion;
    distortion_ave += distortion;
 

    typename FIELD::mesh_type::Elem::index_type elem_id = *bi;
//     if( shape == 0.0 )
//         cout << "WARNING: Tet " << elem_id << " has negative volume!" << endl;
    if( scaled_jacobian <= 0.0 )
    {
      inversions++;
      cout << "WARNING: Tet " << elem_id << " has negative volume!" << endl;
    }
    
    ofield->set_value(scaled_jacobian,*(bi));
    total_elements++;
    ++bi;
  }
  
  aspect_ave /= total_elements;
  aspect_gamma_ave /= total_elements;
  volume_ave /= total_elements;
  condition_ave /= total_elements;
  jacobian_ave /= total_elements;
  scaled_jacobian_ave /= total_elements;
  shape_ave /= total_elements;
  shape_size_ave /= total_elements;
  distortion_ave /= total_elements;

  typename FIELD::mesh_type::Node::size_type nodes;
  typename FIELD::mesh_type::Edge::size_type edges;
  typename FIELD::mesh_type::Face::size_type faces;
  typename FIELD::mesh_type::Cell::size_type tets;
  mesh->size( nodes );
  mesh->size( edges );
  mesh->size( faces );
  mesh->size( tets );
  int holes = (tets-faces+edges-nodes+2)/2;

//  cout << "Tets: " << tets << " Faces: " << faces << " Edges: " << edges << " Nodes: " << nodes << endl;

  cout << endl << "Number of Tet elements checked = " << total_elements;
  if( inversions != 0 )
      cout << " (" << inversions << " Tets have negative jacobians!)";
  cout << endl << "Euler characteristics for this mesh indicate " << holes << " holes in this block of elements." << endl << "    (Assumes a single contiguous block of elements.)" << endl;
  cout << "Element counts: Tets: " << tets << " Faces: " << faces << " Edges: " << edges << " Nodes: " << nodes << endl;
  cout << "Aspect Ratio: Low = " << aspect_low << ", Average = " << aspect_ave << ", High = " << aspect_high << endl;
  cout << "Aspect Ratio (gamma): Low = " << aspect_gamma_low << ", Average = " << aspect_gamma_ave << ", High = " << aspect_gamma_high << endl;
  cout << "Volume: Low = " << volume_low << ", Average = " << volume_ave << ", High = " << volume_high << endl;
  cout << "Condition: Low = " << condition_low << ", Average = " << condition_ave << ", High = " << condition_high << endl;
  cout << "Jacobian: Low = " << jacobian_low << ", Average = " << jacobian_ave << ", High = " << jacobian_high << endl;
  cout << "Scaled_Jacobian: Low = " << scaled_jacobian_low << ", Average = " << scaled_jacobian_ave << ", High = " << scaled_jacobian_high << endl;
  cout << "Shape: Low = " << shape_low << ", Average = " << shape_ave << ", High = " << shape_high << endl;
  cout << "Shape_Size: Low = " << shape_size_low << ", Average = " << shape_size_ave << ", High = " << shape_size_high << endl;
  cout << "Distortion: Low = " << distortion_low << ", Average = " << distortion_ave << ", High = " << distortion_high << endl;
 
  return output;
}


//For Hexes...
template <class FIELD>
class MeshQualityAlgoHex : public MeshQualityAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshQualityAlgoHex<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
//   enum hexMetrics {ASPECT, SKEW, TAPER, VOLUME, STRETCH,
//                    DIAGONALS, CHARDIM, CONDITION, JACOBIAN,
//                    NORM_JACOBIAN, SHEAR, SHAPE, RELSIZE, SHEAR_SIZE, SHAPE_SIZE, 
//                    DISTORTION, ALLMETRICS, ALGEBRAIC, ROBINSON, TRADITIONAL,
//                    NUM_HEX_METRICS};

//   const int metricBitFlags[NUM_HEX_METRICS] = 
//       { 
//           V_HEX_ASPECT,
//           V_HEX_SKEW,
//           V_HEX_TAPER,
//           V_HEX_VOLUME,
//           V_HEX_STRETCH,
//           V_HEX_DIAGONAL,
//           V_HEX_DIMENSION,
//           V_HEX_CONDITION,
//           V_HEX_JACOBIAN,
//           V_HEX_SCALED_JACOBIAN,
//           V_HEX_SHEAR,
//           V_HEX_SHAPE,
//           V_HEX_RELATIVE_SIZE_SQUARED,
//           V_HEX_SHEAR_AND_SIZE,
//           V_HEX_SHAPE_AND_SIZE,
//           V_HEX_DISTORTION,
//           V_HEX_ALL,
//           V_HEX_ALGEBRAIC,
//           V_HEX_ROBINSON,
//           V_HEX_TRADITIONAL
//       };
  
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());

  double node_pos[8][3];
  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);

    //perform Euler checks for topology errors...
    // 2-2g = -#hexes+#faces-#edges+#nodes
  mesh->synchronize( Mesh::ALL_ELEMENTS_E );

  typedef GenericField<typename FIELD::mesh_type, ConstantBasis<double>, vector<double> > out_fld;  
  out_fld *ofield = scinew out_fld( field->get_typed_mesh() );
  FieldHandle output = dynamic_cast<Field*>(ofield);
  if (output.get_rep() == 0)
  {
    return (fieldh); 
  }

  int total_elements = 0;
  double aspect_high = 0, aspect_low = 0, aspect_ave = 0;
  double skew_high = 0, skew_low = 0, skew_ave = 0;
  double taper_high = 0, taper_low = 0, taper_ave = 0;
  double volume_high = 0, volume_low = 0, volume_ave = 0;
  double stretch_high = 0, stretch_low = 0, stretch_ave = 0;
  double diagonal_high = 0, diagonal_low = 0, diagonal_ave = 0;
  double dimension_high = 0, dimension_low = 0, dimension_ave = 0;
  double condition_high = 0, condition_low = 0, condition_ave = 0;
  double jacobian_high = 0, jacobian_low = 0, jacobian_ave = 0;
  double scaled_jacobian_high = 0, scaled_jacobian_low = 0, scaled_jacobian_ave = 0;
  double shear_high = 0, shear_low = 0, shear_ave = 0;
  double shape_high = 0, shape_low = 0, shape_ave = 0;
  double shear_size_high = 0, shear_size_low = 0, shear_size_ave = 0;
  double shape_size_high = 0, shape_size_low = 0, shape_size_ave = 0;
  double distortion_high = 0, distortion_low = 0, distortion_ave = 0;
  
  int inversions = 0;
  int first_time_thru = 1;
  
  while (bi != ei)
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    mesh->get_nodes(onodes, *bi);

    int i;
    for( i = 0; i < 8; i++ )
    {
      Point p;
      mesh->get_center( p, onodes[i] );
      node_pos[i][0] = p.x(); 
      node_pos[i][1] = p.y(); 
      node_pos[i][2] = p.z(); 
    }
    
    HexMetricVals values;
//    int verdict_metric = metricBitFlags[V_HEX_ALL];
    int verdict_metric = V_HEX_ALL;
    
    v_hex_quality(8, node_pos, verdict_metric, &values);

    double aspect = values.aspect;
    double skew = values.skew;
    double taper = values.taper;
    double volume = values.volume;
    double stretch = values.stretch;
    double diagonal = values.diagonal;
    double dimension = values.dimension;
    double condition = values.condition;
    double jacobian = values.jacobian;
    double scaled_jacobian = values.scaled_jacobian;
    double shear = values.shear;
    double shape = values.shape;
    double shear_size = values.shear_and_size;
    double shape_size = values.shape_and_size;
    double distortion = values.distortion;

    if( first_time_thru )
    {
      aspect_high = aspect;
      aspect_low = aspect;
      skew_high = skew;
      skew_low = skew;
      taper_high = taper;
      taper_low = taper;
      volume_high = volume;
      volume_low = volume;
      stretch_high = stretch;
      stretch_low = stretch;
      diagonal_high = diagonal;
      diagonal_low = diagonal;
      condition_high = condition;
      condition_low = condition;
      jacobian_high = jacobian;
      jacobian_low = jacobian;
      scaled_jacobian_high = scaled_jacobian;
      scaled_jacobian_low = scaled_jacobian;
      shear_high = shear;
      shear_low = shear;
      shape_high = shape;
      shape_low = shape;
      shape_size_high = shape_size;
      shape_size_low = shape_size;
      shear_size_high = shear_size;
      shear_size_low = shear_size;
      distortion_high = distortion;
      distortion_low = distortion;
      first_time_thru = 0;
    }

    if( aspect > aspect_high )
        aspect_high = aspect;
    else if( aspect < aspect_low )
        aspect_low = aspect;
    aspect_ave += aspect;

    if( skew > skew_high )
        skew_high = skew;
    else if( skew < skew_low )
        skew_low = skew;
    skew_ave += skew;
    
    if( taper > taper_high )
        taper_high = taper;
    else if( taper < taper_low )
        taper_low = taper;
    taper_ave += taper;
    
    if( volume > volume_high )
        volume_high = volume;
    else if( volume < volume_low )
        volume_low = volume;
    volume_ave += volume;

    if( stretch > stretch_high )
        stretch_high = stretch;
    else if( stretch < stretch_low )
        stretch_low = stretch;
    stretch_ave += stretch;

    if( diagonal > diagonal_high )
        diagonal_high = diagonal;
    else if( diagonal < diagonal_low )
        diagonal_low = diagonal;
    diagonal_ave += diagonal;

    if( dimension > dimension_high )
        dimension_high = dimension;
    else if( dimension < dimension_low )
        dimension_low = dimension;
    dimension_ave += dimension;

    if( condition > condition_high )
        condition_high = condition;
    else if( condition < condition_low )
        condition_low = condition;
    condition_ave += condition;

    if( jacobian > jacobian_high )
        jacobian_high = jacobian;
    else if( jacobian < jacobian_low )
        jacobian_low = jacobian;
    jacobian_ave += jacobian;

    if( scaled_jacobian > scaled_jacobian_high )
        scaled_jacobian_high = scaled_jacobian;
    else if( scaled_jacobian < scaled_jacobian_low )
        scaled_jacobian_low = scaled_jacobian;
    scaled_jacobian_ave += scaled_jacobian;

    if( shear > shear_high )
        shear_high = shear;
    else if( shear < shear_low )
        shear_low = shear;
    shear_ave += shear;

    if( shape > shape_high )
        shape_high = shape;
    else if( shape < shape_low )
        shape_low = shape;
    shape_ave += shape;

    if( shear_size > shear_size_high )
        shear_size_high = shear_size;
    else if( shear_size < shear_size_low )
        shear_size_low = shear_size;
    shear_size_ave += shear_size;

    if( shape_size > shape_size_high )
        shape_size_high = shape_size;
    else if( shape_size < shape_size_low )
        shape_size_low = shape_size;
    shape_size_ave += shape_size;

    if( distortion > distortion_high )
        distortion_high = distortion;
    else if( distortion < distortion_low )
        distortion_low = distortion;
    distortion_ave += distortion;

    typename FIELD::mesh_type::Elem::index_type elem_id = *bi;
//     if( shape == 0.0 )
//         cout << "WARNING: Hex " << elem_id << " has negative volume!" << endl;
//     if( jacobian <= 0.0 )
//        cout << "WARNING: Hex " << elem_id << " has negative volume!" << endl;
    if( scaled_jacobian <= 0.0 )
    {
      inversions++;
      cout << "WARNING: Hex " << elem_id << " has negative volume!" << endl;
    }

    ofield->set_value(scaled_jacobian,*(bi));
    total_elements++;
    ++bi;
  }
  
  aspect_ave /= total_elements;
  skew_ave /= total_elements;
  taper_ave /= total_elements;
  volume_ave /= total_elements;
  stretch_ave /= total_elements;
  diagonal_ave /= total_elements;
  dimension_ave /= total_elements;
  condition_ave /= total_elements;
  jacobian_ave /= total_elements;
  scaled_jacobian_ave /= total_elements;
  shear_ave /= total_elements;
  shape_ave /= total_elements;
  shear_size_ave /= total_elements;
  shape_size_ave /= total_elements;
  distortion_ave /= total_elements;

  typename FIELD::mesh_type::Node::size_type nodes;
  typename FIELD::mesh_type::Edge::size_type edges;
  typename FIELD::mesh_type::Face::size_type faces;
  typename FIELD::mesh_type::Cell::size_type hexes;
  mesh->size( nodes );
  mesh->size( edges );
  mesh->size( faces );
  mesh->size( hexes );
  signed int holes = (hexes-faces+edges-nodes+2)/2;
  
  cout << endl << "Number of Hex elements checked = " << total_elements;
  if( inversions != 0 )
      cout << " (" << inversions << " Hexes have negative jacobians!)";
  cout << endl << "Euler characteristics for this mesh indicate " << holes << " holes in this block of elements." << endl << "    (Assumes a single contiguous block of elements.)" << endl;
  cout << "Number of Elements = Hexes: " << hexes << " Faces: " << faces << " Edges: " << edges << " Nodes: " << nodes << endl;
  cout << "Aspect Ratio: Low = " << aspect_low << ", Average = " << aspect_ave << ", High = " << aspect_high << endl;
  cout << "Skew: Low = " << skew_low << ", Average = " << skew_ave << ", High = " << skew_high << endl;
  cout << "Taper: Low = " << taper_low << ", Average = " << taper_ave << ", High = " << taper_high << endl;
  cout << "Volume: Low = " << volume_low << ", Average = " << volume_ave << ", High = " << volume_high << endl;
  cout << "Stretch: Low = " << stretch_low << ", Average = " << stretch_ave << ", High = " << stretch_high << endl;
  cout << "Diagonal: Low = " << diagonal_low << ", Average = " << diagonal_ave << ", High = " << diagonal_high << endl;
  cout << "Dimension: Low = " << dimension_low << ", Average = " << dimension_ave << ", High = " << dimension_high << endl;
  cout << "Condition: Low = " << condition_low << ", Average = " << condition_ave << ", High = " << condition_high << endl;
  cout << "Jacobian: Low = " << jacobian_low << ", Average = " << jacobian_ave << ", High = " << jacobian_high << endl;
  cout << "Scaled_Jacobian: Low = " << scaled_jacobian_low << ", Average = " << scaled_jacobian_ave << ", High = " << scaled_jacobian_high << endl;
  cout << "Shear: Low = " << shear_low << ", Average = " << shear_ave << ", High = " << shear_high << endl;
  cout << "Shape: Low = " << shape_low << ", Average = " << shape_ave << ", High = " << shape_high << endl;
  cout << "Shear_Size: Low = " << shear_size_low << ", Average = " << shear_size_ave << ", High = " << shear_size_high << endl;
  cout << "Shape_Size: Low = " << shape_size_low << ", Average = " << shape_size_ave << ", High = " << shape_size_high << endl;
  cout << "Distortion: Low = " << distortion_low << ", Average = " << distortion_ave << ", High = " << distortion_high << endl;
 
  return output;
}


//For Tris...
template <class FIELD>
class MeshQualityAlgoTri : public MeshQualityAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshQualityAlgoTri<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
//   enum triMetrics {AREA, ANGLE, MIN_ANGLE, CONDITION_NUMBER, 
//                    MIN_SC_JAC, REL_SIZE, SHAPE, SHAPE_SIZE, DISTORTION,
//                    ALLMETRICS, ALGEBRAIC, TRADITIONAL,  NUM_TRI_METRICS};
// // Note: if you want to add a new metric, do it immediately before 
// //       "ALLMETRICS" so that grouping capability is not broken
  
//   const int metricBitFlags[NUM_TRI_METRICS] = 
//       {
//           V_TRI_AREA,
//           V_TRI_MAXIMUM_ANGLE,
//           V_TRI_MINIMUM_ANGLE,
//           V_TRI_CONDITION,
//           V_TRI_SCALED_JACOBIAN,
//           V_TRI_RELATIVE_SIZE_SQUARED,
//           V_TRI_SHAPE,
//           V_TRI_SHAPE_AND_SIZE,
//           V_TRI_DISTORTION,
//           V_TRI_ALL,
//           V_TRI_ALGEBRAIC,
//           V_TRI_TRADITIONAL
//       };
  
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());

  double node_pos[3][3];
  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);

  mesh->synchronize( Mesh::EDGES_E );

  typedef GenericField<typename FIELD::mesh_type, ConstantBasis<double>, vector<double> > out_fld;  
  out_fld *ofield = scinew out_fld( field->get_typed_mesh() );
  FieldHandle output = dynamic_cast<Field*>(ofield);
  if (output.get_rep() == 0)
  {
    return (fieldh); 
  }

  int total_elements = 0;
  double area_high = 0, area_low = 0, area_ave = 0;
  double minimum_angle_high = 0, minimum_angle_low = 0, minimum_angle_ave = 0;
  double maximum_angle_high = 0, maximum_angle_low = 0, maximum_angle_ave = 0;
  double condition_high = 0, condition_low = 0, condition_ave = 0;
//  double jacobian_high = 0, jacobian_low = 0, jacobian_ave = 0;
  double scaled_jacobian_high = 0, scaled_jacobian_low = 0, scaled_jacobian_ave = 0;
  double shape_high = 0, shape_low = 0, shape_ave = 0;
  double shape_size_high = 0, shape_size_low = 0, shape_size_ave = 0;
  double distortion_high = 0, distortion_low = 0, distortion_ave = 0;
  
  int inversions = 0;
  int first_time_thru = 1;
  
  while (bi != ei)
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    mesh->get_nodes(onodes, *bi);

    int i;
    for( i = 0; i < 3; i++ )
    {
      Point p;
      mesh->get_center( p, onodes[i] );
      node_pos[i][0] = p.x(); 
      node_pos[i][1] = p.y(); 
      node_pos[i][2] = p.z(); 
    }

// #if VERDICT_VERSION >=112
//       //if verdict version is greater than 112, we want to provide
//       //a way for verdict to calculate the surface normal at the
//       //center of this triangle.
//     MRefEntity* owning_entity = tri->owner();
//     owningFace = CAST_TO(owning_entity, MRefFace);
//     if(owningFace == NULL)
//         PRINT_WARNING("Cubit will not be able to determine whether this element is inverted.\n");
  
//     v_set_tri_normal_func( (ComputeNormal)&tri_normal_function );
      
//       //reset these to avoid stale pointers.
//     if( owningFace == NULL )
//         v_set_tri_normal_func( NULL );
// #endif  
  
    TriMetricVals values;
//    int verdict_metric = metricBitFlags[V_TRI_ALL];
    int verdict_metric = V_TRI_ALL;
    v_tri_quality(3, node_pos, verdict_metric, &values);
    
    double area = values.area;
    double minimum_angle = values.minimum_angle;
    double maximum_angle = values.maximum_angle;
    double condition = values.condition;
//    double jacobian = values.jacobian;
    double scaled_jacobian = values.scaled_jacobian;
    double shape = values.shape;
    double shape_size = values.shape_and_size;
    double distortion = values.distortion;

    if( first_time_thru )
    {
      area_high = area;
      area_low = area;
      minimum_angle_high = minimum_angle;
      minimum_angle_low = minimum_angle;
      maximum_angle_high = maximum_angle;
      maximum_angle_low = maximum_angle;
      condition_high = condition;
      condition_low = condition;
//      jacobian_high = jacobian;
//      jacobian_low = jacobian;
      scaled_jacobian_high = scaled_jacobian;
      scaled_jacobian_low = scaled_jacobian;
      shape_high = shape;
      shape_low = shape;
      shape_size_high = shape_size;
      shape_size_low = shape_size;
      distortion_high = distortion;
      distortion_low = distortion;
      first_time_thru = 0;
    }

    if( area > area_high )
        area_high = area;
    else if( area < area_low )
        area_low = area;
    area_ave += area;

    if( minimum_angle > minimum_angle_high )
        minimum_angle_high = minimum_angle;
    else if( minimum_angle < minimum_angle_low )
        minimum_angle_low = minimum_angle;
    minimum_angle_ave += minimum_angle;
    
    if( maximum_angle > maximum_angle_high )
        maximum_angle_high = maximum_angle;
    else if( maximum_angle < maximum_angle_low )
        maximum_angle_low = maximum_angle;
    maximum_angle_ave += maximum_angle;
    
    if( condition > condition_high )
        condition_high = condition;
    else if( condition < condition_low )
        condition_low = condition;
    condition_ave += condition;

//     if( jacobian > jacobian_high )
//         jacobian_high = jacobian;
//     else if( jacobian < jacobian_low )
//         jacobian_low = jacobian;
//     jacobian_ave += jacobian;

    if( scaled_jacobian > scaled_jacobian_high )
        scaled_jacobian_high = scaled_jacobian;
    else if( scaled_jacobian < scaled_jacobian_low )
        scaled_jacobian_low = scaled_jacobian;
    scaled_jacobian_ave += scaled_jacobian;

    if( shape > shape_high )
        shape_high = shape;
    else if( shape < shape_low )
        shape_low = shape;
    shape_ave += shape;

    if( shape_size > shape_size_high )
        shape_size_high = shape_size;
    else if( shape_size < shape_size_low )
        shape_size_low = shape_size;
    shape_size_ave += shape_size;

    if( distortion > distortion_high )
        distortion_high = distortion;
    else if( distortion < distortion_low )
        distortion_low = distortion;
    distortion_ave += distortion;

    typename FIELD::mesh_type::Elem::index_type elem_id = *bi;
//     if( shape == 0.0 )
//         cout << "WARNING: Tri " << elem_id << " has negative area!" << endl;
    if( scaled_jacobian <= 0.0 )
    {
      inversions++;
      cout << "WARNING: Tri " << elem_id << " has negative area!" << endl;
    }

    ofield->set_value(scaled_jacobian,*(bi));
    total_elements++;
    ++bi;
  }
  
  area_ave /= total_elements;
  minimum_angle_ave /= total_elements;
  maximum_angle_ave /= total_elements;
  condition_ave /= total_elements;
//  jacobian_ave /= total_elements;
  scaled_jacobian_ave /= total_elements;
  shape_ave /= total_elements;
  shape_size_ave /= total_elements;
  distortion_ave /= total_elements;

  typename FIELD::mesh_type::Node::size_type nodes;
  typename FIELD::mesh_type::Edge::size_type edges;
  typename FIELD::mesh_type::Face::size_type faces;
  mesh->size( nodes );
  mesh->size( edges );
  mesh->size( faces );
  int holes = (faces-edges+nodes-2)/2;
//  

  cout << endl << "Number of Tri elements checked = " << total_elements;
  if( inversions != 0 )
      cout << " (" << inversions << " Tris have negative jacobians!)";
  cout << endl << "Euler characteristics for this mesh indicate " << holes << " holes in this block of elements." << endl << "    (Assumes a single contiguous block of elements.)" << endl;
  cout << " Tris: " << faces << " Edges: " << edges << " Nodes: " << nodes << endl;
  cout << "Area: Low = " << area_low << ", Average = " << area_ave << ", High = " << area_high << endl;
  cout << "Minimum_Angle: Low = " << minimum_angle_low << ", Average = " << minimum_angle_ave << ", High = " << minimum_angle_high << endl;
  cout << "Maximum_Angle: Low = " << maximum_angle_low << ", Average = " << maximum_angle_ave << ", High = " << maximum_angle_high << endl;
  cout << "Condition: Low = " << condition_low << ", Average = " << condition_ave << ", High = " << condition_high << endl;
//  cout << "Jacobian: Low = " << jacobian_low << ", Average = " << jacobian_ave << ", High = " << jacobian_high << endl;
  cout << "Scaled_Jacobian: Low = " << scaled_jacobian_low << ", Average = " << scaled_jacobian_ave << ", High = " << scaled_jacobian_high << endl;
  cout << "Shape: Low = " << shape_low << ", Average = " << shape_ave << ", High = " << shape_high << endl;
  cout << "Shape_Size: Low = " << shape_size_low << ", Average = " << shape_size_ave << ", High = " << shape_size_high << endl;
  cout << "Distortion: Low = " << distortion_low << ", Average = " << distortion_ave << ", High = " << distortion_high << endl;

   return output;
}


//For Quads...
template <class FIELD>
class MeshQualityAlgoQuad : public MeshQualityAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshQualityAlgoQuad<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
//   enum quadMetrics {ASPECT, SKEW, TAPER, WARPAGE, AREA, STRETCH, 
//                     ANGLE, MIN_ANGLE, MAX_COND, MIN_JAC, MIN_SC_JAC, 
//                     SHEAR, SHAPE, RELSIZE, SHEAR_SIZE, SHAPE_SIZE, DISTORTION,
//                     ALLMETRICS, ALGEBRAIC, ROBINSON, TRADITIONAL, NUM_QUAD_METRICS};
// // Note: if you want to add a new metric, do it immediately before 
// //       "ALLMETRICS" so that grouping capability is not broken

//   const int metricBitFlags[NUM_QUAD_METRICS] =
//       {
//           V_QUAD_ASPECT,
//           V_QUAD_SKEW,
//           V_QUAD_TAPER,
//           V_QUAD_WARPAGE,
//           V_QUAD_AREA,
//           V_QUAD_STRETCH,
//           V_QUAD_MAXIMUM_ANGLE,
//           V_QUAD_MINIMUM_ANGLE,
//           V_QUAD_CONDITION,
//           V_QUAD_JACOBIAN,
//           V_QUAD_SCALED_JACOBIAN,
//           V_QUAD_SHEAR,
//           V_QUAD_SHAPE,
//           V_QUAD_RELATIVE_SIZE_SQUARED,
//           V_QUAD_SHEAR_AND_SIZE,
//           V_QUAD_SHAPE_AND_SIZE,
//           V_QUAD_DISTORTION,
//           V_QUAD_ALL,
//           V_QUAD_ALGEBRAIC,
//           V_QUAD_ROBINSON,
//           V_QUAD_TRADITIONAL
//       };
  
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh = dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());

  double node_pos[4][3];
  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);

  mesh->synchronize( Mesh::EDGES_E );
  
   typedef GenericField<typename FIELD::mesh_type, ConstantBasis<double>, vector<double> > out_fld;  
  out_fld *ofield = scinew out_fld( field->get_typed_mesh() );
  FieldHandle output = dynamic_cast<Field*>(ofield);
  if (output.get_rep() == 0)
  {
    return (fieldh); 
  }

  int total_elements = 0;
  double aspect_high = 0, aspect_low = 0, aspect_ave = 0;
  double skew_high = 0, skew_low = 0, skew_ave = 0;
  double taper_high = 0, taper_low = 0, taper_ave = 0;
  double warpage_high = 0, warpage_low = 0, warpage_ave = 0;
  double area_high = 0, area_low = 0, area_ave = 0;
  double stretch_high = 0, stretch_low = 0, stretch_ave = 0;
  double minimum_angle_high = 0, minimum_angle_low = 0, minimum_angle_ave = 0;
  double maximum_angle_high = 0, maximum_angle_low = 0, maximum_angle_ave = 0;
  double condition_high = 0, condition_low = 0, condition_ave = 0;
  double jacobian_high = 0, jacobian_low = 0, jacobian_ave = 0;
  double scaled_jacobian_high = 0, scaled_jacobian_low = 0, scaled_jacobian_ave = 0;
  double shear_high = 0, shear_low = 0, shear_ave = 0;
  double shape_high = 0, shape_low = 0, shape_ave = 0;
  double shear_size_high = 0, shear_size_low = 0, shear_size_ave = 0;
  double shape_size_high = 0, shape_size_low = 0, shape_size_ave = 0;
  double distortion_high = 0, distortion_low = 0, distortion_ave = 0;
  
  int inversions = 0;
  int first_time_thru;
  
  while (bi != ei)
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    mesh->get_nodes(onodes, *bi);

    int i;
    for( i = 0; i < 4; i++ )
    {
      Point p;
      mesh->get_center( p, onodes[i] );
      node_pos[i][0] = p.x(); 
      node_pos[i][1] = p.y(); 
      node_pos[i][2] = p.z(); 
    }
    
    QuadMetricVals values;
//    int verdict_metric = metricBitFlags[V_QUAD_ALL];
    int verdict_metric = V_QUAD_ALL;
    v_quad_quality(4, node_pos, verdict_metric, &values);
    
    double aspect = values.aspect;
    double skew = values.skew;
    double taper = values.taper;
    double warpage = values.warpage;
    double area = values.area;
    double stretch = values.stretch;
    double minimum_angle = values.minimum_angle;
    double maximum_angle = values.maximum_angle;
    double condition = values.condition;
    double jacobian = values.jacobian;
    double scaled_jacobian = values.scaled_jacobian;
    double shear = values.shear;
    double shape = values.shape;
    double shear_size = values.shear_and_size;
    double shape_size = values.shape_and_size;
    double distortion = values.distortion;

    if( first_time_thru )
    {
      aspect_high = aspect;
      aspect_low = aspect;
      skew_high = skew;
      skew_low = skew;
      taper_high = taper;
      taper_low = taper;
      warpage_high = warpage;
      warpage_low = warpage;
      area_high = area;
      area_low = area;
      stretch_high = stretch;
      stretch_low = stretch;
      minimum_angle_high = minimum_angle;
      minimum_angle_low = minimum_angle;
      maximum_angle_high = maximum_angle;
      maximum_angle_low = maximum_angle;
      condition_high = condition;
      condition_low = condition;
      jacobian_high = jacobian;
      jacobian_low = jacobian;
      scaled_jacobian_high = scaled_jacobian;
      scaled_jacobian_low = scaled_jacobian;
      shear_high = shear;
      shear_low = shear;
      shape_high = shape;
      shape_low = shape;
      shape_size_high = shape_size;
      shape_size_low = shape_size;
      shear_size_high = shear_size;
      shear_size_low = shear_size;
      distortion_high = distortion;
      distortion_low = distortion;
      first_time_thru = 0;
    }

    if( aspect > aspect_high )
        aspect_high = aspect;
    else if( aspect < aspect_low )
        aspect_low = aspect;
    aspect_ave += aspect;

    if( skew > skew_high )
        skew_high = skew;
    else if( skew < skew_low )
        skew_low = skew;
    skew_ave += skew;
    
    if( taper > taper_high )
        taper_high = taper;
    else if( taper < taper_low )
        taper_low = taper;
    taper_ave += taper;
    
    if( warpage > warpage_high )
        warpage_high = warpage;
    else if( warpage < warpage_low )
        warpage_low = warpage;
    warpage_ave += warpage;

    if( area > area_high )
        area_high = area;
    else if( area < area_low )
        area_low = area;
    area_ave += area;

    if( stretch > stretch_high )
        stretch_high = stretch;
    else if( stretch < stretch_low )
        stretch_low = stretch;
    stretch_ave += stretch;

    if( minimum_angle > minimum_angle_high )
        minimum_angle_high = minimum_angle;
    else if( minimum_angle < minimum_angle_low )
        minimum_angle_low = minimum_angle;
    minimum_angle_ave += minimum_angle;

    if( maximum_angle > maximum_angle_high )
        maximum_angle_high = maximum_angle;
    else if( maximum_angle < maximum_angle_low )
        maximum_angle_low = maximum_angle;
    maximum_angle_ave += maximum_angle;

    if( condition > condition_high )
        condition_high = condition;
    else if( condition < condition_low )
        condition_low = condition;
    condition_ave += condition;

    if( jacobian > jacobian_high )
        jacobian_high = jacobian;
    else if( jacobian < jacobian_low )
        jacobian_low = jacobian;
    jacobian_ave += jacobian;

    if( scaled_jacobian > scaled_jacobian_high )
        scaled_jacobian_high = scaled_jacobian;
    else if( scaled_jacobian < scaled_jacobian_low )
        scaled_jacobian_low = scaled_jacobian;
    scaled_jacobian_ave += scaled_jacobian;

    if( shear > shear_high )
        shear_high = shear;
    else if( shear < shear_low )
        shear_low = shear;
    shear_ave += shear;

    if( shape > shape_high )
        shape_high = shape;
    else if( shape < shape_low )
        shape_low = shape;
    shape_ave += shape;

    if( shear_size > shear_size_high )
        shear_size_high = shear_size;
    else if( shear_size < shear_size_low )
        shear_size_low = shear_size;
    shear_size_ave += shear_size;

    if( shape_size > shape_size_high )
        shape_size_high = shape_size;
    else if( shape_size < shape_size_low )
        shape_size_low = shape_size;
    shape_size_ave += shape_size;

    if( distortion > distortion_high )
        distortion_high = distortion;
    else if( distortion < distortion_low )
        distortion_low = distortion;
    distortion_ave += distortion;

    typename FIELD::mesh_type::Elem::index_type elem_id = *bi;
//     if( shape == 0.0 )
//         cout << "WARNING: Quad " << elem_id << " has negative area!" << endl;
//     if( area <= 0.0 )
//        cout << "WARNING: Quad " << elem_id << " has negative area!" << endl;
    if( scaled_jacobian <= 0.0 )
    {
      inversions++;
      cout << "WARNING: Quad " << elem_id << " has negative area!" << endl;
    }

    ofield->set_value(scaled_jacobian,*(bi));
    total_elements++;
    ++bi;
  }
  
  aspect_ave /= total_elements;
  skew_ave /= total_elements;
  taper_ave /= total_elements;
  warpage_ave /= total_elements;
  area_ave /= total_elements;
  stretch_ave /= total_elements;
  minimum_angle_ave /= total_elements;
  maximum_angle_ave /= total_elements;
  condition_ave /= total_elements;
  jacobian_ave /= total_elements;
  scaled_jacobian_ave /= total_elements;
  shear_ave /= total_elements;
  shape_ave /= total_elements;
  shear_size_ave /= total_elements;
  shape_size_ave /= total_elements;
  distortion_ave /= total_elements;

  typename FIELD::mesh_type::Node::size_type nodes;
  typename FIELD::mesh_type::Edge::size_type edges;
  typename FIELD::mesh_type::Face::size_type faces;
  mesh->size( nodes );
  mesh->size( edges );
  mesh->size( faces );
  int holes = (faces-edges+nodes-2)/2;
//  cout << "Quads: " << faces << " Edges: " << edges << " Nodes: " << nodes << endl;

  cout << endl << "Number of Quad elements checked = " << total_elements;
  if( inversions != 0 )
      cout << " (" << inversions << " Quads have negative jacobians!)";
  cout << endl << "Euler characteristics for this mesh indicate " << holes << " holes in this block of elements." << endl << "    (Assumes a single contiguous block of elements.)" << endl;
  cout << "Quads: " << faces << " Edges: " << edges << " Nodes: " << nodes << endl;
  cout << "Aspect Ratio: Low = " << aspect_low << ", Average = " << aspect_ave << ", High = " << aspect_high << endl;
  cout << "Skew: Low = " << skew_low << ", Average = " << skew_ave << ", High = " << skew_high << endl;
  cout << "Taper: Low = " << taper_low << ", Average = " << taper_ave << ", High = " << taper_high << endl;
  cout << "Warpage: Low = " << warpage_low << ", Average = " << warpage_ave << ", High = " << warpage_high << endl;
  cout << "Area: Low = " << area_low << ", Average = " << area_ave << ", High = " << area_high << endl;
  cout << "Stretch: Low = " << stretch_low << ", Average = " << stretch_ave << ", High = " << stretch_high << endl;
  cout << "Minimum_Angle: Low = " << minimum_angle_low << ", Average = " << minimum_angle_ave << ", High = " << minimum_angle_high << endl;
  cout << "Maximum_Angle: Low = " << maximum_angle_low << ", Average = " << maximum_angle_ave << ", High = " << maximum_angle_high << endl;
  cout << "Condition: Low = " << condition_low << ", Average = " << condition_ave << ", High = " << condition_high << endl;
  cout << "Jacobian: Low = " << jacobian_low << ", Average = " << jacobian_ave << ", High = " << jacobian_high << endl;
  cout << "Scaled_Jacobian: Low = " << scaled_jacobian_low << ", Average = " << scaled_jacobian_ave << ", High = " << scaled_jacobian_high << endl;
  cout << "Shear: Low = " << shear_low << ", Average = " << shear_ave << ", High = " << shear_high << endl;
  cout << "Shape: Low = " << shape_low << ", Average = " << shape_ave << ", High = " << shape_high << endl;
  cout << "Shear_Size: Low = " << shear_size_low << ", Average = " << shear_size_ave << ", High = " << shear_size_high << endl;
  cout << "Shape_Size: Low = " << shape_size_low << ", Average = " << shape_size_ave << ", High = " << shape_size_high << endl;
  cout << "Distortion: Low = " << distortion_low << ", Average = " << distortion_ave << ", High = " << distortion_high << endl;

   return output;
}


} // end namespace SCIRun

#endif // MeshQuality_h
