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
//    File   : CompareMMS.h
//    Author : Kurt Zimmerman
//    Date   : March 2004


#if !defined(COMPAREMMS_H)
#define COMPAREMMS_H


#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Datatype.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Dataflow/Network/Module.h>

#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS.h>
#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS1.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_off.h>
#include   <string>
#include   <vector>

#include <sgi_stl_warnings_on.h>

#include <sci_values.h>


namespace Uintah {

using std::vector;
using std::string;
using std::cout;

using SCIRun::CompileInfoHandle;
using SCIRun::DynamicAlgoBase;
using SCIRun::FieldHandle;
using SCIRun::Point;
using SCIRun::IntVector;
using SCIRun::Vector;
using SCIRun::Transform;

class CompareMMSAlgo : public DynamicAlgoBase
{
public:
  enum compare_field_type { PRESSURE, UVEL, VVEL, INVALID };
  
  virtual FieldHandle compare(FieldHandle fh, 
                              const vector<unsigned int>& nCells,
                              const IntVector extraCells,
                              const Point spatial_min,
                              const Point spatial_max,
                              compare_field_type  field_type,
                              const string& field_name,
                              const double  field_time,
                              const int output_choice,
                              const double time) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *td);
};


template <class FIELD>
class CompareMMSAlgoT : public CompareMMSAlgo
{
public:
  virtual FieldHandle compare(FieldHandle fh, 
                              const vector<unsigned int>& nCells,
                              const IntVector extraCells,
                              const Point spatial_min,
                              const Point spatial_max,
                              compare_field_type  field_type,
                              const string& field_name,
                              const double  field_time,
                              const int output_choice,
                              const double time);
};

template <class FIELD>
FieldHandle
CompareMMSAlgoT<FIELD>::compare(FieldHandle fh, 
                                const vector<unsigned int>& nCells,
                                const IntVector extraCells,
                                const Point spatial_min,
                                const Point spatial_max,
                                compare_field_type  field_type,
                                const string& field_name,
                                const double  field_time,
                                const int output_choice,
                                const double time)
{
  // We know that the data is arranged as a LatVolMesh of some type.
  typedef typename FIELD::mesh_type LVMesh;
  typedef typename FIELD::field_type LVField;

  Point minb, maxb;
    
  minb = Point(0,0,0);
  maxb = Point(1, 1, 1);

  // grab the current field and mesh
  LVField *field = (LVField *)fh.get_rep();
  LVMesh *mesh = field->get_typed_mesh().get_rep(); 
    
  // Create blank mesh. 
  LVMesh *outputMesh = scinew LVMesh(nCells[0], 
                                     nCells[1], 
                                     nCells[2], 
                                     minb, maxb);

  Transform temp;
  
  mesh->get_canonical_transform( temp );
  outputMesh->transform( temp );


  LVField *lvf = scinew LVField(outputMesh);

  char field_info[128];
  sprintf( field_info, "Exact %s - %lf", field_name.c_str(), field_time );
  lvf->set_property( "varname", string(field_info), true );

  MMS * mms = new MMS1();

  bool showDif = (output_choice == 2);

  // Find the number of interior cells and the cell size
  // nCells is an exclusive number
  // remove the ghostcells (2 ghost cells)
  IntVector nInteriorCells(nCells[0], nCells[1], nCells[2]);
  nInteriorCells = nInteriorCells - IntVector(1,1,1); 
  nInteriorCells = nInteriorCells + extraCells * IntVector(2,2,2);
  Vector dx = (spatial_max - spatial_min)/nInteriorCells;

  // Indexing in SCIRun fields starts from 0, thus start
  // from zero and subtract 1 from high index
  for( unsigned int i = 0; i < nCells[0]-1; i++ ) {
    for( unsigned int j = 0; j < nCells[1]-1; j++ ) {
      for( unsigned int k = 0; k < nCells[2]-1; k++ ) {
        typename LVMesh::Cell::index_type pos(outputMesh,i,j,k);
        IntVector c(i,j,k);
        c = c + extraCells;  // shift (c) my the extra cells
        
        double x_pos_CC = spatial_min.x() + c.x() * dx.x() + dx.x()/2.0; 
        double y_pos_CC = spatial_min.y() + c.y() * dx.y() + dx.y()/2.0;
        double z_pos_CC = spatial_min.z() + c.z() * dx.z() + dx.z()/2.0;
        
        // sanity output
        if(i == 0 && j == 0 && k == 1){
          cout  << " dx " << dx << " cell " << c 
                << " x_pos_CC " << x_pos_CC 
                << " y_pos_CC " << y_pos_CC 
                << " z_pos_CC " << z_pos_CC<< "\n";
        }
        
        double calculatedValue;
        string msg;

        switch( field_type ) {
        case PRESSURE:
          calculatedValue = mms->pressure( x_pos_CC, y_pos_CC, time );
          break;
        case UVEL:
          calculatedValue = mms->uVelocity( x_pos_CC, y_pos_CC, time );
          break;
        case VVEL:
          calculatedValue = mms->vVelocity( x_pos_CC, y_pos_CC, time );
          break;
        case INVALID:
          msg = "We should not reach this point anyway, but you have selected a variable that is usupported by MMS";
          printf("%s\n", msg.c_str());
          break;
        default:
          printf( "ERROR: CompareMMS.cc - Bad field_type %d\n", field_type );
          exit(1);
        }
        if( showDif ) {
          double val;
          typename LVMesh::Cell::index_type inputMeshPos(mesh,i,j,k);

          field->value( val, inputMeshPos ); // Get the value at pos

          lvf->set_value( calculatedValue - val, pos );
        } else {
          lvf->set_value( calculatedValue, pos );
        }
      }
    }
  } 
    
  return lvf;
}

} // end namespace Uintah

#endif
