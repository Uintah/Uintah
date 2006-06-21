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

#include <sgi_stl_warnings_off.h>
#include   <string>
#include   <vector>
#include <sgi_stl_warnings_on.h>

#include <sci_values.h>


namespace Uintah {

using std::vector;
using std::string;

using SCIRun::CompileInfoHandle;
using SCIRun::DynamicAlgoBase;
using SCIRun::FieldHandle;
using SCIRun::Point;
using SCIRun::Transform;

class CompareMMSAlgo : public DynamicAlgoBase
{
public:
  enum compare_field_type { PRESSURE, UVEL, VVEL, INVALID };
  
  virtual FieldHandle compare(FieldHandle fh, 
                              const vector<unsigned int>& dimensions,
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
                              const vector<unsigned int>& dimensions,
                              compare_field_type  field_type,
                              const string& field_name,
                              const double  field_time,
                              const int output_choice,
                              const double time);
};

template <class FIELD>
FieldHandle
CompareMMSAlgoT<FIELD>::compare(FieldHandle fh, 
                                const vector<unsigned int>& dimensions,
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
  LVMesh *outputMesh = scinew LVMesh(dimensions[0], 
                                     dimensions[1], 
                                     dimensions[2], 
                                     minb, maxb);

  Transform temp;
  
  mesh->get_canonical_transform( temp );
  outputMesh->transform( temp );


  LVField *lvf = scinew LVField(outputMesh);

  char field_info[128];
  sprintf( field_info, "Exact %s - %lf", field_name.c_str(), field_time );
  lvf->set_property( "varname", string(field_info), true );

  MMS * mms = new MMS1();

  bool   showDif = (output_choice == 2);

  // Indexing in SCIRun fields apparently starts from 0, thus start
  // from zero and subtract 1 from high index
  for( unsigned int xx = 0; xx < dimensions[0]-1; xx++ ) {
    for( unsigned int yy = 0; yy < dimensions[1]-1; yy++ ) {
      for( unsigned int zz = 0; zz < dimensions[2]-1; zz++ ) {
        typename LVMesh::Cell::index_type pos(outputMesh,xx,yy,zz);

        //WARNING: "grid index to physical position" conversion has been hardcoded here!
        double x_pos = -0.5 + (xx-0.5) * 1.0 / 50;
        double y_pos = -0.5 + (yy-0.5) * 1.0 / 50;

        double calculatedValue;
        string msg;

        switch( field_type ) {
        case PRESSURE:
          calculatedValue = mms->pressure( x_pos, y_pos, time );
          break;
        case UVEL:
          calculatedValue = mms->uVelocity( x_pos, y_pos, time );
          break;
        case VVEL:
          calculatedValue = mms->vVelocity( x_pos, y_pos, time );
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
          typename LVMesh::Cell::index_type inputMeshPos(mesh,xx,yy,zz);

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
