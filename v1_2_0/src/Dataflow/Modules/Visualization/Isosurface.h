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

/*
 *  Isosurface.cc:  
 *
 *   \authur Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *
 *   \date Feb 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Geom/Material.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Disclosure/DynamicLoader.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>


namespace SCIRun {
  class MarchingCubesAlg;
  class NoiseAlg;
  class SageAlg;
  class GeomObj;

class MinmaxFunctor {
public:
  virtual bool get( Field *, pair<double,double>& ) = 0;
};

template<class F>
class Minmax : public MinmaxFunctor {
public:
  virtual bool get( Field *field, pair<double,double> &p ) {
    F *f = dynamic_cast<F *>(field);
    if ( !f ) return false;
    return field_minmax( *f, p );
  }
};

class Isosurface : public Module {
  // Input Ports
  FieldIPort* infield;
  FieldIPort* incolorfield;
  ColorMapIPort* inColorMap;

  // Output Ports
  FieldOPort* osurf;
  GeometryOPort* ogeom;
  

  //! GUI variables
  GuiDouble  gui_iso_value;
  GuiInt     extract_from_new_field;
  GuiInt     use_algorithm;
  GuiInt     build_trisurf_;
  GuiInt     np_;          
  GuiString  active_tab_; //! for saving nets state
  GuiString  update_type_; //! for saving nets state

  double iso_value;
  FieldHandle field_;
  GeomObj *surface;
  FieldHandle colorfield;
  ColorMapHandle cmap;
  TriSurfMeshHandle trisurf_mesh_;

  //! status variables
  int init;
  int geom_id;
  double prev_min, prev_max;
  int last_generation;
  int build_trisurf;
  bool have_colorfield;
  bool have_ColorMap;

  //! Handles to the isosurfacing algorithms.
  DynamicAlgoHandle            mc_alg_;
  DynamicAlgoHandle            noise_alg_;
  DynamicAlgoHandle            sage_alg_;

  MaterialHandle               matl_;

public:
  Isosurface(const string& id);
  virtual ~Isosurface();
  virtual void execute();

  void new_field( FieldHandle & );
  void send_results();
};

} // End namespace SCIRun


