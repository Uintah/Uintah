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
#include <Core/Util/TypeDescription.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>


namespace SCIRun {

class Isosurface : public Module {

  //! GUI variables
  GuiDouble  gui_iso_value_;
  GuiDouble  gui_iso_value_min_;
  GuiDouble  gui_iso_value_max_;
  GuiDouble  gui_iso_value_typed_;
  GuiInt     gui_iso_value_quantity_;
  GuiString  gui_iso_quantity_range_;
  GuiDouble  gui_iso_quantity_min_;
  GuiDouble  gui_iso_quantity_max_;
  GuiString  gui_iso_value_list_;
  GuiInt     gui_extract_from_new_field_;
  GuiInt     gui_use_algorithm_;
  GuiInt     gui_build_field_;
  GuiInt     gui_np_;          
  GuiString  gui_active_isoval_selection_tab_;
  GuiString  gui_active_tab_; //! for saving nets state
  GuiString  gui_update_type_; //! for saving nets state
  GuiDouble  gui_color_r_;
  GuiDouble  gui_color_g_;
  GuiDouble  gui_color_b_;

  //! status variables
  int        geom_id_;
  double     prev_min_;
  double     prev_max_;
  int        last_generation_;

  bool new_field(FieldHandle field);

public:
  Isosurface(GuiContext* ctx);
  virtual ~Isosurface();
  virtual void execute();
};

} // End namespace SCIRun


