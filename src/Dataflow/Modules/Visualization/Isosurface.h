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


#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class Isosurface : public Module {

public:
  Isosurface(GuiContext* ctx);
  virtual ~Isosurface();
  virtual void execute();

private:
  //! GUI variables
  GuiDouble  gui_iso_value_min_;
  GuiDouble  gui_iso_value_max_;
  GuiDouble  gui_iso_value_;
  GuiDouble  gui_iso_value_typed_;
  GuiInt     gui_iso_value_quantity_;
  GuiString  gui_iso_quantity_range_;
  GuiString  gui_iso_quantity_clusive_;
  GuiDouble  gui_iso_quantity_min_;
  GuiDouble  gui_iso_quantity_max_;
  GuiString  gui_iso_quantity_list_;
  GuiString  gui_iso_value_list_;
  GuiString  gui_iso_matrix_list_;
  GuiInt     gui_extract_from_new_field_;
  GuiInt     gui_use_algorithm_;
  GuiInt     gui_build_field_;
  GuiInt     gui_build_geom_;
  GuiInt     gui_np_;          
  GuiString  gui_active_isoval_selection_tab_;
  GuiString  gui_active_tab_; 
  //gui_update_type_ must be declared after gui_iso_value_max_ which is
  //traced in the tcl code. If gui_update_type_ is set to Auto having it
  //last will prevent the net from executing when it is instantiated.
  GuiString  gui_update_type_;

  GuiDouble  gui_color_r_;
  GuiDouble  gui_color_g_;
  GuiDouble  gui_color_b_;

  //! status variables
  vector< double > isovals_;

  double iso_value_min_;
  double iso_value_max_;

  int     use_algorithm_;
  int     build_field_;
  int     build_geom_;
  int     np_;          
  double  color_r_;
  double  color_g_;
  double  color_b_;

  int fGeneration_;
  int cmGeneration_;
  int mGeneration_;

  FieldHandle  fHandle_;
  MatrixHandle mHandle_;
  int          geomID_;

  bool error_;
};

class IsosurfaceAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(vector<FieldHandle>& fields) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd);
};


template< class IFIELD >
class IsosurfaceAlgoT : public IsosurfaceAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(vector<FieldHandle>& fields);
};


template< class IFIELD >
FieldHandle
IsosurfaceAlgoT<IFIELD>::execute(vector<FieldHandle>& fields)
{
  vector<IFIELD *> qfields(fields.size());
  for (unsigned int i=0; i < fields.size(); i++)
    qfields[i] = (IFIELD *)(fields[i].get_rep());
  
  return append_fields(qfields);
}

} // End namespace SCIRun


