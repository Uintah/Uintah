#ifndef Uintah_Package_Dataflow_Modules_Visualization_VariablePlotter_h
#define Uintah_Package_Dataflow_Modules_Visualization_VariablePlotter_h 1

#include <Dataflow/Network/Module.h>
#include <Core/Geometry/IntVector.h>
#include <Core/GuiInterface/GuiVar.h>
// needs to be included here so Array3 knows about swapbytes(Matrix3)
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/VariableCache.h>
#include <Packages/Uintah/Core/Grid/GridP.h>

namespace Uintah {

class Index_ID {
public:
  Index_ID(): id_(IntVector(0,0,0)), level_(0) {}
  
  IntVector id_;
  int       level_;
};
  
class VariablePlotter : public Module {
public:
  VariablePlotter(GuiContext* ctx);
  VariablePlotter(const string& name, GuiContext* ctx);
  virtual ~VariablePlotter();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);

protected:
  bool getGrid();
  int add_type(string &type_list,const TypeDescription *subtype);
  void setVars(GridP grid);
  void extract_data(string display_mode, string varname,
		    vector<string> mat_list, vector<string> type_list,
		    string index);
  void pick(); // synchronize user set index values with currentNode
  
  string currentNode_str();

  void initialize_ports();
  // return 0 for success 1 for grid_change 2 for error
  int initialize_grid();
  void update_tcl_window();
  
  ArchiveIPort* in_; // incoming data archive

  GuiInt var_orientation_; // whether node center or cell centered
  GuiInt nl_;      // number of levels in the scene
  GuiInt index_x_; // x index for the variable
  GuiInt index_y_; // y index for the variable
  GuiInt index_z_; // z index for the variable
  GuiInt index_l_; // index of the level for the variable
  GuiString curr_var_;
  
  Index_ID currentNode_;
  vector< string > names_;
  vector< double > times_;
  vector< const TypeDescription *> types_;
  double time_;
  int old_generation_;
  int old_timestep_;
  GridP grid_;
  DataArchiveHandle archive_;
  int numLevels_;

  VariableCache material_data_list_;
};

} // namespace Uintah

#endif // Uintah_Package_Dataflow_Modules_Visualization_VariablePlotter_h
