#ifndef Uintah_Package_Dataflow_Modules_Visualization_VariablePlotter_h
#define Uintah_Package_Dataflow_Modules_Visualization_VariablePlotter_h 1

#include <Dataflow/Network/Module.h>
#include <Core/Geometry/IntVector.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/VariableCache.h>
#include <Packages/Uintah/Core/Grid/GridP.h>

namespace Uintah {

class Index_ID {
public:
  Index_ID(): id(IntVector(0,0,0)), level(0) {}
  IntVector id;
  int level;
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
  
  ArchiveIPort* in; // incoming data archive

  GuiInt var_orientation; // whether node center or cell centered
  GuiInt nl;      // number of levels in the scene
  GuiInt index_x; // x index for the variable
  GuiInt index_y; // y index for the variable
  GuiInt index_z; // z index for the variable
  GuiInt index_l; // index of the level for the variable
  GuiString curr_var;
  
  Index_ID currentNode;
  vector< string > names;
  vector< double > times;
  vector< const TypeDescription *> types;
  double time;
  int old_generation;
  int old_timestep;
  GridP grid;
  DataArchive* archive;
  int numLevels;

  VariableCache material_data_list;
};

} // namespace Uintah

#endif // Uintah_Package_Dataflow_Modules_Visualization_VariablePlotter_h
