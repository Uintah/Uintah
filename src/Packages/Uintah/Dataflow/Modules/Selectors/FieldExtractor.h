/****************************************
Class
    FieldExtractor

    

OVERVIEW TEXT
    This module receives a DataArchive object.  The user
    interface is dynamically created based information provided by the
    DataArchive.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman, James Bigler
    Department of Computer Science
    University of Utah
    June, 2000

    Copyright (C) 2000 SCI Group

LOG
    Created June 27, 2000
****************************************/
#ifndef FIELDEXTRACTOR_H
#define FIELDEXTRACTOR_H 1


#include <Packages/Uintah/Dataflow/Modules/Selectors/PatchToField.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Datatypes/MRLatVolField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Util/Timer.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MinMax.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>


namespace Uintah {
using namespace SCIRun;

class FieldExtractor : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  FieldExtractor(const string& name,
                 GuiContext* ctx,
                 const string& cat="unknown",
                 const string& pack="unknown");
  // GROUP: Destructors
  //////////
  virtual ~FieldExtractor(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  // This class is used to store parameters passed up the chain to the
  // build field functions.  This makes it easier to pass a large
  // number of parameters.
  class QueryInfo {
  public:
    QueryInfo() {}
    QueryInfo(DataArchive* archive, int generation,
              GridP& grid, LevelP& level,
              string varname,
              int mat,
              const Uintah::TypeDescription *type,
              bool get_all_levels,
              double time, int timestep, double dt):
      archive(archive), generation(generation),
      grid(grid), level(level),
      varname(varname), mat(mat), type(type),
      get_all_levels(get_all_levels),
      time(time), timestep(timestep), dt(dt)
    {}
    
    DataArchive* archive;
    int generation;
    GridP grid;
    LevelP level;
    string varname;
    int mat;
    const Uintah::TypeDescription *type;
    bool get_all_levels;
    double time;
    int timestep;
    double dt;
  };

protected:
  virtual void get_vars(vector< string >&,
                        vector< const TypeDescription *>&) = 0;
  void build_GUI_frame();
  void update_GUI(const string& var,
                  const string& varnames);
  double field_update();
  bool is_periodic_bcs(IntVector cellir, IntVector ir);
  void get_periodic_bcs_range(IntVector cellir, IntVector ir,
                              IntVector range, IntVector& newir);

  // Sets all sorts of properties using the PropertyManager facility
  // of the Field.  This is called for all types of Fields.
  void set_field_properties(Field* field, QueryInfo& qinfo,
                            IntVector& offset);

  // Creates a MRLatVolField.
  template <class Var, class T>
  FieldHandle build_multi_level_field( QueryInfo& qinfo, int basis_order);


  // This does the actuall work of getting the data from the
  // DataArchive for a single patch and filling the field.  This is
  // called by both build_field and build_patch_field.
  template <class Var, class T>
  void getPatchData(QueryInfo& qinfo, IntVector& offset,
                    LatVolField<T>* sfield, const Patch* patch);
  
  // Similar to build_field, but is called from build_multi_level_field.
  template <class Var, class T>
  void build_patch_field(QueryInfo& qinfo,
                         const Patch* patch,
                         IntVector& offset,
                         LatVolField<T>* field);

  // Calls query for a single-level data set.
  template <class Var, class T>
  void build_field(QueryInfo& qinfo, IntVector& offset, LatVolField<T>* field);

  // This function makes a switch between building multi-level data or
  // single-level data.  Makes a call to either build_field or or
  // build_multi_level_field.  The basis_order pertains to whether the
  // data is node or cell centerd.  Type Var should look something
  // like CCVariable<T> or NCVariable<T>.
  template<class Var, class T>
  FieldHandle getData(QueryInfo& qinfo, IntVector& offset,
                      LatVolMeshHandle mesh_handle,
                      int basis_order);

  // This is the first function on your way to getting a field.  This
  // makes a template switch on the type of variable (CCVariable,
  // NCVariable, etc.).  It then calls getData.  The type of T is
  // double, int, Vector, etc.
  template<class T>
  FieldHandle getVariable(QueryInfo& qinfo, IntVector& offset,
                          LatVolMeshHandle mesh_handle);

  void update_mesh_handle( LevelP& level,
                           IntVector& hi,
                           IntVector& range,
                           BBox& box,
                           TypeDescription::Type type,
                           LatVolMeshHandle& mesh_handle);

  GridP build_minimal_patch_grid( GridP oldGrid );

  vector< double > times;
  int generation;
  int timestep;
  int material;
  int levelnum;
  GuiInt level_;
  GridP grid;
  ArchiveHandle  archiveH;
  LatVolMeshHandle mesh_handle_;

  GuiString tcl_status;

  GuiString sVar;
  GuiInt sMatNum;

  const TypeDescription *type;

  map<const Patch*, list<const Patch*> > new2OldPatchMap_;

}; //class 

} // End namespace Uintah



#endif
