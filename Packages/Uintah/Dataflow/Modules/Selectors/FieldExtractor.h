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
    Packages/Kurt Zimmerman
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


#define TEMPLATE_FUN 1

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

#ifdef TEMPLATE_FUN
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
#endif

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
  
#ifdef TEMPLATE_FUN
  void set_field_properties(Field* field, QueryInfo& qinfo,
                            IntVector& offset);

  template <class Var, class T>
  FieldHandle build_multi_level_field( QueryInfo& qinfo, int loc);


  template <class Var, class T>
  void getPatchData(QueryInfo& qinfo, IntVector& offset,
                    LatVolField<T>* sfield, const Patch* patch);
  
  // For help with build_multi_level_field
  template <class Var, class T>
  void build_patch_field(QueryInfo& qinfo,
                         const Patch* patch,
                         IntVector& offset,
                         LatVolField<T>* field);
  
  template <class Var, class T>
  void build_field(QueryInfo& qinfo, IntVector& offset, LatVolField<T>* field);

  template<class Var, class T>
  FieldHandle getData(QueryInfo& qinfo, IntVector& offset,
                      LatVolMeshHandle mesh_handle,
                      int basis_order);
  
  template<class T>
  FieldHandle getVariable(QueryInfo& qinfo, IntVector& offset,
                          LatVolMeshHandle mesh_handle);
#endif

  template <class T, class Var>
  void build_patch_field(DataArchive& archive,
                         const Patch* patch,
                         IntVector& lo,
                         const string& varname,
                         int mat,
                         double time,
                         Var& v,
                         LatVolField<T>*& sfd);
  template <class T, class Var>
  void build_field(DataArchive& archive,
                   const LevelP& level,
                   IntVector& lo,
                   const string& varname,
                   int mat,
                   double time,
                   Var& v,
                   LatVolField<T>*& sfd);

  template <class T, class Var>
  void
  build_multi_level_field( DataArchive& archive, GridP grid,
                           string& var, Var& v, int mat,
                           int generation, double time, int timestep,
                           double dt, int loc,
                           TypeDescription::Type type,
                           TypeDescription::Type subtype,
                           MRLatVolField<T>*& mrfield);
  template <class T>
  void set_scalar_properties(LatVolField<T>*& sfd, string& varname,
                             double time, IntVector& low,
                             TypeDescription::Type type);
  template <class T>
  void set_vector_properties(LatVolField<T>*& vfd, string& var,
                             int generation, int timestep,
                             IntVector& low, double dt,
                             TypeDescription::Type type);
  template <class T>
  void set_tensor_properties(LatVolField<T>*& tfd, 
                             IntVector& low,
                             TypeDescription::Type type);
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
