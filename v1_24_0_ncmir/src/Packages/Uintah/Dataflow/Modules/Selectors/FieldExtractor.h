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

template <class T, class Var>
void
FieldExtractor::build_multi_level_field( DataArchive& archive, GridP grid,
					 string& var, Var& v, int mat,
					 int generation,  double time, 
					 int timestep, double dt,
					 int loc,
					 TypeDescription::Type type,
					 TypeDescription::Type subtype,
					 MRLatVolField<T>*& mrfield)

{
  vector<MultiResLevel<T>*> levelfields;
  for(int i = 0; i < grid->numLevels(); i++){
    LevelP level = grid->getLevel( i );
    vector<LockingHandle<LatVolField<T> > > patchfields;
    int count = 0;
    
    // At this point we should have a mimimal patch set in our grid
    // And we want to make a LatVolField for each patch
    for(Level::const_patchIterator r = level->patchesBegin();
	r != level->patchesEnd(); ++r){
      
      IntVector hi, low, range;
      low = (*r)->getLowIndex();
      hi = (*r)->getHighIndex();	

      // ***** This seems like a hack *****
      range = hi - low + IntVector(1,1,1); 
      // **********************************

      BBox pbox;
      pbox.extend((*r)->getBox().lower());
      pbox.extend((*r)->getBox().upper());
      
      cerr<<"before mesh update: range is "<<range.x()<<"x"<<
	range.y()<<"x"<< range.z()<<",  low index is "<<low<<
	"high index is "<<hi<<" , size is  "<<
	pbox.min()<<", "<<pbox.max()<<"\n";
      
      LatVolMeshHandle mh = 0;
      update_mesh_handle(level, hi, range, pbox, type, mh);
      LatVolField<T> *fd = 
	scinew LatVolField<T>( mh, loc );
      if( subtype == TypeDescription::Vector ) {
	set_vector_properties( fd, var, generation, timestep, low, dt, type);
      }	else if( subtype == TypeDescription::Matrix3 ){
	set_tensor_properties( fd, low, type);
      } else {
	set_scalar_properties( fd, var, time, low, type);
      }
      cerr<<"Field "<<count<<", level "<<i<<" ";
      build_patch_field(archive, (*r), low, var, mat, time, v, fd);
      patchfields.push_back( fd );
      count++;
    }
    cerr<<"Added "<<count<<" fields to level "<<i<<"\n";
    MultiResLevel<T> *mrlevel = 
      new MultiResLevel<T>( patchfields, i );
    levelfields.push_back(mrlevel);
  }
  //  	    MRLatVolField<double>* mrfield =
  mrfield =  new MRLatVolField<T>( levelfields );
}
  
template <class T, class Var>
void
FieldExtractor::build_patch_field(DataArchive& archive,
                                  const Patch* patch,
				  IntVector& lo,
				  const string& varname,
				  int mat,
				  double time,
				  Var& /*var*/,
				  LatVolField<T>*& sfd)
{
  // Initialize the data
  sfd->fdata().initialize(T(0));

  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* thread_sema = scinew Semaphore( "extractor semaphore",
                                             max_workers);
  Mutex lock("PatchtoData lock");
  int count = 0;
  list<const Patch*> oldPatches = new2OldPatchMap_[patch];
  for(list<const Patch*>::iterator r = oldPatches.begin();
      r != oldPatches.end(); ++r){
    IntVector low, hi;
    Var v;
    int vartype;
    archive.query( v, varname, mat, *r, time);
    if( sfd->basis_order() == 0){
      low = (*r)->getCellLowIndex();
      hi = (*r)->getCellHighIndex();
    } else if(sfd->get_property("vartype", vartype)){
      low = (*r)->getNodeLowIndex();
      switch (vartype) {
      case TypeDescription::SFCXVariable:
	hi = (*r)->getSFCXHighIndex();
	break;
      case TypeDescription::SFCYVariable:
	hi = (*r)->getSFCYHighIndex();
	break;
      case TypeDescription::SFCZVariable:
	hi = (*r)->getSFCZHighIndex();
	break;
      default:
	hi = (*r)->getNodeHighIndex();	
      } 
    } 

    IntVector range = hi - low;

    int z_min = low.z();
    int z_max = low.z() + hi.z() - low.z();
    int z_step, z, N = 0;
    if ((z_max - z_min) >= max_workers){
      // in case we have large patches we'll divide up the work 
      // for each patch, if the patches are small we'll divide the
      // work up by patch.
      int cs = 25000000;  
      int S = range.x() * range.y() * range.z() * sizeof(T);
      N = Min(Max(S/cs, 1), (max_workers-1));
    }
    N = Max(N,2);
    z_step = (z_max - z_min)/(N - 1);
    for(z = z_min ; z < z_max; z += z_step) {
      
      IntVector min_i(low.x(), low.y(), z);
      IntVector max_i(hi.x(), hi.y(), Min(z+z_step, z_max));
      thread_sema->down();
      PatchToFieldThread<Var, T>* fldthrd = 
	scinew PatchToFieldThread<Var, T>(sfd, v, lo, min_i, max_i,// low, hi,
					  thread_sema, lock);
      fldthrd->run();
    
//       Thread *thrd = scinew Thread( 
//         (scinew PatchToFieldThread<Var, T>(sfd, v, lo, min_i, max_i,// low, hi,
// 				      thread_sema, lock)),
// 	"patch_to_field_worker");
//       thrd->detach();
    }
    count++;
  }
  cerr<<"used "<<count<<" patches to fill field\n";
  thread_sema->down(max_workers);
  if( thread_sema ) delete thread_sema;
}

template <class T, class Var>
void FieldExtractor::build_field(DataArchive& archive,
                                  const LevelP& level,
                                  IntVector& lo,
                                  const string& varname,
                                  int mat,
                                  double time,
				 Var& /*var*/,
                                  LatVolField<T>*& sfd)
{
  // Initialize the data
  sfd->fdata().initialize(T(0));

  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* thread_sema = scinew Semaphore( "extractor semaphore",
                                             max_workers);
//  WallClockTimer my_timer;
//  my_timer.start();
  Mutex lock("PatchtoData lock");
  
//   double size = level->numPatches();
//   int count = 0;
  
  for( Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); ++r){
    IntVector low, hi;
    Var v;
    int vartype;
    archive.query( v, varname, mat, *r, time);
    if( sfd->basis_order() == 0){
      low = (*r)->getCellLowIndex();
      hi = (*r)->getCellHighIndex();
//       low = (*r)->getNodeLowIndex();
//       hi = (*r)->getNodeHighIndex() - IntVector(1,1,1);

//       cerr<<"v.getLowIndex() = "<<v.getLowIndex()<<"\n";
//       cerr<<"v.getHighIndex() = "<<v.getHighIndex()<<"\n";
//       cerr<<"getCellLowIndex() = "<< (*r)->getCellLowIndex()
// 	  <<"\n";
//       cerr<<"getCellHighIndex() = "<< (*r)->getCellHighIndex()
// 	  <<"\n";
//       cerr<<"getInteriorCellLowIndex() = "<< (*r)->getInteriorCellLowIndex()
// 	  <<"\n";
//       cerr<<"getInteriorCellHighIndex() = "<< (*r)->getInteriorCellHighIndex()
// 	  <<"\n";
//       cerr<<"getNodeLowIndex() = "<< (*r)->getNodeLowIndex()
// 	  <<"\n";
//       cerr<<"getNodeHighIndex() = "<< (*r)->getNodeHighIndex()
// 	  <<"\n";
//       cerr<<"getInteriorNodeLowIndex() = "<< (*r)->getInteriorNodeLowIndex()
// 	  <<"\n";
//       cerr<<"getInteriorNodeHighIndex() = "<< (*r)->getInteriorNodeHighIndex()
// 	  <<"\n\n";
    } else if(sfd->get_property("vartype", vartype)){
      low = (*r)->getNodeLowIndex();
      switch (vartype) {
      case TypeDescription::SFCXVariable:
	hi = (*r)->getSFCXHighIndex();
	break;
      case TypeDescription::SFCYVariable:
	hi = (*r)->getSFCYHighIndex();
	break;
      case TypeDescription::SFCZVariable:
	hi = (*r)->getSFCZHighIndex();
	break;
      default:
	hi = (*r)->getNodeHighIndex();	
      } 
    } 

    IntVector range = hi - low;

    int z_min = low.z();
    int z_max = low.z() + hi.z() - low.z();
    int z_step, z, N = 0;
    if ((z_max - z_min) >= max_workers){
      // in case we have large patches we'll divide up the work 
      // for each patch, if the patches are small we'll divide the
      // work up by patch.
      int cs = 25000000;  
      int S = range.x() * range.y() * range.z() * sizeof(T);
      N = Min(Max(S/cs, 1), (max_workers-1));
    }
    N = Max(N,2);
    z_step = (z_max - z_min)/(N - 1);
    for(z = z_min ; z < z_max; z += z_step) {
      
      IntVector min_i(low.x(), low.y(), z);
      IntVector max_i(hi.x(), hi.y(), Min(z+z_step, z_max));
//      update_progress((count++/double(N))/size, my_timer);
      
      thread_sema->down();
       PatchToFieldThread<Var, T> *ptft = 
        scinew PatchToFieldThread<Var, T>(sfd, v, lo, min_i, max_i,//low, hi, 
  					  thread_sema, lock); 
       ptft->run(); 

//        cerr<<"low = "<<low<<", hi = "<<hi<<", min_i = "<<min_i 
//  	  <<", max_i = "<<max_i<<endl; 
  
//       Thread *thrd = scinew Thread( 
//         (scinew PatchToFieldThread<Var, T>(sfd, v, lo, min_i, max_i,// low, hi,
// 				      thread_sema, lock)),
// 	"patch_to_field_worker");
//       thrd->detach();
    }
  }
  thread_sema->down(max_workers);
  if( thread_sema ) delete thread_sema;
//  timer.add( my_timer.time());
//  my_timer.stop();
}

template <class T>
void 
FieldExtractor::set_scalar_properties(LatVolField<T>*& sfd,
				      string& varname,
				      double time, IntVector& low,
				      TypeDescription::Type type)
{  
  sfd->set_property( "variable", string(varname), true );
  sfd->set_property( "time", double( time ), true);
  sfd->set_property( "offset", IntVector(low), true);
  sfd->set_property( "vartype", int(type),true);
}

template <class T>
void
FieldExtractor::set_vector_properties(LatVolField<T>*& vfd, string& var,
				      int generation, int timestep,
				      IntVector& low, double dt,
				      TypeDescription::Type type)
{
  vfd->set_property("varname",string(var), true);
  vfd->set_property("generation",generation, true);
  vfd->set_property("timestep",timestep, true);
  vfd->set_property( "offset", IntVector(low), true);
  vfd->set_property("delta_t",dt, true);
  vfd->set_property( "vartype", int(type),true);
}

template <class T>
void 
FieldExtractor::set_tensor_properties(LatVolField<T>*& tfd,  IntVector& low,
				      TypeDescription::Type type)
{
  tfd->set_property( "vartype",
		     int(TypeDescription::CCVariable),true);
  tfd->set_property( "offset", IntVector(low), true);
}



} // End namespace Uintah



#endif
