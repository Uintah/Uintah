/****************************************
CLASS
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
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Util/Timer.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/IntVector.h>
#include <string>
#include <vector>


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

protected:
  virtual void get_vars(vector< string >&,
		       vector< const TypeDescription *>&) = 0;
  void build_GUI_frame();
  void update_GUI(const string& var,
		 const string& varnames);
  double field_update();
  template <class T, class Var>
    void build_field(DataArchive& archive,
		      const LevelP& level,
		      IntVector& lo,
		      const string& varname,
		      int mat,
		      double time,
		      Var& v,
		      LatVolField<T>*& sfd);
  vector< double > times;
  int generation;
  int timestep;
  int material;
  int levelnum;
  GuiInt level_;
  GridP grid;
  ArchiveHandle  archiveH;
  LatVolMeshHandle mesh_handle_;
}; //class 


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
  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* thread_sema = scinew Semaphore( "extractor semaphore",
                                             max_workers);
//  WallClockTimer my_timer;
//  my_timer.start();
  Mutex lock("PatchtoData lock");
  
  double size = level->numPatches();
  int count = 0;
  
  for( Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); ++r){
    IntVector low, hi;
    Var v;
    int vartype;
    archive.query( v, varname, mat, *r, time);
    if( sfd->data_at() == Field::CELL){
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
//        PatchToFieldThread<Var, T> *ptft = 
//         scinew PatchToFieldThread<Var, T>(sfd, v, lo, min_i, max_i,//low, hi, 
//   					  thread_sema, lock); 
//        ptft->run(); 

//        cerr<<"low = "<<low<<", hi = "<<hi<<", min_i = "<<min_i 
//  	  <<", max_i = "<<max_i<<endl; 
  
      Thread *thrd = scinew Thread( 
        (scinew PatchToFieldThread<Var, T>(sfd, v, lo, min_i, max_i,// low, hi,
				      thread_sema, lock)),
	"patch_to_field_worker");
      thrd->detach();
    }
  }
  thread_sema->down(max_workers);
  if( thread_sema ) delete thread_sema;
//  timer.add( my_timer.time());
//  my_timer.stop();
}



} // End namespace Uintah



#endif
