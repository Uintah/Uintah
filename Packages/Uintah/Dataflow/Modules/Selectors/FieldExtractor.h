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


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Dataflow/Modules/Selectors/PatchDataThread.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <Core/Util/Timer.h>
#include <string>
#include <vector>


namespace Uintah {
using namespace SCIRun;

class FieldExtractor : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  FieldExtractor(const string& name,
		 const string& id,
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
  double update();
  template <class T, class Var>
    void build_field(DataArchive& archive,
		    const LevelP& level,
		    const string& varname,
		    int mat,
		    double time,
		    const Var& v,
		    LevelField<T>*& sfd);

  vector< double > times;
  int generation;
  int timestep;
  int material;
  GridP grid;
  ArchiveHandle  archiveH;
}; //class 


template <class T, class Var>
void FieldExtractor::build_field(DataArchive& archive,
				      const LevelP& level,
				      const string& varname,
				      int mat,
				      double time,
				      const Var& v,
				      LevelField<T>*& sfd)
{
  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* thread_sema = scinew Semaphore( "extractor semahpore",
					     max_workers); 
  WallClockTimer my_timer;
  my_timer.start();

  vector<ShareAssignArray3<T> > &data = sfd->fdata();
  data.resize(level->numPatches());
  double size = data.size();
  int count = 0;
  vector<ShareAssignArray3<T> >::iterator it = data.begin();
  for(Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); r++, ++it){
    update_progress(count++/size, my_timer);
    thread_sema->down();
    Thread *thrd =
      scinew Thread(scinew PatchDataThread<Var,
		    vector<ShareAssignArray3<T> >::iterator>
		    (archive, it, varname, mat, *r, time, thread_sema),
		    "patch_data_worker");
    thrd->detach();
//     PatchDataThread<Var, vector<ShareAssignArray3<T> >::iterator> *pdt =
//       scinew PatchDataThread<Var, vector<ShareAssignArray3<T> >::iterator>
//       (archive, it, varname, mat, *r, time, thread_sema);
//     pdt->run();
  }
  thread_sema->down(max_workers);
  if( thread_sema ) delete thread_sema;
  timer.add( my_timer.time());
  my_timer.stop();
}



} // End namespace Uintah



#endif
