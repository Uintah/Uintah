#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Components/Schedulers/DWDatabase.h>
#include <map>

namespace Uintah {
   class DataItem;
   class TypeDescription;
   class Region;
/**************************************

  CLASS
        OnDemandDataWarehouse
   
	Short description...

  GENERAL INFORMATION

        OnDemandDataWarehouse.h

	Steven G. Parker
	Department of Computer Science
	University of Utah

	Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
	Copyright (C) 2000 SCI Group

  KEYWORDS
        On_Demand_Data_Warehouse

  DESCRIPTION
        Long description...
  
  WARNING
  
****************************************/

class OnDemandDataWarehouse : public DataWarehouse {
public:
   OnDemandDataWarehouse( int MpiRank, int MpiProcesses );
   virtual ~OnDemandDataWarehouse();
   
   virtual void setGrid(const GridP&);
   
   virtual void allocate(ReductionVariableBase&, const VarLabel*);
   virtual void get(ReductionVariableBase&, const VarLabel*) const;
   virtual void put(const ReductionVariableBase&, const VarLabel*);
   virtual void allocate(int numParticles, ParticleVariableBase&,
			 const VarLabel*, int matlIndex, const Region*);
   virtual void allocate(ParticleVariableBase&, const VarLabel*,
			 int matlIndex, const Region*);
   virtual void get(ParticleVariableBase&, const VarLabel*,
		    int matlIndex, const Region*, int numGhostCells) const;
   virtual void put(const ParticleVariableBase&, const VarLabel*,
		    int matlIndex, const Region*);
   virtual void allocate(NCVariableBase&, const VarLabel*,
			 int matlIndex, const Region*);
   virtual void get(NCVariableBase&, const VarLabel*,
		    int matlIndex, const Region*, int numGhostCells) const;
   virtual void put(const NCVariableBase&, const VarLabel*,
		    int matlIndex, const Region*);
private:
   DWDatabase<NCVariableBase> ncdb;
   DWDatabase<ParticleVariableBase> particledb;
   struct ReductionRecord {
      ReductionVariableBase* var;
      ReductionRecord(ReductionVariableBase*);
   };
   typedef std::map<const VarLabel*, ReductionRecord*, VarLabel::Compare> reductionDBtype;
   reductionDBtype reductiondb;
   //////////
   // Insert Documentation Here:
   mutable SCICore::Thread::CrowdMonitor  d_lock;
   bool                                   d_allowCreation;
   GridP grid;
   
   const VarLabel* position_label;
};

} // end namespace Uintah

//
// $Log$
// Revision 1.13  2000/05/02 06:07:16  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.12  2000/04/28 07:35:34  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.11  2000/04/27 23:18:48  sparker
// Added problem initialization for MPM
//
// Revision 1.10  2000/04/26 06:48:33  sparker
// Streamlined namespaces
//
// Revision 1.9  2000/04/24 15:17:01  sparker
// Fixed unresolved symbols
//
// Revision 1.8  2000/04/20 18:56:26  sparker
// Updates to MPM
//
// Revision 1.7  2000/04/19 21:20:03  dav
// more MPI stuff
//
// Revision 1.6  2000/04/19 05:26:11  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/13 06:50:57  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/03/22 00:36:37  sparker
// Added new version of getRegionData
//
// Revision 1.3  2000/03/17 01:03:17  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//

#endif
