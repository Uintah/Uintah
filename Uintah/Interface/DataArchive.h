#ifndef UINTAH_HOMEBREW_DataArchive_H
#define UINTAH_HOMEBREW_DataArchive_H

#include <string>
#include <vector>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>


namespace Uintah {

class Region;
   
   /**************************************
     
     CLASS
       DataArchive
      
       Short Description...
      
     GENERAL INFORMATION
      
       DataArchive.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       DataArchive
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
class DataArchive {
public:
   DataArchive(const std::string& filebase);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~DataArchive();

    
   // GROUP:  Information Access
   //////////
   // However, we need a means of determining the names of existing
   // variables. We also need to determine the type of each variable.
   // Get a list of scalar or vector variable names and  
   // a list of corresponding data types
  void listVariables( std::vector< std::string>& names,
		      std::vector< const TypeDescription *>& type );
  void listTimesteps( std::vector<int>& index,
		      std::vector<double>& times );
  void listRegions( std::vector<const Region*> regions,
		    double time );

  GridP getGrid( double time );

   
#if 0
   //////////
   // Does a variable exist in a particular region?
   bool exists(const std::string&, const Region*, int) {
      return true;
   }
#endif
   
   //////////
   // how long does a particle live?  Not variable specific.
   void lifetime( double& min, double& max, particleIndex id);
   
   //////////
   // how long does a region live?  Not variable specific
   void lifetime( double& min, double& max, const Region* region);
   

   //////////
   // list the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void list( ParticleVariable< T >, const std::string& name, 
	      particleIndex idx,
	      double min, double max);
   
   //////////
   // list the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void list( ParticleVariable< T >, const std::string& name, 
	      const Region*, double time );
   
   //////////
   // list the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void list( NCVariable< T >, const std::string& name, 
	      const Region*, double time );


   //////////
   // list the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void list( NCVariable< T >, const std::string& name, 
	      const IntVector& index,
	      double min, double max);
   
#if 0
   //////////
   // similarly, we want to be able to track variable values in a particular
   // region cell over time.
   template<class T>
   void list( std::vector< T >, const std::string& name,  
	      const Region *,
	      IntVector i, const time& min, const time& max);
   
   //////////
   // In other cases we will have noticed something interesting and we
   // will want to access some small portion of a region.  We will need
   // to request some range of data in index space.
   template<class T> void get(T& data, const std::string& name,
			      const Region* region, cellIndex min, cellIndex max);
#endif

   
   
protected:
   DataArchive();
   
private:
   DataArchive(const DataArchive&);
   DataArchive& operator=(const DataArchive&);
   
   
};

} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/05/20 02:34:56  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.1  2000/05/18 16:01:30  sparker
// Add data archive interface
//
//

#endif

