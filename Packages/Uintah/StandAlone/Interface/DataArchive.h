#ifndef UINTAH_HOMEBREW_DataArchive_H
#define UINTAH_HOMEBREW_DataArchive_H

#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/GridP.h>
#include <string>
#include <vector>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <util/XMLURL.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

namespace Uintah {
   class Patch;
   
   /**************************************
     
     CLASS
       DataArchive
      
       Short Description...
      
     GENERAL INFORMATION
      
       DataArchive.h
      
       Kurt Zimmerman
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
   void queryVariables( std::vector< std::string>& names,
		       std::vector< const TypeDescription *>&  );
   void queryTimesteps( std::vector<int>& index,
		       std::vector<double>& times );
   GridP queryGrid( double time );
   
#if 0
   //////////
   // Does a variable exist in a particular patch?
   bool exists(const std::string&, const Patch*, int) {
      return true;
   }
#endif
   
   //////////
   // how long does a particle live?  Not variable specific.
   void queryLifetime( double& min, double& max, particleId id);
   
   //////////
   // how long does a patch live?  Not variable specific
   void queryLifetime( double& min, double& max, const Patch* patch);

   int queryNumMaterials(const std::string& name, const Patch* patch,
			double time);

   //////////
   // query the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void query( ParticleVariable< T >&, const std::string& name, int matlIndex,
	      particleId id,
	      double min, double max);
   
   //////////
   // query the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void query( ParticleVariable< T >&, const std::string& name, int matlIndex,
	      const Patch*, double time );
   
   //////////
   // query the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void query( NCVariable< T >&, const std::string& name, int matlIndex,
	      const Patch*, double time );


   //////////
   // query the variable value for a particular particle  overtime;
   // T = double/float/vector/Tensor I'm not sure of the proper
   // syntax.
   template<class T>
   void query( NCVariable< T >&, const std::string& name, int matlIndex,
	      const IntVector& index,
	      double min, double max);
   
#if 0
   //////////
   // similarly, we want to be able to track variable values in a particular
   // patch cell over time.
   template<class T>
   void query( std::vector< T >, const std::string& name,  
	      const Patch *,
	      IntVector i, const time& min, const time& max);
   
   //////////
   // In other cases we will have noticed something interesting and we
   // will want to access some small portion of a patch.  We will need
   // to request some range of data in index space.
   template<class T> void get(T& data, const std::string& name,
			      const Patch* patch, cellIndex min, cellIndex max);
#endif

   
   
protected:
   DataArchive();
   
private:
   DataArchive(const DataArchive&);
   DataArchive& operator=(const DataArchive&);

   DOM_Node getTimestep(double time, XMLURL& url);
   
   std::string d_filebase;
   DOM_Document d_indexDoc;
   XMLURL d_base;

   bool have_timesteps;
   std::vector<int> d_tsindex;
   std::vector<double> d_tstimes;
   std::vector<DOM_Node> d_tstop;
   std::vector<XMLURL> d_tsurl;

   DOM_Node findVariable(const string& name, const Patch* patch,
			 int matl, double time, XMLURL& url);
};

} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/05/30 20:19:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.4  2000/05/21 08:19:11  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.3  2000/05/20 08:09:36  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.2  2000/05/20 02:34:56  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.1  2000/05/18 16:01:30  sparker
// Add data archive interface
//
//

#endif

