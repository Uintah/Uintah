
#ifndef UINTAH_HOMEBREW_VarLabel_H
#define UINTAH_HOMEBREW_VarLabel_H

#include <string>
#include <iostream>
#include <map>

using std::ostream;
using std::string;
using std::map;

namespace Uintah {

class TypeDescription;
class Patch;

    /**************************************
      
      CLASS
        VarLabel
      
        Short Description...
      
      GENERAL INFORMATION
      
        VarLabel.h
      
        Steven G. Parker
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2000 SCI Group
      
      KEYWORDS
        VarLabel
      
      DESCRIPTION
        Long description...
      
      WARNING
      
      ****************************************/
    
   class VarLabel {
   public:
      enum VarType {
	 Normal,
	 Internal,
	 PositionVariable
      };
     
      VarLabel(const string&, const TypeDescription*,
	       VarType vartype = Normal);

      ~VarLabel();

      // VarLabel(const string&, const TypeDescription*);
      
      inline const string& getName() const {
	 return d_name;
      }
      string getFullName(int matlIndex, const Patch* patch) const;
      bool isPositionVariable() const {
	 return d_vartype == PositionVariable;
      }

      const TypeDescription* typeDescription() const {
	 return d_td;
      }

      void allowMultipleComputes();

      bool allowsMultipleComputes() const
      { return d_allowMultipleComputes; }

      static VarLabel* find(string name);

      class Compare {
      public:
	 inline bool operator()(const VarLabel* v1, const VarLabel* v2) const {
	    if(v1 == v2)
	       return false;
	    return v1->getName() < v2->getName();
	 }
      private:
      };

      void setCompressionMode(string compressionMode)
      { d_compressionMode = compressionMode; }

      const string& getCompressionMode() const
      {
	return (d_compressionMode == "default") ?
	  defaultCompressionMode : d_compressionMode;
      }
     
      static void setDefaultCompressionMode(string compressionMode)
      { defaultCompressionMode = compressionMode; }
     
      string                 d_name;
   private:
      const TypeDescription* d_td;
      VarType                d_vartype;
      mutable string                 d_compressionMode;
      static string defaultCompressionMode;

      // Allow a variable of this label to be computed multiple
      // times in a TaskGraph without complaining.
      bool                   d_allowMultipleComputes;
     
      static map<string, VarLabel*> allLabels;     
      
      VarLabel(const VarLabel&);
      VarLabel& operator=(const VarLabel&);
   };

} // End namespace Uintah

ostream & operator<<( ostream & out, const Uintah::VarLabel & vl );

#endif
