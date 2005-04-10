//----- Stream.h -----------------------------------------------

#ifndef Uintah_Components_Arches_Stream_h
#define Uintah_Components_Arches_Stream_h

/**************************************
CLASS
   Stream
   
   Class Stream creates and stores the mixing variables that are used in Arches

GENERAL INFORMATION
   Stream.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   July 20, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/


namespace Uintah {
  namespace ArchesSpace {
    class Stream {
    public:
      Stream();
      Stream(const Stream& strm); // copy constructor
      ~Stream();
      Stream& linInterpolate(double upfactor, double lowfactor,
			     Stream& rightvalue);
      int speciesIndex(char* name); //need an access to chemkinInterface
      void print(std::ostream& out);
      double d_pressure;
      double d_density;
      double d_temperature;
      double d_enthalpy;
      bool d_mole;
      std::vector<double> d_speciesConcn;

    }; // End class Stream

  }  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/15 23:38:21  rawat
// added some more classes for implementing mixing model
//
// Revision 1.1  2000/12/18 17:53:10  rawat
// adding mixing model for reacting flows
//
//
