#ifndef Uintah_Component_Arches_Common_h
#define Uintah_Component_Arches_Common_h 
  /****************************************************************************
   CLASS
     Common
       This class allocates contiguous memory to two-dimensional arrays of type
       char or double.
 
   GENERAL INFORMATION
      Common.h - Declaration of Common class

      Author: Jennifer Spinti (spinti@crsim.utah.edu) & Rajesh Rawat

      Creation Date: 1 March 2000
 
      C-SAFE

      Copyright U of U 2000

   KEYWORDS
      Two_dimensional_array 

   DESCRIPTION

   PATTERNS
      None

   WARNINGS
      None

   POSSIBLE REVISIONS:

  ***************************************************************************/

namespace  Uintah {
// Two-dimensional character array allocation
  char **CharArray(int length, int n);

// Delete character array
  void DeleteCharArray(char **matrix, int n);

// Two-dimensional array allocation for type double
  double **AllocMatrix(int rows, int cols);

// Delete double array
  void DeallocMatrix(double **matrix, int cols);

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2001/01/31 16:35:30  rawat
// Implemented mixing and reaction models for fire.
//
// Revision 1.1.1.1 1999/06/03 14:40 Raj
// Initial New Public Checkin to CVS
//
//
