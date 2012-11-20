/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Uintah_Components_Arches_Filter_h
#define Uintah_Components_Arches_Filter_h


#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/PetscCommon.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Containers/Array1.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah {

/**
  * @class Filter.cc
  * @author Jeremy Thornock
  * @date Oct, 2012
  *
  * @brief StandAlone filter class for filter vector and scalar variables.
  *
  */

class Filter {

public:

  Filter( bool use_old_filter ) : d_use_old_filter(use_old_filter){

    // Reference for this filter from: 
    // A general class of commutative filters for LES in complex geometries
    // Vasilyev, Lund, and Moin
    // JCP 146, 82-104, 1998
    //
    _filter_width = 3; 

    for ( int i = -(_filter_width-1)/2; i <= (_filter_width-1)/2; i++ ){
      for ( int j = -(_filter_width-1)/2; j <= (_filter_width-1)/2; j++ ){
        for ( int k = -(_filter_width-1)/2; k <= (_filter_width-1)/2; k++ ){

          int offset = abs(i) + abs(j) + abs(k); 
          double my_value = offset+3; 
          filter_array[i+1][j+1][k+1] = 1.0 / (pow(2.0,my_value)); 

        }
      }
    }
  }
  ~Filter(){
  };

/* @brief Apply a filter to a Uintah::CCVariable<double> */
template<class T>
bool applyFilter_noPetsc(const ProcessorGroup* ,
                         const Patch* patch,               
                         T& var,                           
                         constCCVariable<double>& filterVol, 
                         constCCVariable<int>& cellType, 
                         Array3<double>& filterVar)        
{

  bool it_worked = false; 

  if ( d_use_old_filter ){ 
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
      int filter_width = 3; //hard coded for now
      int shift = (filter_width-1)/2;

      filterVar[c] = 0.0; 

      for ( int i = -(filter_width-1)/2; i <= (filter_width-1)/2; i++ ){
        for ( int j = -(filter_width-1)/2; j <= (filter_width-1)/2; j++ ){
          for ( int k = -(filter_width-1)/2; k <= (filter_width-1)/2; k++ ){

            IntVector offset = c + IntVector(i,j,k);
            if ( cellType[offset] == -1 ){ 
              filterVar[c] += filter_array[i+shift][j+shift][k+shift] * var[c + IntVector(i,j,k)]; 
            }

          }
        }
      }

      filterVar[c] /= filterVol[c]; 

    }

    it_worked = true; 

  } else { 
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
      int filter_width = 3; //hard coded for now
      int shift = (filter_width-1)/2;

      filterVar[c] = 0.0; 

      for ( int i = -(filter_width-1)/2; i <= (filter_width-1)/2; i++ ){
        for ( int j = -(filter_width-1)/2; j <= (filter_width-1)/2; j++ ){
          for ( int k = -(filter_width-1)/2; k <= (filter_width-1)/2; k++ ){

            filterVar[c] += filter_array[i+shift][j+shift][k+shift] * var[c + IntVector(i,j,k)]; 

          }
        }
      }

    }

    it_worked = true; 

  } 
  return it_worked;
}

/* @brief Apply a filter to a component of a Uintah::Vector. dim = (0,1,2) = vector component */
bool applyFilter_noPetsc(const ProcessorGroup* ,
                         const Patch* patch,               
                         constCCVariable<Vector>& var,                           
                         constCCVariable<double>& filterVol, 
                         constCCVariable<int>& cellType, 
                         Array3<double>& filterVar,
                         int dim )        
{

  bool it_worked = false; 

  if ( d_use_old_filter ){ 
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
      int filter_width = 3; //hard coded for now
      int shift = (filter_width-1)/2;

      filterVar[c] = 0.0; 

      for ( int i = -(filter_width-1)/2; i <= (filter_width-1)/2; i++ ){
        for ( int j = -(filter_width-1)/2; j <= (filter_width-1)/2; j++ ){
          for ( int k = -(filter_width-1)/2; k <= (filter_width-1)/2; k++ ){

            IntVector offset = c + IntVector(i,j,k);
            if ( cellType[offset] == -1 ){ 
              filterVar[c] += filter_array[i+shift][j+shift][k+shift] * var[c + IntVector(i,j,k)][dim]; 
            }

          }
        }
      }

      filterVar[c] /= filterVol[c]; 

    }

    it_worked = true; 

  } else { 
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 
      int filter_width = 3; //hard coded for now
      int shift = (filter_width-1)/2;

      filterVar[c] = 0.0; 

      for ( int i = -(filter_width-1)/2; i <= (filter_width-1)/2; i++ ){
        for ( int j = -(filter_width-1)/2; j <= (filter_width-1)/2; j++ ){
          for ( int k = -(filter_width-1)/2; k <= (filter_width-1)/2; k++ ){

            filterVar[c] += filter_array[i+shift][j+shift][k+shift] * var[c + IntVector(i,j,k)][dim]; 

          }
        }
      }

    }

    it_worked = true; 

  } 
  return it_worked;
}
//______________________________________________________________________
protected:

private:

  // Notice: the filter width is hard coded in the contructor (until we allow for variable filter widths)
  // delta = 3. 
  int _filter_width;                          ///< Filter width

  double filter_array[3][3][3];               ///< Filter weights for a width of 3*dx

  bool d_use_old_filter;                      ///< Adjusts the filter at boundaries when this variable is true 


}; // End class Filter.h

} // End namespace Uintah

#endif  
  





