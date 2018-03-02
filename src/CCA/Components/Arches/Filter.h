/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

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

  enum FILTER_TYPE { BOX, MOIN98 };

  Filter( bool use_old_filter, std::string filter_type, int filter_width ): 
  _use_old_filter(use_old_filter), _filter_width(filter_width){

    if ( filter_type == "moin98" ){ 
      _filter_type = MOIN98; 
    } else if ( filter_type == "box"){ 
      _filter_type = BOX; 

      //For box filter, we need to alway normalize by the total box volume to get the 
      //correct coefficients:
      if ( !_use_old_filter ){ 
        proc0cout << "\n Warning: Renormalizing filter because it is a box filter.\n"; 
        _use_old_filter = true; 
      }

    } else { 
      throw InvalidValue("Error: Filter type not defined.", __FILE__, __LINE__);
    }

    if ( _filter_type == MOIN98 ){ 

      // Reference for this filter from: 
      // A general class of commutative filters for LES in complex geometries
      // Vasilyev, Lund, and Moin
      // JCP 146, 82-104, 1998
      for ( int i = -(_filter_width-1)/2; i <= (_filter_width-1)/2; i++ ){
        for ( int j = -(_filter_width-1)/2; j <= (_filter_width-1)/2; j++ ){
          for ( int k = -(_filter_width-1)/2; k <= (_filter_width-1)/2; k++ ){

            int offset = abs(i) + abs(j) + abs(k); 
            double my_value = offset+3; 
            _filter_array[i+1][j+1][k+1] = 1.0 / (pow(2.0,my_value)); 

          }
        }
      }

    } else if ( _filter_type == BOX ) { 

      //Run-of-the-mill box filter.  
      for ( int i = -(_filter_width-1)/2; i <= (_filter_width-1)/2; i++ ){
        for ( int j = -(_filter_width-1)/2; j <= (_filter_width-1)/2; j++ ){
          for ( int k = -(_filter_width-1)/2; k <= (_filter_width-1)/2; k++ ){

            _filter_array[i+1][j+1][k+1] = 1.0;

          }
        }
      }
    }
  }


  ~Filter(){
  };

/* @brief Apply a filter to density */
bool applyFilter( const ProcessorGroup* ,
                  const Patch* patch,               
                  constCCVariable<double>& var,                           
                  constCCVariable<double>& eps, 
                  CCVariable<double>& filterVar )        
{
  int shift = (_filter_width-1)/2;
  int fstart = -1*shift;
  int fend   = shift;

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 
    filterVar[c] = 0.0;

    for ( int i = fstart; i <= fend; i++ ){
      for ( int j = fstart; j <= fend; j++ ){
        for ( int k = fstart; k <= fend; k++ ){

          IntVector offset = c + IntVector(i,j,k);
          filterVar[c] += _filter_array[i+shift][j+shift][k+shift] * 
                         (eps[offset]*var[offset]+ (1.-eps[offset])*var[c]); 

        }
      }
    }


  }
  return true;
}
/* @brief Apply a filter to a Uintah::CCVariable<double> */
template<class T>
bool applyFilter( const ProcessorGroup* ,
                  const Patch* patch,               
                  T& var,                           
                  constCCVariable<double>& filterVol, 
                  constCCVariable<double>& eps, 
                  Array3<double>& filterVar )        
{
  int shift = (_filter_width-1)/2;
  int fstart = -1*shift;
  int fend   = shift;

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 
    filterVar[c] = 0.0;

    for ( int i = fstart; i <= fend; i++ ){
      for ( int j = fstart; j <= fend; j++ ){
        for ( int k = fstart; k <= fend; k++ ){

          IntVector offset = c + IntVector(i,j,k);
          filterVar[c] += _filter_array[i+shift][j+shift][k+shift] * 
                         eps[offset]*var[offset]; 

        }
      }
    }

    filterVar[c] /= filterVol[c]; 

  }
  return true;
}

/* @brief Apply a filter to a RHO*Uintah::SFCX,Y,ZVariable<double> */ 
template<class T, class constT>
bool applyFilter( const ProcessorGroup* ,
                  CellIterator iter, 
                  constT& var,                           
                  constCCVariable<double>& rho, 
                  constCCVariable<double>& filterVol, 
                  constCCVariable<double>& eps, 
                  T& filterVar,
                  int dim )        
{
  int shift = (_filter_width-1)/2;
  int fstart = -1*shift;
  int fend   = shift;

  iter.reset(); 

  IntVector neigh = IntVector(0,0,0); 
  neigh[dim] = 1; 

  for (; !iter.done(); iter++){

    IntVector c = *iter; 
    filterVar[c] = 0.0;

    for ( int i = fstart; i <= fend; i++ ){
      for ( int j = fstart; j <= fend; j++ ){
        for ( int k = fstart; k <= fend; k++ ){

          IntVector offset   = c + IntVector(i,j,k);
          IntVector offset_2 = offset - neigh;  
          double vf = std::floor((eps[offset]+eps[offset_2])/2.0);

          filterVar[c] += vf * 
            _filter_array[i+shift][j+shift][k+shift] * 
            (rho[c]+rho[c+neigh])/2.0 * var[c + IntVector(i,j,k)]; 
            //(rho[offset]+rho[offset-neigh])/2.0 * var[offset]; 

        }
      }
    }

    filterVar[c] /= filterVol[c]; 

  }
  return true;
}

/* @brief Apply a filter to a Uintah::CCVariable<double> */
template<class T>
bool applyFilter_noPetsc(const ProcessorGroup* ,
                         const Patch* patch,               
                         T& var,                           
                         constCCVariable<double>& filterVol, 
                         constCCVariable<int>& cellType, 
                         Array3<double>& filterVar)        
{
  int shift = (_filter_width-1)/2;
  int fstart = -1*shift;
  int fend   = shift;

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    filterVar[c] = 0.0; 

    for ( int i = fstart; i <= fend; i++ ){
      for ( int j = fstart; j <= fend; j++ ){
        for ( int k = fstart; k <= fend; k++ ){

          IntVector offset = c + IntVector(i,j,k);
          if ( cellType[offset] == -1 ){ 
            filterVar[c] += _filter_array[i+shift][j+shift][k+shift] * var[c + IntVector(i,j,k)]; 
          }
        }
      }
    }

    filterVar[c] /= filterVol[c]; 

  }
  return true;

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

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 
    int shift = (_filter_width-1)/2;

    filterVar[c] = 0.0; 

    for ( int i = -(_filter_width-1)/2; i <= (_filter_width-1)/2; i++ ){
      for ( int j = -(_filter_width-1)/2; j <= (_filter_width-1)/2; j++ ){
        for ( int k = -(_filter_width-1)/2; k <= (_filter_width-1)/2; k++ ){

          IntVector offset = c + IntVector(i,j,k);
          if ( cellType[offset] == -1 ){ 
            filterVar[c] += _filter_array[i+shift][j+shift][k+shift] * var[c + IntVector(i,j,k)][dim]; 
          }

        }
      }
    }

    filterVar[c] /= filterVol[c]; 

  }

  return true; 

}

/** @brief This method computes the weighting factors for the filter 
 * if the weights are to be renomalized after considering non-flow cells **/ 
void computeFilterVolume( const Patch* patch, 
                          constCCVariable<int>& cellType, 
                          CCVariable<double>& fvol ){ 

  fvol.initialize(0.0); 

  if ( _filter_type == MOIN98 ){ 
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      for ( int i = -(_filter_width-1)/2; i <= (_filter_width-1)/2; i++ ){
        for ( int j = -(_filter_width-1)/2; j <= (_filter_width-1)/2; j++ ){
          for ( int k = -(_filter_width-1)/2; k <= (_filter_width-1)/2; k++ ){

            IntVector offset = c + IntVector(i,j,k);
            int fil_off = abs(i) + abs(j) + abs(k); 
            double my_value = fil_off + 3; 

            //using the old filter volume renomalizes the coefficients 
            //for the presence of boundaries. 
            if ( cellType[offset] == -1 && _use_old_filter ){ 
              fvol[c] += 1.0 / ( pow( 2.0,my_value ));
            } else { 
              fvol[c] += 1.0 / ( pow( 2.0,my_value ));
            }

          }
        }
      }
    }
  } else if ( _filter_type == BOX ){ 

    Vector Dx = patch->dCell(); 
    double vol  = Dx.x()*Dx.y()*Dx.z();

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      for ( int i = -(_filter_width-1)/2; i <= (_filter_width-1)/2; i++ ){
        for ( int j = -(_filter_width-1)/2; j <= (_filter_width-1)/2; j++ ){
          for ( int k = -(_filter_width-1)/2; k <= (_filter_width-1)/2; k++ ){

            IntVector offset = c + IntVector(i,j,k);

            //using the old filter volume renomalizes the coefficients 
            //for the presence of boundaries. 
            if ( cellType[offset] == -1 && _use_old_filter ){ 
              fvol[c] += vol; 
            } else { 
              fvol[c] += vol; 
            }

          }
        }
      }

      if ( cellType[c] == -1 ) {
        //Because we divide by the filter volume
        //in the actually filter proceedure, this should 
        //be the inverse. 
        fvol[c] = fvol[c]/vol; 
      }

    }
  }
}

//______________________________________________________________________
protected:

private:

  // Notice: the filter width is hard coded in the contructor (until we allow for variable filter widths)
  // delta = 3. 

  double _filter_array[3][3][3];              ///< Filter weights for a width of 3*dx

  bool _use_old_filter;                       ///< Adjusts the filter at boundaries when this variable is true 
  int _filter_width;                          ///< Filter width


  FILTER_TYPE _filter_type; 


}; // End class Filter.h

} // End namespace Uintah

#endif  
  





