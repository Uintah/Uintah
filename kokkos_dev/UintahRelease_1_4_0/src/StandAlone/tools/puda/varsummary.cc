/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <StandAlone/tools/puda/varsummary.h>

#include <StandAlone/tools/puda/util.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>

#include <Core/Containers/ConsecutiveRangeSet.h>

#include <sci_defs/bits_defs.h>
#include <sci_defs/osx_defs.h>  // For OSX_SNOW_LEOPARD_OR_LATER

#include <iostream>
#include <vector>
#include <string>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

namespace SCIRun {

// Need these min/max functions for Matrix3 in order for the templated findMinMax functions to work.
//
#if defined(OSX_SNOW_LEOPARD_OR_LATER) || defined(__PGI) || ( !defined(SCI_64BITS) && !defined(REDSTORM) )
  long64  Min( long64 l, long64 r) { return l < r ? l : r;  }
  long64  Max( long64 l, long64 r) { return l > r ? l : r;  }
#endif
  // The Matrix Min/Max functions are kind of arbitrary and should NOT
  // be propagated out of this file.  The only reason they are used
  // here is that the code was using a Norm comparison for Matrices,
  // and this allows them to be used in the templated findMinMax
  // functions just like everything else (thus cutting down on excess
  // code).
  Matrix3 Min( const Matrix3 & l, const Matrix3 & r )  { return l.Norm() < r.Norm() ? l : r; }
  Matrix3 Max( const Matrix3 & l, const Matrix3 & r )  { return l.Norm() > r.Norm() ? l : r; }
}

// Operator < defined by smaller Norm.  This is done so that I can use
// the templated findMinMax() without having to specialize the function for Matrix3.
bool operator < (const Matrix3 & l, const Matrix3 & r) { return l.Norm() < r.Norm(); }
bool operator > (const Matrix3 & l, const Matrix3 & r) { return l.Norm() > r.Norm(); }

////////////////////////////////////////////////////////////////////////////////////

class MinMaxInfoBase {

public:

  // Prints out the min/max values for each level.
  virtual ~MinMaxInfoBase() {};
  virtual void display() = 0;

};

template<class T>
class MinMaxInfo : public MinMaxInfoBase {

public:

  void initializeMinMax( int startingIndex );

  // Makes sure that our storage vector<> is large enough to hold the data for the current level.
  void verifyNumberOfLevels( unsigned int levelIndex )
  {
    unsigned int numLevels = levelIndex + 1;
    if( min_.size() < numLevels ) {

      min_.resize( numLevels );
      max_.resize( numLevels );

      initializeMinMax( levelIndex );
    }
  }

  // Updates the stored min_/max_ values based on the passed in min/max.
  void updateMinMax( int levelIndex, T & min, T & max );

  virtual void display();

private:
  vector<T> min_, max_; // One per level of the variable.  
};

/////////////////////////////////////////////////////////////////////
// display()

template<class Type>
void
MinMaxInfo<Type>::display()
{
  cout << "\n";
  for( unsigned int level = 0; level < min_.size(); level++ ) {
    cout << "   Level " << level << ": Min/Max: " << min_[level] << ", " << max_[level] << "\n";
  }
}

template<>
void
MinMaxInfo<Matrix3>::display()
{
  for( unsigned int level = 0; level < min_.size(); level++ ) {
    cout << "Level " << level << ": Min/Max: " << min_[0].Norm() << ", " << max_[0].Norm() << "\n";
  }
}

/////////////////////////////////////////////////////////////////////
// updateMinMax()

template<>
void
MinMaxInfo<Point>::updateMinMax( int levelIndex, Point & min, Point & max )
{
  min_[ levelIndex ] = Min( min_[ levelIndex ], min );
  max_[ levelIndex ] = Max( max_[ levelIndex ], max );
}

template<class Type>
void
MinMaxInfo<Type>::updateMinMax( int levelIndex, Type & min, Type & max ) 
{
  min_[ levelIndex ] = Min( min_[ levelIndex ], min );
  max_[ levelIndex ] = Max( max_[ levelIndex ], max );
}

/////////////////////////////////////////////////////////////////////
// initializeMinMax()

template<>
void
MinMaxInfo<Point>::initializeMinMax( int startingIndex )
{
  Point min, max;
  for( unsigned int level = startingIndex; level < min_.size(); level++ ) { // Loop through different levels.
    min_[ level ] = Point(  DBL_MAX,  DBL_MAX,  DBL_MAX );
    max_[ level ] = Point( -DBL_MAX, -DBL_MAX, -DBL_MAX );
  }
}

template<>
void
MinMaxInfo<Vector>::initializeMinMax( int startingIndex )
{
  for( unsigned int level = startingIndex; level < min_.size(); level++ ) {
    min_[ level ] = Vector(  DBL_MAX,  DBL_MAX,  DBL_MAX );
    max_[ level ] = Vector( -DBL_MAX, -DBL_MAX, -DBL_MAX );
  }
}

template<>
void
MinMaxInfo<Matrix3>::initializeMinMax( int startingIndex )
{
  for( unsigned int level = startingIndex; level < min_.size(); level++ ) {
    min_[ level ] = Matrix3( DBL_MAX, DBL_MAX, DBL_MAX,    DBL_MAX, DBL_MAX, DBL_MAX,   DBL_MAX, DBL_MAX, DBL_MAX );
    max_[ level ] = Matrix3(  0, 0, 0,   0, 0, 0,   0, 0, 0 );
  }
}

template<class Type>
void
MinMaxInfo<Type>::initializeMinMax( int startingIndex )
{
  for( unsigned int level = startingIndex; level < min_.size(); level++ ) {
    min_[ level ] =  INT_MAX;
    max_[ level ] = -INT_MAX;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
//
// This variable is used to store the global min/max values.
//
//  * Map of string ("variable_name malt#") to min/max info
//
map< string, MinMaxInfoBase * > globalMinMax;

static
void
displayGlobalMinMax()
{
  cout << "Global Min/Max are:\n\n";

  for( map< string, MinMaxInfoBase * >::iterator iter = globalMinMax.begin(); iter != globalMinMax.end(); iter++ ) {

    cout << iter->first << ": ";
    iter->second->display();

    delete iter->second;  // Free up memory
  }

  cout << "\n";

  globalMinMax.clear(); // Reset so we can find the global min/max for the next time step.
}

////////////////////////////////////////////////////////////////////////////////////

template <class Type>
void
printMinMax( CommandLineFlags & clf,
             const string     & var,
             int                matl,
             const Patch      * patch,
             const Uintah::TypeDescription * td,
             Type             * min,
             Type             * max,
             IntVector        * c_min = NULL,
             IntVector        * c_max = NULL,
             int                minCnt = -1, 
             int                maxCnt = -1 )
{
  stringstream ss;
  ss << var << " (matl: " << matl << ")";

  MinMaxInfoBase   * mmBase = globalMinMax[ ss.str() ];
  MinMaxInfo<Type> * mmInfo = dynamic_cast< MinMaxInfo<Type> *>( mmBase );
  if( mmInfo == NULL ) {
    // cout << "Creating new data store for " << var << ", malt: " << matl << " for Type: " << td->getName() << "\n";
    mmInfo = new MinMaxInfo<Type>();
    globalMinMax[ ss.str() ] = mmInfo;

  }
  mmInfo->verifyNumberOfLevels( patch->getLevel()->getIndex() );

  //cout << "Min max for '" << var << "' matl " << matl << " on level " << patch->getLevel()->getIndex() << " is:\n";

  // In order to print out the values in a type-unique way, we have
  // to do the following switch(), and then cast the variables to
  // what they really are.

  if( !clf.be_brief ) {
    cout << "\t\t\t\tmin value: " << *min << "\n";
    cout << "\t\t\t\tmax value: " << *max << "\n";
  }
  mmInfo->updateMinMax( patch->getLevel()->getIndex(), *min, *max );
  if( c_min != NULL && !clf.be_brief ) {
    cout << "\t\t\t\tmin location: " << *c_min << " (Occurrences: ~" << minCnt << ")\n";
  }
  if( c_max != NULL && !clf.be_brief ) {
    cout << "\t\t\t\tmax location: " << *c_max << " (Occurrences: ~" << maxCnt << ")\n";
  }

} // end printMinMax()

template <>
void
printMinMax<Matrix3>( CommandLineFlags & clf,
             const string     & var,
             int                matl,
             const Patch      * patch,
             const Uintah::TypeDescription * td,
             Matrix3             * min,
             Matrix3             * max,
             IntVector        * c_min ,
             IntVector        * c_max ,
             int                minCnt, 
             int                maxCnt )
{
  stringstream ss;
  ss << var << " (matl: " << matl << ")";

  MinMaxInfoBase   * mmBase = globalMinMax[ ss.str() ];
  MinMaxInfo<Matrix3> * mmInfo = dynamic_cast< MinMaxInfo<Matrix3> *>( mmBase );
  if( mmInfo == NULL ) {
    // cout << "Creating new data store for " << var << ", malt: " << matl << " for Type: " << td->getName() << "\n";
    mmInfo = new MinMaxInfo<Matrix3>();
    globalMinMax[ ss.str() ] = mmInfo;

  }
  mmInfo->verifyNumberOfLevels( patch->getLevel()->getIndex() );

  //cout << "Min max for '" << var << "' matl " << matl << " on level " << patch->getLevel()->getIndex() << " is:\n";

  // In order to print out the values in a type-unique way, we have
  // to do the following switch(), and then cast the variables to
  // what they really are.

  double patchMin = min->Norm();
  double patchMax = max->Norm();

  if( !clf.be_brief ) {
    cout << "\t\t\t\tMin Norm: " << patchMin << "\n";
    cout << "\t\t\t\tMax Norm: " << patchMax << "\n";
  }

  // Have to cast to 'what it already is' so that compiler won't
  // complain when instantiating this function for other types.
  // (Note: When this function if instantiated for, say, a Point,
  // (ie: another type), this line won't be called, so the cast is
  // ok.)

  mmInfo->updateMinMax( patch->getLevel()->getIndex(), *min, *max );

} // end printMinMax()
template <>
void
printMinMax<Vector>( CommandLineFlags & clf,
             const string     & var,
             int                matl,
             const Patch      * patch,
             const Uintah::TypeDescription * td,
             Vector             * min,
             Vector             * max,
             IntVector        * c_min,
             IntVector        * c_max,
             int                minCnt, 
             int                maxCnt)
{
  stringstream ss;
  ss << var << " (matl: " << matl << ")";

  MinMaxInfoBase   * mmBase = globalMinMax[ ss.str() ];
  MinMaxInfo<Vector> * mmInfo = dynamic_cast< MinMaxInfo<Vector> *>( mmBase );
  if( mmInfo == NULL ) {
    // cout << "Creating new data store for " << var << ", malt: " << matl << " for Type: " << td->getName() << "\n";
    mmInfo = new MinMaxInfo<Vector>();
    globalMinMax[ ss.str() ] = mmInfo;

  }
  mmInfo->verifyNumberOfLevels( patch->getLevel()->getIndex() );

  //cout << "Min max for '" << var << "' matl " << matl << " on level " << patch->getLevel()->getIndex() << " is:\n";

  // In order to print out the values in a type-unique way, we have
  // to do the following switch(), and then cast the variables to
  // what they really are.

  double minMagnitude = min->length();
  double maxMagnitude = max->length();

  if( minMagnitude > maxMagnitude ) {
    IntVector * c_temp = c_min;
    c_min = c_max;
    c_max = c_temp;

    double temp = minMagnitude;
    minMagnitude = maxMagnitude;
    maxMagnitude = temp;

    int cntTemp = minCnt;
    minCnt = maxCnt;
    maxCnt = cntTemp;
  }

  ((MinMaxInfo<Vector>*) mmInfo)->updateMinMax( patch->getLevel()->getIndex(), *min, *max );

  if( !clf.be_brief ) {
    cout << "\t\t\t\tmin magnitude: " << minMagnitude << "\n";
    cout << "\t\t\t\tmax magnitude: " << maxMagnitude << "\n";
  }
} // end printMinMax()





////////////////////////////////////////////////////////////////////////////////////
// Returns the appropriate iterator depending on the type (td) of the variable.
GridIterator
getIterator( const Uintah::TypeDescription * td, const Patch * patch, bool use_extra_cells ) 
{
  switch( td->getType() )
    {
    case Uintah::TypeDescription::NCVariable :    return GridIterator( patch->getNodeIterator() );
    case Uintah::TypeDescription::CCVariable :    return (use_extra_cells ? GridIterator( patch->getExtraCellIterator() ) : 
                                                                            GridIterator( patch->getCellIterator() ) );
    case Uintah::TypeDescription::SFCXVariable :  return GridIterator( patch->getSFCXIterator() );
    case Uintah::TypeDescription::SFCYVariable :  return GridIterator( patch->getSFCYIterator() );
    case Uintah::TypeDescription::SFCZVariable :  return GridIterator( patch->getSFCZIterator() );
    default:
      cout << "ERROR: Don't know how to handle type: " << td->getName() << "\n";
      exit( 1 );
    }
} // end getIterator()

////////////////////////////////////////////////////////////////////////////////////
//
template <class Tvar, class Ttype>
void
findMinMax( DataArchive         * da,
            const string        & var,
            int                   matl,
            const Patch         * patch,
            int                   timestep,
            CommandLineFlags    & clf )
{
  Tvar value;

  const Uintah::TypeDescription * td = value.getTypeDescription();

  GridIterator iter = getIterator( td, patch, clf.use_extra_cells );

  if( !iter.done() ) {

    da->query(value, var, matl, patch, timestep);

    if( !clf.be_brief ) {
      cout << "\t\t\t\t" << td->getName() << " over " << iter.begin() << " (inclusive) to " 
           << iter.end() << " (excluive)\n";
    }

    Ttype min, max;
    IntVector c_min, c_max;

    int minCnt = 1;
    int maxCnt = 1;

    // Set initial values:
    max = value[*iter];
    min = max;
    c_max = *iter;
    c_min = c_max;
    
    iter++; // No need to do a comparison on the initial cell
    
    for( ; !iter.done(); iter++ ) {

      Ttype val = value[*iter];

      if (val == min) { minCnt++; }
      if (val == max) { maxCnt++; }

      if (val < min) {
        c_min = *iter;
        minCnt = 1;
      }
      if (val > max ) {
        c_max = *iter;
        maxCnt = 1;
      }

      // Have to use the Min/Max functions here (instead of just
      // setting the values above) becuase we don't know what type we
      // are dealing with, and Min/Max take care of it for us.
      // (Specifically, if it is a Matrix, it uses Norm(), etc).
      min = Min(min, val);
      max = Max(max, val);
    }

    printMinMax<Ttype>( clf, var, matl, patch, td->getSubType(), &min, &max, &c_min, &c_max, minCnt, maxCnt );

  } // end if( dx dy dz )

} // end findMinMax()

////////////////////////////////////////////////////////////////////////////////
///
/// Particle Variable FindMinMax
///
template <class Tvar, class Ttype>
void
findMinMaxPV( DataArchive      * da,
              const string     & var,
              int                matl,
              const Patch      * patch,
              int                timestep,
              CommandLineFlags & clf ){
  Tvar value;

  const Uintah::TypeDescription * td = value.getTypeDescription();

  da->query(value, var, matl, patch, timestep);
  ParticleSubset* pset = value.getParticleSubset();
  if( !clf.be_brief ) {
    cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
  }
  if( pset->numParticles() > 0 ) {
    Ttype min, max;
    ParticleSubset::iterator iter = pset->begin();
    max = value[*iter++];
    min = max;
    for( ;iter != pset->end(); iter++ ) {
      // Forced to cast to (T) so that the non-ambiguous min/max function is used.
      min = Min( (Ttype)min, (Ttype)(value[*iter]) );
      max = Max( (Ttype)max, (Ttype)(value[*iter]) );
    }
    //IntVector c_min, c_max;
    //int       minCnt = -1, maxCnt = -1;
    printMinMax<Ttype>( clf, var, matl, patch, td->getSubType(), &min, &max );
  }
}

void
Uintah::varsummary( DataArchive* da, CommandLineFlags & clf, int mat )
{
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(16);
 
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());

  cout << "There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++)
    cout << "  " << vars[i] << ": " << types[i]->getName() << endl;
  cout << "\n";

  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());

  cout << "There are " << index.size() << " timesteps:\n";

  for(int i=0;i<(int)index.size();i++) {
    cout << "  " << index[i] << ": " << times[i] << endl;
  }

  cout << "\n";
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
      
  for( unsigned long t = clf.time_step_lower; t <= clf.time_step_upper; t++ ) {
    double time = times[t];

    cout << "----------------------------------------------------------------------\n";
    cout << "Time = " << time << endl;
    cout << "\n";
    GridP grid = da->queryGrid(t);
    for(int v=0;v<(int)vars.size();v++) {
      string var = vars[v];
      const Uintah::TypeDescription* td = types[v];
      const Uintah::TypeDescription* subtype = td->getSubType();
      if( !clf.be_brief ) {
        cout << "\tVariable: " << var << ", type " << td->getName() << endl;
      }
      for(int l=0;l<grid->numLevels();l++){
        LevelP level = grid->getLevel(l);
        if( !clf.be_brief ) {
          cout << "\t    Level: " << level->getIndex() << ", id " << level->getID() << endl;
        }
        for(Level::const_patchIterator iter = level->patchesBegin();
            iter != level->patchesEnd(); iter++){
          const Patch* patch = *iter;
          if( !clf.be_brief ) {
            cout << "\t\tPatch: " << patch->getID() << endl;
          }
          ConsecutiveRangeSet matls = da->queryMaterials(var, patch, t);
          // loop over materials
          for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
              matlIter != matls.end(); matlIter++){
            int matl = *matlIter;
            if (mat != -1 && matl != mat) continue;
            if( !clf.be_brief ) {
              cout << "\t\t\tMaterial: " << matl << endl;
            }
            switch(td->getType()){
              //__________________________________
              //   P A R T I C L E   V A R I A B L E
            case Uintah::TypeDescription::ParticleVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::double_type:
                {
                  findMinMaxPV<ParticleVariable<double>,double>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMaxPV<ParticleVariable<float>,float>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::int_type:
                {
                  findMinMaxPV<ParticleVariable<int>,int>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Point:
                {
                  findMinMaxPV<ParticleVariable<Point>,Point>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Vector:
                {
                  findMinMaxPV<ParticleVariable<Vector>,Vector>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Matrix3:
                {
                  findMinMaxPV<ParticleVariable<Matrix3>,Matrix3>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::long64_type:
                {
                  findMinMaxPV<ParticleVariable<long64>,long64>( da, var, matl, patch, t, clf );
                  break;
                }
              default:
                cerr << "Particle Variable of unknown type: " << subtype->getName() << endl;
                break;
              }
              break;
              //__________________________________  
              //  N C   V A R I A B L E S           
            case Uintah::TypeDescription::NCVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::double_type:
                {
                  findMinMax<NCVariable<double>,double>( da, var, matl, patch, t, clf );
                }
              break;
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<NCVariable<float>,float>( da, var, matl, patch, t, clf );
                }
              break;
              case Uintah::TypeDescription::Point:
                {
                  cout << "I don't think these type of variables exist... and I don't think the original\n";
                  cout << "puda was handling them correctly... If we need them, we will need to figure out\n";
                  cout << "how to deal with them properly\n";
                  exit( 1 );
                //findMinMax<NCVariable<Point>,Point>( da, var, matl, patch, time, clf );
                }
              break;
              case Uintah::TypeDescription::Vector:
                {
                  findMinMax<NCVariable<Vector>,Vector>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Matrix3:
                {
                  findMinMax<NCVariable<Matrix3>,Matrix3>( da, var, matl, patch, t, clf );
                  break;
                }
              default:
                cerr << "NC Variable of unknown type: " << subtype->getName() << endl;
                break;
              }
              break;
              //__________________________________
              //   C C   V A R I A B L E S
            case Uintah::TypeDescription::CCVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::int_type:
                {
                  findMinMax<CCVariable<int>,int>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::double_type:
                {
                  findMinMax<CCVariable<double>,double>( da, var, matl, patch, t, clf);
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<CCVariable<float>,float>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Point:
                {
                  cout << "I don't think these type of variables exist... and I don't think the original\n";
                  cout << "puda was handling them correctly if they do... If we need them, we will need to\n";
                  cout << "figure out how to deal with them properly\n";
                  exit( 1 );
                  //findMinMax<NCVariable<Point>,Point>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Vector:
                {
                  findMinMax<CCVariable<Vector>,Vector>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::Matrix3:
                {
                  findMinMax<CCVariable<Matrix3>,Matrix3>( da, var, matl, patch, t, clf );
                  break;
                }
              break;
              default:
                cerr << "CC Variable of unknown type: " << subtype->getName() << endl;
                break;
              }
              break;
              //__________________________________
              //   S F C X   V A R I A B L E S
            case Uintah::TypeDescription::SFCXVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::double_type:
                {
                  findMinMax<SFCXVariable<double>,double>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<SFCXVariable<float>,float>( da, var, matl, patch, t, clf );
                  break;
                }
              default:
                cerr << "SCFXVariable  of unknown type: " << subtype->getType() << endl;
                break;
              }
              break;
              //__________________________________
              //   S F C Y  V A R I A B L E S
            case Uintah::TypeDescription::SFCYVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::double_type:
                {
                  findMinMax<SFCYVariable<double>,double>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<SFCYVariable<float>,float>( da, var, matl, patch, t, clf );
                  break;
                }
              default:
                cerr << "SCFYVariable  of unknown type: " << subtype->getType() << "\n";
                break;
              }
              break;
              //__________________________________
              //   S F C Z   V A R I A B L E S
            case Uintah::TypeDescription::SFCZVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::double_type:
                {
                  findMinMax<SFCZVariable<double>,double>( da, var, matl, patch, t, clf );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<SFCZVariable<float>,float>( da, var, matl, patch, t, clf );
                  break;
                }
              default:
                cerr << "SCFZVariable  of unknown type: " << subtype->getType() << "\n";
                break;
              }
              break;
              //__________________________________
              //  BULLET PROOFING
            default:
              cerr << "Variable of unknown type: " << td->getName() << endl;
              break;

            } // end switch( type )
          } // end for( matlIter )
        } // end for( patchIter )
      } // end for( l )
    } // end for( v )

    // Display the global min/max for this timestep.
    displayGlobalMinMax();

  } // end for( t )
} // end varsummary()
