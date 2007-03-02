
#include <Packages/Uintah/StandAlone/tools/puda/varsummary.h>

#include <Packages/Uintah/StandAlone/tools/puda/util.h>

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/Variables/GridIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>

#include <Core/Containers/ConsecutiveRangeSet.h>

#include <iostream>
#include <vector>
#include <string>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

namespace SCIRun {

  // Need these min/max functions for Matrix3 in order for the templated findMinMax functions to work.
  //
  long64  Min( long64 l, long64 r) { return l < r ? l : r;  }
  long64  Max( long64 l, long64 r) { return l > r ? l : r;  }
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

template <class Type>
void
printMinMax( const Uintah::TypeDescription * td,
             Type            * min,
             Type            * max,
             IntVector       * c_min = NULL,
             IntVector       * c_max = NULL,
             int               minCnt = -1, 
             int               maxCnt = -1 )
{
  // In order to print out the values in a type-unique way, we have
  // to do the following switch(), and then cast the variables to
  // what they really are.

  switch( td->getType() ) {
  case( Uintah::TypeDescription::Matrix3 ) :
    {
      cout << "\t\t\t\tMin Norm: " << ((Matrix3*)(min))->Norm() << "\n";
      cout << "\t\t\t\tMax Norm: " << ((Matrix3*)(max))->Norm() << "\n";
      break;
    }
  case( Uintah::TypeDescription::Vector ) :
    {
      double minMagnitude = ((Vector*)(min))->length();
      double maxMagnitude = ((Vector*)(max))->length();

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

      cout << "\t\t\t\tmin magnitude: " << minMagnitude << "\n";
      cout << "\t\t\t\tmax magnitude: " << maxMagnitude << "\n";
      break;
    }
  default: 
    {
      cout << "\t\t\t\tmin value: " << *min << "\n";
      cout << "\t\t\t\tmax value: " << *max << "\n";
    }
  }
  if( c_min != NULL ) {
    cout << "\t\t\t\tmin location: " << *c_min << " (Occurrences: ~" << minCnt << ")\n";
  }
  if( c_max != NULL ) {
    cout << "\t\t\t\tmax location: " << *c_max << " (Occurrences: ~" << maxCnt << ")\n";
  }

}

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
findMinMax( DataArchive *         da,
            const string &        var,
            int                   matl,
            const Patch *         patch,
            double                time,
            bool                  use_extra_cells )
{
  Tvar value;

  const Uintah::TypeDescription * td = value.getTypeDescription();

  GridIterator iter = getIterator( td, patch, use_extra_cells );

  if( !iter.done() ) {

    da->query(value, var, matl, patch, time);

    cout << "\t\t\t\t" << td->getName() << " over " << iter.begin() << " (inclusive) to " << iter.end() << " (excluive)\n";

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

    printMinMax( td->getSubType(), &min, &max, &c_min, &c_max, minCnt, maxCnt );

  } // end if( dx dy dz )

} // end findMinMax()

////////////////////////////////////////////////////////////////////////////////
///
/// Particle Variable FindMinMax
///
template <class Tvar, class Ttype>
void
findMinMaxPV( DataArchive*          da,
              const string &        var,
              int                   matl,
              const Patch *         patch,
              double                time )
{
  Tvar value;

  const Uintah::TypeDescription * td = value.getTypeDescription();

  da->query(value, var, matl, patch, time);
  ParticleSubset* pset = value.getParticleSubset();
  cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
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
    printMinMax( td->getSubType(), &min, &max );
  }
}

void
Uintah::varsummary( DataArchive* da, CommandLineFlags & clf )
{
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(16);
  
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  cout << "There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++)
    cout << vars[i] << ": " << types[i]->getName() << endl;
      
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());

  cout << "There are " << index.size() << " timesteps:\n";

  for(int i=0;i<(int)index.size();i++) {
    cout << index[i] << ": " << times[i] << endl;
  }
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
      
  for( unsigned long t = clf.time_step_lower; t <= clf.time_step_upper; t++ ) {
    double time = times[t];
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(time);
    for(int v=0;v<(int)vars.size();v++) {
      string var = vars[v];
      const Uintah::TypeDescription* td = types[v];
      const Uintah::TypeDescription* subtype = td->getSubType();
      cout << "\tVariable: " << var << ", type " << td->getName() << endl;
      for(int l=0;l<grid->numLevels();l++){
        LevelP level = grid->getLevel(l);
        cout << "\t    Level: " << level->getIndex() << ", id " << level->getID() << endl;
        for(Level::const_patchIterator iter = level->patchesBegin();
            iter != level->patchesEnd(); iter++){
          const Patch* patch = *iter;
          cout << "\t\tPatch: " << patch->getID() << endl;
          ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
          // loop over materials
          for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
              matlIter != matls.end(); matlIter++){
            int matl = *matlIter;
            cout << "\t\t\tMaterial: " << matl << endl;
            switch(td->getType()){
              //__________________________________
              //   P A R T I C L E   V A R I A B L E
            case Uintah::TypeDescription::ParticleVariable:
              switch(subtype->getType()){
              case Uintah::TypeDescription::double_type:
                {
                  findMinMaxPV<ParticleVariable<double>,double>( da, var, matl, patch, time );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMaxPV<ParticleVariable<float>,float>( da, var, matl, patch, time );
                  break;
                }
              case Uintah::TypeDescription::int_type:
                {
                  findMinMaxPV<ParticleVariable<int>,int>( da, var, matl, patch, time );
                  break;
                }
              case Uintah::TypeDescription::Point:
                {
                  findMinMaxPV<ParticleVariable<Point>,Point>( da, var, matl, patch, time );
                  break;
                }
              case Uintah::TypeDescription::Vector:
                {
                  findMinMaxPV<ParticleVariable<Vector>,Vector>( da, var, matl, patch, time );
                  break;
                }
              case Uintah::TypeDescription::Matrix3:
                {
                  findMinMaxPV<ParticleVariable<Matrix3>,Matrix3>( da, var, matl, patch, time );
                  break;
                }
              case Uintah::TypeDescription::long64_type:
                {
                  findMinMaxPV<ParticleVariable<long64>,long64>( da, var, matl, patch, time );
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
                  findMinMax<NCVariable<double>,double>( da, var, matl, patch, time, clf.use_extra_cells );
                }
              break;
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<NCVariable<float>,float>( da, var, matl, patch, time, clf.use_extra_cells );
                }
              break;
              case Uintah::TypeDescription::Point:
                {
                  cout << "I don't think these type of variables exist... and I don't think the original\n";
                  cout << "puda was handling them correctly... If we need them, we will need to figure out\n";
                  cout << "how to deal with them properly\n";
                  exit( 1 );
                //findMinMax<NCVariable<Point>,Point>( da, var, matl, patch, time, clf.use_extra_cells );
                }
              break;
              case Uintah::TypeDescription::Vector:
                {
                  findMinMax<NCVariable<Vector>,Vector>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::Matrix3:
                {
                  findMinMax<NCVariable<Matrix3>,Matrix3>( da, var, matl, patch, time, clf.use_extra_cells );
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
                  findMinMax<CCVariable<int>,int>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::double_type:
                {
                  findMinMax<CCVariable<double>,double>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<CCVariable<float>,float>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::Point:
                {
                  cout << "I don't think these type of variables exist... and I don't think the original\n";
                  cout << "puda was handling them correctly if they do... If we need them, we will need to\n";
                  cout << "figure out how to deal with them properly\n";
                  exit( 1 );
                  //findMinMax<NCVariable<Point>,Point>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::Vector:
                {
                  findMinMax<CCVariable<Vector>,Vector>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::Matrix3:
                {
                  findMinMax<CCVariable<Matrix3>,Matrix3>( da, var, matl, patch, time, clf.use_extra_cells );
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
                  findMinMax<SFCXVariable<double>,double>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<SFCXVariable<float>,float>( da, var, matl, patch, time, clf.use_extra_cells );
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
                  findMinMax<SFCYVariable<double>,double>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<SFCYVariable<float>,float>( da, var, matl, patch, time, clf.use_extra_cells );
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
                  findMinMax<SFCZVariable<double>,double>( da, var, matl, patch, time, clf.use_extra_cells );
                  break;
                }
              case Uintah::TypeDescription::float_type:
                {
                  findMinMax<SFCZVariable<float>,float>( da, var, matl, patch, time, clf.use_extra_cells );
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
  } // end for( t )
} // end varsummary()
