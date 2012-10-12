/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>

#include <StandAlone/tools/uda2nrrd/particles.h>

#include <Core/Geometry/Point.h>

using namespace std;
using namespace SCIRun;

//////////////////////////////////////////////////////////////////////////////////

static bool
machineIsBigEndian()
{
  short i = 0x4321;
  if((*(char *)&i) != 0x21 ){
    return true;
  } else {
    return false;
  }
}

//////////////////////////////////////////////////////////////////////////////////

template<>
ParticleDataContainer
handleParticleData<Point>( QueryInfo & qinfo )
{
  vector<float> dataX, dataY, dataZ;

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;

      ParticleVariable<Point> value;
      qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );

      ParticleSubset* pset = value.getParticleSubset();

      if (!pset) { 
        printf("not sure if this case is handled correctly....\n");
        exit( 1 );
      }

      int numParticles = pset->numParticles();

      if (numParticles > 0) {
        for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
          dataX.push_back( (float)value[*iter].x() );
          dataY.push_back( (float)value[*iter].y() );
          dataZ.push_back( (float)value[*iter].z() );
        }
      }
    } // end for materials
  } // end for each Patch

  float * floatArrayX = (float*)malloc(sizeof(float)*dataX.size());
  float * floatArrayY = (float*)malloc(sizeof(float)*dataX.size());
  float * floatArrayZ = (float*)malloc(sizeof(float)*dataX.size());
    
  float min[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
  float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

  for( unsigned int pos = 0; pos < dataX.size(); pos++ ) {
    floatArrayX[ pos ] = dataX[ pos ];
    floatArrayY[ pos ] = dataY[ pos ];
    floatArrayZ[ pos ] = dataZ[ pos ];

    if( dataX[ pos ] > max[0] ) { max[0] = dataX[ pos ]; }
    if( dataX[ pos ] < min[0] ) { min[0] = dataX[ pos ]; }
    if( dataY[ pos ] > max[1] ) { max[1] = dataY[ pos ]; }
    if( dataY[ pos ] < min[1] ) { min[1] = dataY[ pos ]; }
    if( dataZ[ pos ] > max[2] ) { max[2] = dataZ[ pos ]; }
    if( dataZ[ pos ] < min[2] ) { min[2] = dataZ[ pos ]; }
  }

  printf("%s (%d):  min/max: %f / %f, %f / %f, %f / %f\n", 
         qinfo.varname.c_str(), (int)dataX.size(), min[0], max[0], min[1], max[1], min[2], max[2] );

  ParticleDataContainer result;

  result.name = qinfo.varname;

  result.x = floatArrayX;
  result.y = floatArrayY;
  result.z = floatArrayZ;

  result.numParticles = dataX.size();

  return result;
}

template<>
ParticleDataContainer
handleParticleData<Vector>( QueryInfo & qinfo )
{
  vector<float> data;

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;

      ParticleVariable<Vector> value;
      qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );
      ParticleSubset* pset = value.getParticleSubset();
      if (!pset) {
        printf("NOT sure that this case is being handled correctly...\n");
        exit( 1 );
      }
      int numParticles = pset->numParticles();

      if (numParticles > 0) {
        for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
          data.push_back( (float)value[*iter].length() );
        }
      } // end if numParticles > 0
    } // end for each matl
  } // end for each Patch

  float * floatArray = (float*)malloc(sizeof(float)*data.size());

  float min =  FLT_MAX;
  float max = -FLT_MAX;

  for( unsigned int pos = 0; pos < data.size(); pos++ ) {
    floatArray[ pos ] = data[ pos ];

    if( data[ pos ] > max ) { max = data[ pos ]; }
    if( data[ pos ] < min ) { min = data[ pos ]; }
  }

  printf("%s (%d):  min/max: %f / %f\n", qinfo.varname.c_str(), (int)data.size(), min, max);

  ParticleDataContainer result( qinfo.varname, floatArray, data.size() );

  return result;
} // end handleParticleData<Vector>

template<>
ParticleDataContainer
handleParticleData<Matrix3>( QueryInfo & qinfo )
{
  vector<float> data;

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;

      ParticleVariable<Matrix3> value;
      qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );

      ParticleSubset* pset = value.getParticleSubset();

      if (!pset) { 
        printf("not sure if this case is handled correctly....\n");
        exit( 1 );
      }

      int numParticles = pset->numParticles();

      if (numParticles > 0) {
        for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
          float temp_value = (float)(value[*iter].Trace()/3.0); // Trace 3
          // float temp_value = (float)(sqrt(1.5*(value[*iter]-one*temp_value).NormSquared())); // Equivalent 
          
          data.push_back( temp_value );
        }
      }
    } // end for each Material
  } // end for each Patch

  float * floatArray = (float*)malloc(sizeof(float)*data.size());

  float min =  FLT_MAX;
  float max = -FLT_MAX;

  for( unsigned int pos = 0; pos < data.size(); pos++ ) {

    floatArray[ pos ] = data[ pos ];

    if( data[ pos ] > max ) { max = data[ pos ]; }
    if( data[ pos ] < min ) { min = data[ pos ]; }
  }

  printf("%s (%d):  min/max: %f / %f\n", qinfo.varname.c_str(), (int)data.size(), min, max);

  // vardata.name = string(var+" Trace/3");
  // vardata2.name = string(var+" Equivalent");

  // TODO: just doing trace 3 right now... update to to both...

  ParticleDataContainer result( qinfo.varname + " Trace/3", floatArray, data.size() );

  return result;
} // end handleParticleData<Matrix3>

template<class PartT>
ParticleDataContainer
handleParticleData( QueryInfo & qinfo )
{
  vector<float> data;

  bool do_radius_computation = false;
  bool compute_radius = false;

  if( compute_radius && qinfo.varname == "p.volume" ) do_radius_computation = true;

  // If we are computing the radius, change the name
  string name;
  if (!do_radius_computation) {
    name = qinfo.varname;
  }
  else {
    name = "Radius from p.volume";
  }

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;

      ParticleVariable<PartT> value;
      qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );
      ParticleSubset* pset = value.getParticleSubset();
      if (!pset) {
        printf("NOT sure that this case is being handled correctly...\n");
        exit( 1 );
      }
      int numParticles = pset->numParticles();

      if (numParticles > 0) {
        if (!do_radius_computation) {
          for( ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
            data.push_back( (float) value[*iter] );
          }
        }
        else { // do_radius_computation
          for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
            float temp_value = (float)value[*iter];
            if (temp_value > 0) {
              // Create a radius variable value...
              temp_value = powf(temp_value*(3.0f/4.0f*M_1_PI), 1.0f/3.0f);
            } else {
              temp_value = 0;
            }
            data.push_back( temp_value );
          }
        }
      } // end if numParticles > 0
    } // end for each Material
  } // end for each Patch

  float * floatArray = (float*)malloc(sizeof(float)*data.size());

  float min =  FLT_MAX;
  float max = -FLT_MAX;
  for( unsigned int pos = 0; pos < data.size(); pos++ ) {
    floatArray[ pos ] = data[ pos ];
    if( data[ pos ] > max ) { max = data[ pos ]; }
    if( data[ pos ] < min ) { min = data[ pos ]; }
  }

  if( data.size() == 0 ) {
    printf( "\n" );
    printf( "ERROR?  Data size is 0 for %s....\n", name.c_str() );
    printf( "        In other words, no particles were found for this variable.\n" );
    printf( "\n" );
  }
  else {
    printf("%s (%d):  min/max: %f / %f\n", name.c_str(), (int)data.size(), min, max);
  }

  ParticleDataContainer result( name, floatArray, data.size() );

  return result;

} // end handleParticleData<generic>()


void
saveParticleData( vector<ParticleDataContainer> & particleVars,
                  const string                  & filename,
                  double                          current_time )
{

  string header = filename + ".nhdr";

  FILE * out = fopen( header.c_str(), "wb" );

  if( !out ) {
    cerr << "ERROR: Could not open '" << header << "' for writing.\n";
    return;
  }

  unsigned int numParticles = particleVars[0].numParticles;
  unsigned int numVars      = 0;

  for( unsigned int cnt = 0; cnt < particleVars.size(); cnt++ ) {

    if( particleVars[cnt].data ) {
      numVars++;
    }
    else {
      numVars += 3; // x, y, and z
    }
    
    // Bullet proofing...
    if( particleVars[cnt].numParticles != numParticles ) {
      cout << "ERROR: Inconsistency in number of particles...\n";
      cout << "       " << particleVars[0].name << " had " << numParticles << " particles\n";
      cout << "       but " << particleVars[cnt].name << " has " << numParticles << " particles...\n";
      exit( 1 );
    }
  }

  string endianness = machineIsBigEndian() ? "big" : "little";

  //////////////////////
  // Write NRRD Header:

  fprintf( out, "NRRD0001\n"
                "# Complete NRRD file format specification at:\n"
                "# http://teem.sourceforge.net/nrrd/format.html\n"
                "type: float\n"
                "dimension: 2\n"
                "sizes: %d %d\n"
                "endian: %s\n"
                "encoding: raw\n"
                "Num Particles:=%d\n"
                "Num Variables:=%d\n"
                "time:=%.9lf\n",
           numVars, numParticles, endianness.c_str(), numParticles, numVars, current_time );

  int pos = 0;
  for( unsigned int cnt = 0; cnt < particleVars.size(); cnt++ ) {
    
    if( particleVars[cnt].name == "p.x" ) {
      fprintf( out, "p.x (x) index:=%d\n", pos++ );
      fprintf( out, "p.x (y) index:=%d\n", pos++ );
      fprintf( out, "p.x (z) index:=%d\n", pos++ );
    } 
    else {
      fprintf( out, "%s index:=%d\n", particleVars[cnt].name.c_str(), pos++ );
    }
  }

  string rawfile = filename + ".raw";

  fprintf( out, "data file: ./%s\n", rawfile.c_str() );
  fclose( out );
 
  out = fopen( rawfile.c_str(), "wb" );

  if( !out ) {
    cerr << "ERROR: Could not open '" << rawfile << "' for writing.\n";
    return;
  }

  //////////////////////
  // Write NRRD Data:

  for( unsigned int particle = 0; particle < numParticles; particle++ ) {

    for( unsigned int cnt = 0; cnt < particleVars.size(); cnt++ ) {

      // Size_t is unsigned so this is invalid:
      size_t wrote = 0;

      if( particleVars[cnt].data != NULL ) {

        wrote = fwrite( &particleVars[cnt].data[particle], sizeof(float), 1, out );
        
      } else {
        wrote = fwrite( &particleVars[cnt].x[particle], sizeof(float), 1, out );
        wrote = fwrite( &particleVars[cnt].y[particle], sizeof(float), 1, out );
        wrote = fwrite( &particleVars[cnt].z[particle], sizeof(float), 1, out );
      }

      if( wrote != 1 ) {
        cerr << "ERROR: Wrote out " << wrote << " floats instead of just one...\n";
        fclose(out);
        return;
      }
    }
  }

  fclose(out);
  
  cout << "\nDone writing out particle NRRD: " << filename << ".{raw,nhdr}\n\n";
}


///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.

template ParticleDataContainer handleParticleData<int>    (QueryInfo&);
template ParticleDataContainer handleParticleData<long64> (QueryInfo&);
template ParticleDataContainer handleParticleData<float>  (QueryInfo&);
template ParticleDataContainer handleParticleData<double> (QueryInfo&);
template ParticleDataContainer handleParticleData<Point>  (QueryInfo&);
template ParticleDataContainer handleParticleData<Vector> (QueryInfo&);
template ParticleDataContainer handleParticleData<Matrix3>(QueryInfo&);
