
#include <Packages/Uintah/StandAlone/tools/uda2nrrd/particles.h>

#include <Core/Geometry/Point.h>

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
handleParticleData<Point>( QueryInfo & qinfo, int matlNo, bool matlClassfication )
{
  vector<float> dataX, dataY, dataZ;

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;
      if (matlClassfication && (matl != matlNo))
	    continue;

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

  // printf("%s (%d):  min/max: %f / %f, %f / %f, %f / %f\n", 
  //       qinfo.varname.c_str(), (int)dataX.size(), min[0], max[0], min[1], max[1], min[2], max[2] );

  ParticleDataContainer result;

  result.name = qinfo.varname;

  result.x = floatArrayX;
  result.y = floatArrayY;
  result.z = floatArrayZ;

  result.numParticles = dataX.size();

  return result;
}

/*
template<>
ParticleDataContainer
handleParticleData<Vector>( QueryInfo & qinfo, int matlNo, bool matlClassfication )
{
  vector<float> data;

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;
	  if (matlClassfication && (matl != matlNo))
	    continue;

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

  // printf("%s (%d):  min/max: %f / %f\n", qinfo.varname.c_str(), (int)data.size(), min, max);

  ParticleDataContainer result( qinfo.varname, floatArray, data.size() );

  return result;
} // end handleParticleData<Vector>*/

template<>
ParticleDataContainer
handleParticleData<Vector>( QueryInfo & qinfo, int matlNo, bool matlClassfication )
{
  cout << "In handleParticleData<Vector>\n";
  vector<float> dataX, dataY, dataZ;

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;
	  if (matlClassfication && (matl != matlNo))
	    continue;

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
          dataX.push_back( (float)value[*iter].x() );
		  dataY.push_back( (float)value[*iter].y() );
		  dataZ.push_back( (float)value[*iter].z() );
        }
      } // end if numParticles > 0
    } // end for each matl
  } // end for each Patch

  float * floatArrayX = (float*)malloc(sizeof(float)*dataX.size());
  float * floatArrayY = (float*)malloc(sizeof(float)*dataY.size());
  float * floatArrayZ = (float*)malloc(sizeof(float)*dataZ.size());

  // float min =  FLT_MAX;
  // float max = -FLT_MAX;

  for( unsigned int pos = 0; pos < dataX.size(); pos++ ) {
    floatArrayX[ pos ] = dataX[ pos ];
    floatArrayY[ pos ] = dataY[ pos ];
    floatArrayZ[ pos ] = dataZ[ pos ];

    // if( data[ pos ] > max ) { max = data[ pos ]; }
    // if( data[ pos ] < min ) { min = data[ pos ]; }
  }

  // printf("%s (%d):  min/max: %f / %f\n", qinfo.varname.c_str(), (int)data.size(), min, max);

  // ParticleDataContainer result( qinfo.varname, floatArray, data.size() );
  
  ParticleDataContainer result;

  result.name = qinfo.varname;
  result.x = floatArrayX;
  result.y = floatArrayY;
  result.z = floatArrayZ;
  result.type = 1;
  result.numParticles = dataX.size();

  cout << "Out handleParticleData<Vector>\n";	

  return result;
} // end handleParticleData<Vector>

template<>
ParticleDataContainer
handleParticleData<Matrix3>( QueryInfo & qinfo, int matlNo, bool matlClassfication )
{
  cout << "In handleParticleData<Matrix3>\n";
  vector<float> data;
  matrixVec* matrixRep = new matrixVec();

  // Loop over each patch and get the data from the data archive.
  Level::const_patchIterator patch_it;

  for( patch_it = qinfo.level->patchesBegin(); patch_it != qinfo.level->patchesEnd(); ++patch_it) {
    const Patch* patch = *patch_it;

    for( ConsecutiveRangeSet::iterator matlIter = qinfo.materials.begin(); matlIter != qinfo.materials.end(); matlIter++ ) {

      int matl = *matlIter;
	  if (matlClassfication && (matl != matlNo))
	    continue;

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
		  
		  // pushing individual matrices into the repository
		  matrixRep->push_back(value[*iter]);
          
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

  // printf("%s (%d):  min/max: %f / %f\n", qinfo.varname.c_str(), (int)data.size(), min, max);

  // vardata.name = string(var+" Trace/3");
  // vardata2.name = string(var+" Equivalent");

  // TODO: just doing trace 3 right now... update to to both...

  // ParticleDataContainer result( qinfo.varname + " Trace/3", floatArray, data.size() );

  ParticleDataContainer result;

  result.name = qinfo.varname;
  result.data = floatArray;
  result.matrixRep = matrixRep;
  result.type = 2;
  result.numParticles = data.size();
  
  cout << "Out handleParticleData<Matrix3>\n";

  return result;
}

template<class PartT>
ParticleDataContainer
handleParticleData( QueryInfo & qinfo, int matlNo, bool matlClassfication )
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
	  if (matlClassfication && (matl != matlNo))
	    continue;

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

  // printf("%s (%d):  min/max: %f / %f\n", name.c_str(), (int)data.size(), min, max);

  ParticleDataContainer result( name, floatArray, data.size() );

  return result;

} // end handleParticleData()


void
saveParticleData( vector<ParticleDataContainer> & particleVars,
                  const string                  & filename, 
				  variables & varColln  ) // New parameter
{

  /*string header = filename + ".nhdr";

  FILE * out = fopen( header.c_str(), "wb" );

  if( !out ) {
    cerr << "ERROR: Could not open '" << header << "' for writing.\n";
    return;
  }*/

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

  /*fprintf( out, "NRRD0001\n"
                "# Complete NRRD file format specification at:\n"
                "# http://teem.sourceforge.net/nrrd/format.html\n"
                "type: float\n"
                "dimension: 2\n"
                "sizes: %d %d\n"
                "endian: %s\n"
                "encoding: raw\n"
                "# Num Particles: %d\n"
                "# Variables (%d):\n", numVars, numParticles*4, endianness.c_str(), numParticles, numVars );

  int pos = 0;
  for( unsigned int cnt = 0; cnt < particleVars.size(); cnt++ ) {
    
    if( particleVars[cnt].name == "p.x" ) {
      fprintf( out, "#   %d - p.x (x)\n", pos++ );
      fprintf( out, "#   %d - p.x (y)\n", pos++ );
      fprintf( out, "#   %d - p.x (z)\n", pos++ );
    } 
    else {
      fprintf( out, "#   %d - %s\n", pos++, particleVars[cnt].name.c_str() );
    }
  }
  fclose( out );

  string rawfile = filename + ".raw";
  out = fopen( rawfile.c_str(), "wb" );

  if( !out ) {
    cerr << "ERROR: Could not open '" << rawfile << "' for writing.\n";
    return;
  }*/

  //////////////////////
  // Write NRRD Data:

  for( unsigned int particle = 0; particle < numParticles; particle++ ) {
	
	variable varData;
	unknownData& dataRef = varData.data;
	vecValData& vecDataRef = varData.vecData;
    tenValData& tenDataRef = varData.tenData;
	
    for( unsigned int cnt = 0; cnt < particleVars.size(); cnt++ ) {

      size_t wrote = -1;

      if( particleVars[cnt].data != NULL ) {

        // wrote = fwrite( &particleVars[cnt].data[particle], sizeof(float), 1, out );
		// cout << "cnt: " << cnt << " " << particleVars[cnt].name.c_str() << " " << particleVars[cnt].data[particle] << "\n";
		// if (cnt == 1)
		// 	varData.volume = particleVars[cnt].data[particle];
		// else if (cnt == 2)
		// 	varData.stress = particleVars[cnt].data[particle];

		nameVal nameValObj;
		
		nameValObj.name = particleVars[cnt].name;
		nameValObj.value = particleVars[cnt].data[particle];

		dataRef.push_back(nameValObj);	
		
		// With tensor data we also have the Trace value and data != NULL in that case
		if ( particleVars[cnt].type == 2 ) { // Tensor data 
		  tenVal tenValObj;
		  matrixVec* matrixRepPtr = particleVars[cnt].matrixRep;
		  matrixVec& matrixRepRef = *(matrixRepPtr);
		  
		  tenValObj.name = particleVars[cnt].name;
		  
		  for (unsigned int i = 0; i < 3; i++) {
		    for (unsigned int j = 0; j < 3; j++) {
			  tenValObj.mat[i][j] =  matrixRepRef[particle](i, j);
			}
	      }		  
			  	  
		  tenDataRef.push_back(tenValObj);
		}
        
      } else {
        // wrote = fwrite( &particleVars[cnt].x[particle], sizeof(float), 1, out );
        // wrote = fwrite( &particleVars[cnt].y[particle], sizeof(float), 1, out );
        // wrote = fwrite( &particleVars[cnt].z[particle], sizeof(float), 1, out );

		if( particleVars[cnt].name == "p.x" ) {
		  varData.x = particleVars[cnt].x[particle];
		  varData.y = particleVars[cnt].y[particle];
		  varData.z = particleVars[cnt].z[particle];
		}
		else if ( particleVars[cnt].type == 1 ) { // Vector data
		  vecVal vecValObj;
		 
		  vecValObj.name = particleVars[cnt].name;
		  vecValObj.x = particleVars[cnt].x[particle];
		  vecValObj.y = particleVars[cnt].y[particle];
		  vecValObj.z = particleVars[cnt].z[particle];  

		  vecDataRef.push_back(vecValObj);
		}
		
        // cout << particleVars[cnt].z[particle] << " " << varData.z << endl;
	  }

      /*if( wrote != 1 ) {
        cerr << "ERROR: Wrote out " << wrote << " floats instead of just one...\n";
        fclose(out);
        return;
      }*/
    }
	
	varColln.push_back(varData);
    // cout << endl << endl;
  }
  // fclose(out);
  
  // cout << "\nDone writing out particle NRRD: " << filename << ".{raw,nhdr}\n\n";
}


///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.  This
// function is never called, but forces the compiler to instantiate
// the needed templated functions.

void
templateInstantiationForParticlesCC()
{
  QueryInfo  * qinfo = NULL;
  int matlNo = 0;
  bool matlClassfication = false;

  handleParticleData<int>    ( *qinfo, matlNo, matlClassfication );
  handleParticleData<long64> ( *qinfo, matlNo, matlClassfication );
  handleParticleData<float>  ( *qinfo, matlNo, matlClassfication );
  handleParticleData<double> ( *qinfo, matlNo, matlClassfication );
  handleParticleData<Point>  ( *qinfo, matlNo, matlClassfication );
  handleParticleData<Vector> ( *qinfo, matlNo, matlClassfication );
  handleParticleData<Matrix3>( *qinfo, matlNo, matlClassfication );
}
