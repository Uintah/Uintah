#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>

using namespace std;

int numTimesteps; //number of timesteps around each desired timestep to be averaged

vector<string> argList; //global string for argList so it can be passed outside main

vector<long double> readData (string sourceFile, int numColumns, int desColumn) {
  vector<long double> combColumn;
  vector<long double> outColumn; //vectors for each column and a combined column and the output vector for return

  //file name is command line arg / the sourceFile being passed in
  string infile_name = argList[0] + "/" + sourceFile; 
  ifstream source(infile_name.c_str());

  if(!source) {
     cerr << "File " <<infile_name << " can't be opened." << endl;
  }


  double c1,c2,c3,c4,c5; //indiv cells being read

  int readPos = source.tellg(); //read position prior to reading in the character
  char c;
  c= source.get(); //read the first character as char c
  //if c does not already equal a '#' return to the read position; avoids skipping chars when reading
  if (c != '#') { 
    source.seekg(readPos);
  }

  while (c == '#') { //while the lines are starting with # ignore
    source.ignore(10000, '\n'); //ignore for 10000 characters or until newline is found
    readPos = source.tellg(); //reread the position prior to reading the character in
    c= source.get(); //read first char
    if (c != '#') { //identical, if c not equal to '#' return to read position
      source.seekg(readPos);
    }
  }

  while(source >> c1  >> c2 >> c3 >> c4 >> c5) {
    combColumn.push_back(c1); //pushback each cell into combinedcolumn
    combColumn.push_back(c2);
    combColumn.push_back(c3);
    combColumn.push_back(c4);
    combColumn.push_back(c5);
  }

  /* if we put each cell into an indiv column vector it would work perfectly fine for a 
   * source with 5 columns but for anything else it would skip through the columns if 
   * there are any less (i.e. in a 3 column page,col 1 and 2 would be passed into vectors
   * for col 4 and 5), so we put them all into one combined column that we can skip over 
   * and get the actual column given an input of the desired column and num of columns*/

  //go through combined column starting at the desired column and skipping through based on num columns
  for (int i = desColumn-1; i < combColumn.size(); i+=numColumns) {
    outColumn.push_back(combColumn[i]);
  }
  //output this to outColumn and return
  return outColumn;
}

vector<int> determineTSs () {
  //read time and pressure data
  vector<long double> time = readData ("TimePressure.dat", 5, 2);
  vector<long double> pressure = readData ("TimePressure.dat", 5, 3);
  vector<int> newTimesteps; //initialize variable to pushback and return

  for (int i = 0; i < time.size(); i++) { //index through entirety of time vector
    //in other words, if current index is a whole number and the last index was a whole number
        //and the next index is not a whole number and pressure two indexs ago is the same 
	//because of the repeating timestep issue
	//then it is a changing point and a desired timestep
    if((ceil(pressure[i]) == floor(pressure[i])) && (ceil(pressure[i-1]) == floor(pressure[i-1])) 
		    && (ceil(pressure[i-2]) == floor(pressure[i-2])) && (ceil(pressure[i+1]) != floor(pressure[i+1]))) {
      //if found, push back newTimesteps with rounded version of time at found index
      newTimesteps.push_back(floor(time[i]));  
    }
  }
  newTimesteps.push_back(time[time.size()-1]); //add the last endpoint since this wouldn't be found

  return newTimesteps; //return new vector
}

int main(int argc, char *argv[]) { //pass in command line args

  vector<string> argListTemp(argv + 1, argv + argc); //convert to temp string
  argList = argListTemp; //pass into global vector
  vector<int> timesteps = determineTSs(); //determine desired timesteps to examine using procedure
  int numTimesteps = timesteps.size(); 
  cout << "newTimesteps are " << endl; //print determined timesteps
  for (int i=0; i <numTimesteps; i++) {
    cout << timesteps.at(i) << endl; 
  } 
  cout.precision(8);
  double averagedX_Results[numTimesteps]; //variable array for storing averaged x_force results
  double Z_Results[numTimesteps]; //variable array for storing z_force results
  double PistonPos_Results[numTimesteps]; //variable array for storing piston pos results
  double Pressure_Results[numTimesteps]; //variable array for storing piston pressure results

  //reading x_forces data
  vector<long double> time = readData ("BndyForce_xplus.dat", 4, 1);
  vector<long double> forcePlus = readData ("BndyForce_xplus.dat", 4, 2);
  vector<long double> forceMinus = readData ("BndyForce_xminus.dat", 4, 2);

  cout << "The avg x-force at their respective timesteps are " << endl;
  for (int ts = 0; ts < numTimesteps; ts++) { //run through every timestep
    double timereq = timesteps[ts]; //time required is current timestep
    double t = 0;
    int i = 0;
    while (t < timereq) { //go through time vector/column until timestep required is found
      t = time[i];
      i++;
    }
    int i_req = i; //pass current index as the required 

    //take average of plus and minus face forces within desired num timesteps
    double sumPlus = 0; 
    double sumMinus = 0; 
    for (int j = i_req-numTimesteps; j < i_req; j++) {
      sumPlus+= forcePlus[j];
      sumMinus+= forceMinus[j];
    }
    double avgPlus = sumPlus/numTimesteps;
    double avgMinus = sumMinus/numTimesteps;

    averagedX_Results[ts] = (abs(avgPlus)+abs(avgMinus))/2; //result is average of averages
    cout << averagedX_Results[ts] << ", ";
  } //repeat for timesteps
  cout << endl;

//reading z_forces data 
time = readData ("BndyForce_zminus.dat", 4, 1);
vector<long double> force = readData ("BndyForce_zminus.dat", 4, 4);

cout << "The z-force at their respective timesteps are " << endl;
  for (int ts = 0; ts < numTimesteps; ts++) { //same ideas as above
    double timereq = timesteps[ts];
    double t = 0;
    int i = 0;
    while (t < timereq) {
      t = time[i];
      i++;
    }
    int i_req = i;

    //take average within desired num timesteps
    double sum = 0;
    for (int j = i_req-numTimesteps; j < i_req; j++) { 
      sum+= force[j];
    }
    Z_Results[ts] = sum/numTimesteps; //pass on
    cout << Z_Results[ts] << ",";
  }
  cout << endl;

//reading position of piston bottom
time = readData ("Time_YMean.dat", 2, 1); //changed from Time_DOP to Time_YMean for Mean piston height
vector<long double> position = readData ("Time_YMean.dat", 2, 2);

cout << "The mean piston pos  at their respective timesteps are " << endl; //changed to mean
  for (int ts = 0; ts < numTimesteps; ts++) { //identical
    double timereq = timesteps[ts];
    double t = 0;
    int i = 0;
    while (t < timereq) {
      i++;
      t = time[i];
      if (t == 0 || i == time.size()) {
	i--;
	break;
      }
    }
    int i_req = i;

    //no need for taking averaged result bc small sample already processed so pass on immediate i
    //check if piston pos greater than last one, if it is return an index
    //IF PISTON POSITION INCREASE IS EXPECTED, CHANGE 1 TO 0 TO TURN THIS IF STATEMENT OFF 
#if 1
    if (ts == (numTimesteps-1) && position[i_req] > PistonPos_Results[ts-1]) {
      cout << "Position greater than last, returning one index" << endl;
      i_req--;
    }
#endif

    PistonPos_Results[ts] = position[i_req]-0.01;
    cout << PistonPos_Results[ts] + 0.01 << ", ";
  }
  cout << endl;

time = readData ("TimePressure.dat", 5, 2);
vector<long double> pressure = readData ("TimePressure.dat", 5, 3);

cout << "The piston pressure  at their respective timesteps are " << endl;
  for (int ts = 0; ts < numTimesteps; ts++) { //identical
    double timereq = timesteps[ts];
    double t = 0;
    int i = 0;
    while (t < timereq) {
      i++;
      t = time[i];
      if (i == time.size()) {
	i--;
	break;
      }
    }
    int i_req = i;

    Pressure_Results[ts] = -1 * round(pressure[i_req]); //rounding and making positive
    cout << Pressure_Results[ts] << ", ";
  }
  cout << endl;

//doing math on results for moduli and mean press
  long double Y_Stress [numTimesteps];
  long double X_Stress [numTimesteps];
  long double Z_Stress [numTimesteps];
  long double Mean_Press [numTimesteps];
  long double delta_Height [numTimesteps];
  long double delta_Axial_Strain [numTimesteps];
  long double delta_Y_Stress [numTimesteps];
  long double Lateral_Stress [numTimesteps];
  long double Beta [numTimesteps];
  long double Constrained_Modulus [numTimesteps];
  long double Poissons_Ratio [numTimesteps];
  long double Youngs_Modulus [numTimesteps];
  long double Shear_Modulus [numTimesteps];
  long double Bulk_Modulus [numTimesteps];

  for (int ts = 0; ts < numTimesteps; ts++) {
    Mean_Press[ts] = (Pressure_Results[ts]  + 
                      averagedX_Results[ts]/(.005*PistonPos_Results[ts]) + 
                      Z_Results[ts]/(.1*PistonPos_Results[ts]))/3.;
  }

  for (int ts = 0; ts < numTimesteps-1; ts++) {
    Mean_Press[ts] = (Pressure_Results[ts]  + 
                      averagedX_Results[ts]/(.005*PistonPos_Results[ts]) + 
                      Z_Results[ts]/(.1*PistonPos_Results[ts]))/3.;
    Y_Stress[ts] = 0.5*(Pressure_Results[ts]+Pressure_Results[ts+1]);
    X_Stress[ts] = 0.5*(averagedX_Results[ts]+averagedX_Results[ts+1])
                  /(.005*PistonPos_Results[ts]);
    Z_Stress[ts] = 0.5*(Z_Results[ts]+Z_Results[ts+1])
                      /(.1*PistonPos_Results[ts]);
  }

  for (int ts = 0; ts < numTimesteps-1; ts++) {
    delta_Height[ts] = PistonPos_Results[ts]- PistonPos_Results[ts+1];
    delta_Y_Stress[ts] = Pressure_Results[ts+1] - Pressure_Results[ts];
    cout << "PistonPos_Results[" << ts << "] = " << PistonPos_Results[ts] << endl;
    cout << "delta_Height[" << ts << "] = " << delta_Height[ts] << endl;
    cout << "delta_Y_Stress[" << ts << "] = " << delta_Y_Stress[ts] << endl;
//  }

//  for (int ts = 0; ts < numTimesteps-1; ts++) {
    delta_Axial_Strain[ts] = delta_Height[ts]/PistonPos_Results[ts];
    cout << "delta_Axial_Strain[" << ts << "] = " << delta_Axial_Strain[ts] << endl;
    Lateral_Stress[ts] = 0.5*(X_Stress[ts] + Z_Stress[ts]);
    cout << "Vertical_Stress[" << ts << "] = " << Y_Stress[ts] << endl;
    cout << "Lateral_Stress[" << ts << "] = " << Lateral_Stress[ts] << endl;
    Beta[ts] = Lateral_Stress[ts]/Y_Stress[ts]; //50 is the change in y_stress
    cout << "Beta[" << ts << "] = " << Beta[ts] << endl;
    Constrained_Modulus[ts] = (delta_Y_Stress[ts])/delta_Axial_Strain[ts];
    cout << "Constrained_Modulus[" << ts << "] = " << Constrained_Modulus[ts] << endl;
    Poissons_Ratio[ts] = Beta[ts]/(Beta[ts]+1);
    cout << "Poissons_Ratio[" << ts << "] = " << Poissons_Ratio[ts] << endl;
    Youngs_Modulus[ts] = (Constrained_Modulus[ts]*(Poissons_Ratio[ts]+1)*
                         (-2*Poissons_Ratio[ts]+1))/(-Poissons_Ratio[ts]+1);
    cout << "Youngs_Modulus[" << ts << "] = " << Youngs_Modulus[ts] << endl;
    Shear_Modulus[ts] = Youngs_Modulus[ts]/(2.*(1.+Poissons_Ratio[ts]));
    cout << "Shear_Modulus[" << ts << "] = " << Shear_Modulus[ts] << endl;
    Bulk_Modulus[ts] = Youngs_Modulus[ts]/(3.*(1.-2.*Poissons_Ratio[ts]));
    cout << "Bulk_Modulus["  << ts << "] = " << Bulk_Modulus[ts]  << endl << endl;
  }

  cout << "Modulus calculations complete" << endl;

  //Writing modulus values to a txt file for inplotModuli to use
  ofstream fwLo("loadModuli",   ofstream::out); //open file for writing
  ofstream fwUL("unloadModuli", ofstream::out); //open file for writing
  ofstream fwRL("reloadModuli", ofstream::out); //open file for writing
  ofstream fw("parsedModuli",   ofstream::out); //open file for writing
  cout << "File parsedModuli created; beginning to write" << endl;
  if (fw.is_open()) {
    fw   << "#VertStress Pressure MeanVertStress MeanPressure ShearMod BulkMod" << endl;
    fwLo   << "#VertStress Pressure MeanVertStress MeanPressure ShearMod BulkMod" << endl;
    fwUL   << "#VertStress Pressure MeanVertStress MeanPressure ShearMod BulkMod" << endl;
    fwRL   << "#VertStress Pressure MeanVertStress MeanPressure ShearMod BulkMod" << endl;
    double maxMeanPress = Mean_Press[0];
    for (int ts = 0; ts < numTimesteps-1; ts++) {
      fw << roundf(Pressure_Results[ts]*100)/100 << "\t";
      fw << roundf(Mean_Press[ts]*100)/100 << "\t";
      fw << roundf(Y_Stress[ts]*100)/100 << "\t";
      fw << roundf(0.5*(Mean_Press[ts]+Mean_Press[ts+1])*100)/100 << "\t";
      fw << roundf(Shear_Modulus[ts]*100)/100 << "\t";
      fw << roundf(Bulk_Modulus[ts]*100)/100 << endl;
      if(Pressure_Results[ts+1]>Pressure_Results[ts]){
        if(Mean_Press[ts+1] > 1.1*maxMeanPress){
          fwLo << roundf(Pressure_Results[ts]*100)/100 << "\t";
          fwLo << roundf(Mean_Press[ts]*100)/100 << "\t";
          fwLo << roundf(Y_Stress[ts]*100)/100 << "\t";
          fwLo << roundf(0.5*(Mean_Press[ts]+Mean_Press[ts+1])*100)/100 << "\t";
          fwLo << roundf(Shear_Modulus[ts]*100)/100 << "\t";
          fwLo << roundf(Bulk_Modulus[ts]*100)/100 << endl;
          maxMeanPress = Mean_Press[ts+1];
        } else {
          fwRL << roundf(Pressure_Results[ts]*100)/100 << "\t";
          fwRL << roundf(Mean_Press[ts]*100)/100 << "\t";
          fwRL << roundf(Y_Stress[ts]*100)/100 << "\t";
          fwRL << roundf(0.5*(Mean_Press[ts]+Mean_Press[ts+1])*100)/100 << "\t";
          fwRL << roundf(Shear_Modulus[ts]*100)/100 << "\t";
          fwRL << roundf(Bulk_Modulus[ts]*100)/100 << endl;
        }
      } else {
        fwUL << roundf(Pressure_Results[ts]*100)/100 << "\t";
        fwUL << roundf(Mean_Press[ts]*100)/100 << "\t";
        fwUL << roundf(Y_Stress[ts]*100)/100 << "\t";
        fwUL << roundf(0.5*(Mean_Press[ts]+Mean_Press[ts+1])*100)/100 << "\t";
        fwUL << roundf(Shear_Modulus[ts]*100)/100 << "\t";
        fwUL << roundf(Bulk_Modulus[ts]*100)/100 << endl;
      }
    }
    cout << "Completed writing modulus values to parsedModuli" << endl;
    fw.close();
  } else cout << "Problem opening and writing file parsedModuli";
}
