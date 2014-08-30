#include <tabprops/StateTable.h>

using namespace std;

// jcs not sure why, but without this the serialization crashes...
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;

//--------------------------------------------------------------------
string getp(int& i, int argc, char** args)
{
  string a="--";
  if (i < argc-1) {
    a = string(args[i+1]);
  }
  if (a[0] == '-') {
    a = "<missing>";
  }
  else {
    i += 1;
  }
  return a;
}
//--------------------------------------------------------------------
void show_help()
{
  cout << endl
      << " options:" << endl
      << "   -i <file name>  specify the name of the input file (required)" << endl
      << endl;
}
//--------------------------------------------------------------------
int main( int argc, char** argv )
{
  string inputFileName = "";
  if( argc == 1 ){
    show_help();
    return -1;
  }

  int i=1;
  while (i < argc) {
    const string arg = string(argv[i]);
    if (arg == "-i") {
      inputFileName = getp(i,argc,argv);
    }
    else if( arg == "-h" || arg == "-H" ){
      show_help();
    }
    else {
      cout << endl << endl << "unknown option:" << arg << endl;
      show_help();
      exit(-1);
    }
    ++i;
  }

  if( inputFileName == "" ){
    cout << "ERROR: No input file was specified!" << endl;
    show_help();
    return -1;
  }

  try{
    cout << "Loading table: " << inputFileName << endl;
    StateTable table;
    table.read_table( inputFileName );
    table.output_table_info(cout);
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
    return -1;
  }
  return 0;
}
