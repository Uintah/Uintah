#include <Packages/Plume/StandAlone/Config.h>

#include<iostream>
#include<fstream>

//#include <plume.defs>

#define PLUME_VERSION 0.1

namespace Dugway {

  using namespace std;

  Config::Config()
  {
  }


  bool Config::init( int argc, char *argv[] ) 
  {
    try {
      // command line only
      po::options_description cmd_only("CommandLine Options");
      cmd_only.add_options()
	("version,v", "print version")
	("help",      "this message")
	;

      // command line & config file 
      po::options_description config("Configurations");
      config.add_options()
	("server,s", "run as a server")
	("framework,f", po::value<string>(&framework)->default_value("socket://localhost.localdomain:32780/142355952"), "master framework")
	("builder,b", po::value<string>(&builder)->default_value(""), "Builder")
	;
      
      // config file only
      po::options_description file_only("Configurations");
      file_only.add_options()
	("default_builder", po::value<string>(&default_builder)->default_value("TxtBuilder"))
 	("builder_types", po::value<map<string,string> >(&builder_types))//->default_value(vector<make_pair("TxtBuilder", "cca:SCIRun.TxtBuilder")))
	;

      //
      // organize options
      po::options_description cmdline;
      cmdline.add(cmd_only).add(config).add(file_only);
      
      po::options_description config_file;
      config_file.add(config).add(file_only);

      _visible.add(cmd_only).add(config);

      // parse
      po::store( po::command_line_parser(argc,argv).
		 options(cmdline).run(),
		 _config );

      ifstream ifs(".plumerc");
      po::store( po::parse_config_file(ifs, config_file), _config );

      cerr << "config read file\n";
      // done
      po::notify(_config);
    }
    catch(const exception &e) {
      cerr << "config: " << e.what() << endl;
      return false;
    }

    return !inform();
  };

  bool Config::inform()
  {
    if (_config.count("help")) {
      cout << _visible << endl;
      return true;
    }

    if ( _config.count("version")) {
      std::cout << "Plume version " << PLUME_VERSION << endl;
      return true;
    }

    if ( _config.count("server") ) 
      is_server = true;
    
    return false;
  }
}    
    

	
      
