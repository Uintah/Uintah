#include <Config.h>

#include<iostream>
#include<fstream>

#include <plume.defs>

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
	("framework", po::value<string>(&framework)->default_value(""), "master framework")
	("builder", po::value<pair<string,string> >(&builder)->default_value(make_pair("TxtBuilder", "cca:SCIRun.TxtBuilder")), "Builder")
	;
      
      // config file only
      po::options_description file_only("Configurations");
      file_only.add_options()
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

      ifsteeam ifs(".plumerc");
      po::store( po::parse_config_file(ifs, config_file), _config );

      // done
      po::notify(_config);
    }
    catch(const exception &e) {
      cerr << "config: " << e.what() << endl;
      return false;
    }

  };

  bool Config::inform()
  {
    if (_config.count("help")) {
      cout << _visible << endl;
      return true;
    }

    if ( _config.count("version")) {
      count << "Plume version " << PLUME_VERSION << endl;
      return true;
    }

    return false;
  }
}    
    

	
      
