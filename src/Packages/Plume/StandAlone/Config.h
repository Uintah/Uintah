#ifndef Config_h
#define Config_h

#include <string>

#include <loki/Singleton.h>
#include <boost/program_options.hpp>


namespace po = boost::program_options;

namespace Dugway {

  class Config
  {
  public:
    Config();

    bool init(int char *[]);
    bool inform();

  public:
    std::string framework;
    std::pair<std::string, std::string> > builder;

  private:
    po::variables_map _config;
    po::options_description _visible;
  };

  typedef Loki::SingletonHolder<Options> ProgramOptions;
};


#endif // Config_h
