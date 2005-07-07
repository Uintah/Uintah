#ifndef Config_h
#define Config_h

#include <string>

#include <loki/Singleton.h>
#include <boost/program_options.hpp>


namespace po = boost::program_options;

  std::ostream& operator<<( std::ostream& out, const std::pair<std::string, std::string>& pair )
  {
    out << "(" << pair.first << "," << pair.second << ")";
    return out;
  }

namespace Dugway {

  
//   tamplete< typename First=std::string, typename Second=std::string>
//   class Pair : public std::pair<First,Second>
//   {
//   public:
//     Pair( const First &first, const Second second ): pair(first,second) {}
//   }

  std::istream& operator>>( std::istream& in, std::pair<std::string, std::string>& pair )
  {
    std::string s;
    in >> s;
    pair.first = s;
    in >> s;
    pair.second = s;
    //in >> pair.first >> pair.second;
    return in;
  }

  class Config
  {
  public:
    Config();

    bool init(int, char *[]);
    bool inform();

  public:
    std::string framework;
    std::pair<std::string, std::string > builder;

  private:
    po::variables_map _config;
    po::options_description _visible;
  };

  typedef Loki::SingletonHolder<Config> ProgramOptions;
};


#endif // Config_h
