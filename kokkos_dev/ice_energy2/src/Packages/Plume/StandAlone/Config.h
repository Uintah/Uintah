#ifndef Config_h
#define Config_h

#include <string>
#include <map>
#include <iostream>
#include <loki/Singleton.h>
#include <boost/program_options.hpp>


namespace std {
  template<typename First, typename Second>
  inline std::ostream& operator<<( std::ostream& out, const std::pair<First, Second >& p )
  {
    out << "(" << p.first << "," << p.second << ")";
    return out;
  }
  
  template<typename First, typename Second>
  inline std::istream& operator>>( std::istream& in, std::pair<First, Second >& p )
  {
    in >> p.first >> p.second;
    std::cerr << "pair ("<<p.first<<", "<<p.second<<")\n";
    return in;
  }

  template<typename First, typename Second>
  inline std::istream& operator>>( std::istream& in, std::map<First, Second >& m )
  {
    std::string key,value;
    char ch;

    while (in >> ch && ch == ' ');
    if ( ch == '(' ) {
      while (in >> ch  && ch != ',') key += ch;
      while (in >> ch && ch  == ' ');
      while (ch != ')') { value += ch; in >> ch;}

      //std::cerr << "map add ("<<key<<") ("<<value<<")\n";
      m[key]=value;
    }
    else {
      std::cerr << "error reading map entry\n";
    }
    return in;
  }

}

namespace po = boost::program_options;

namespace Dugway {

  class Config
  {
  public:
    typedef std::map<std::string,std::string> Builders;

  public:
    Config();

    bool init(int, char *[]);
    bool inform();

  public:
    bool is_server;
    std::string framework;
    std::string  builder;
    std::string  default_builder;
    Builders builder_types;

  private:
    po::variables_map _config;
    po::options_description _visible;
  };

  //
  // Singleton: ProgramOptions

  typedef Loki::SingletonHolder<Config> ProgramOptions;
};


#endif // Config_h
