
// Emacs: -*-compile-command: "c++ -o example -I/usr/lib/ruby/1.6/i686-linux -L/usr/lib/ruby/1.6/i686-linux example.cpp rubyeval.cpp -lruby -ldl -lcrypt" -*-

#include "rubyeval.h"
#include "ruby.h"

int main( int argc, char *argv[] ) {

  RubyEval& ruby = *RubyEval::instance();

  ruby.eval("puts 'hello ruby'");

  assert( NUM2INT( ruby.eval("1+1") ) == 2 );

  assert(RubyEval::val2str(ruby.eval("'Regexp'.gsub(/x/, 'X')")) == "RegeXp");

  return 0;
}
