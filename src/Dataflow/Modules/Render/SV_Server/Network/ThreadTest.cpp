#include <Thread/Thread.h>
#include <Thread/Runnable.h>
#include <Thread/Mailbox.h>
#include <iostream.h>
#include <unistd.h>

using namespace SemotusVisum::Thread;

class foo;

class bar : public Runnable {
public:
  bar( Mailbox<int>& m ) : mbox(m) {
    //Thread * t = new Thread( this, "Thread" );
    //t->detach();
    //t->join();
  }
  ~bar() {
    cerr << "Destructor B called" << endl;
    
  }

  void deallocate() {
    
  }
  
  virtual void run(); 

protected:
  Mailbox<int> &mbox;
};


class foo {
  friend class bar;
public:
  foo() : m( "mail", 10 ) {
    b = new bar( m );
    t = new Thread( b, "Thread" );
    //t->join();
    //t->detach();
  }
  
  ~foo() {
    cerr << "Destructor F called" << endl;
    t->join();
  } 

  void doit() {
    int i;
    i = m.receive( );
    return;
  }
  
protected:
  bar *b;
  Thread *t;
  Mailbox<int> m;
};

void
bar::run()
{
  cerr << "Running!" << endl;
  sleep( 2 );
  cerr << "Sending message!" << endl;
  int mess = 1;
  mbox.send( mess );
  //sleep( 10 );
  //cerr << "End of running! - shouldn't see me!" << endl;
  cerr << "Bye bye...";
}

main() {


  foo *t = new foo;

  t->doit();

  delete t;
  cerr << "Is foo dead yet?" << endl;
  sleep( 10 );
  Thread::exitAll( 0 );
}
 

