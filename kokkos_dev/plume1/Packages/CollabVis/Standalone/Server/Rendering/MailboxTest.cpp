#include <iostream.h>
#include <unistd.h>
#include <Thread/Mailbox.h>
#include <Thread/Runnable.h>

using namespace SemotusVisum::Thread;

template<class Item> class MailboxTest : public Runnable {
public:
  MailboxTest(Mailbox<Item>& m) : mbox(m) {}
  ~MailboxTest() {}

   void run() {
     Item foo;
     
     while (1) {
       foo = mbox.receive();
       cerr << "Got message: " << foo << endl;
     }
   }
   
private:
  Mailbox<Item>& mbox;
};

void foo() {
  cerr << "foo" << endl;
}

main() {
  Mailbox<int> m( "mailbox", 10 );
  MailboxTest<int> mt( m );
  Thread *t = new Thread( &mt, "mailboxtestthread");
  t->detach();
  
  for (int i = 0; i < 3; i++) {
    m.send( i );
    sleep( 1 );
  }
  Thread::exitAll(0);
}
