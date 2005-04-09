#include <Datatypes/KludgeMessage.h>
#include <Malloc/Allocator.h>
#include <Classlib/String.h>

static Persistent* make_KludgeMessage()
{
  return scinew KludgeMessage;
}

KludgeMessage::KludgeMessage()
{

}

KludgeMessage::KludgeMessage(KludgeMessage &copy)
  :src_mag(copy.src_mag),src_loc(copy.src_loc)
{
  
}

KludgeMessage* KludgeMessage::clone()
{
  return scinew KludgeMessage(*this);
}

KludgeMessage::~KludgeMessage()
{

}


PersistentTypeID KludgeMessage::type_id("KludgeMessage","Datatype",make_KludgeMessage);

const int KludgeMessage_VERSION = 0;

void KludgeMessage::io(Piostream& stream)
{
  int version = stream.begin_class("KludgeMessage", KludgeMessage_VERSION);
  
  Pio(stream,src_mag);
  Pio(stream,src_loc);

  Pio(stream,surf_pots);
  Pio(stream,surf_pti);
  Pio(stream,surf_wts);

  Pio(stream,src_recs);
#if 0
  cerr << src_mag.size() << " - ";
  cerr << src_loc.size() << " - ";

  cerr << surf_pots.size() << " - ";
  cerr << surf_pti.size() << " - ";
  cerr << surf_wts.size() << " - ";

  cerr << src_recs.size() << endl;
#endif  
  stream.end_class();
}



static Persistent* make_AmoebaMessage()
{
  return scinew AmoebaMessage;
}

AmoebaMessage::AmoebaMessage()
{

}

AmoebaMessage::AmoebaMessage(AmoebaMessage &copy)
  :amoebas(copy.amoebas)
{
  
}

AmoebaMessage* AmoebaMessage::clone()
{
  return scinew AmoebaMessage(*this);
}

AmoebaMessage::~AmoebaMessage()
{

}


PersistentTypeID AmoebaMessage::type_id("AmoebaMessage","Datatype",make_AmoebaMessage);

const int AmoebaMessage_VERSION = 0;

void AmoebaMessage::io(Piostream& stream)
{
  int version = stream.begin_class("AmoebaMessage", AmoebaMessage_VERSION);
  
  Pio(stream,amoebas);

  //cerr << amoebas.size() << " DOing Amoeba\n";

  stream.end_class();
}

void Pio(Piostream &stream, AmoebaRecord& r) 
{
  stream.begin_cheap_delim();
  
  Pio(stream,r.sources);
  Pio(stream,r.generation);

  stream.end_cheap_delim();
}

void Pio(Piostream &stream, SourceRecord& s)
{
  stream.begin_cheap_delim();

  Pio(stream,s.loc);
  Pio(stream,s.theta);
  Pio(stream,s.phi);

  Pio(stream,s.v);

  Pio(stream,s.err);

  stream.end_cheap_delim();
}
