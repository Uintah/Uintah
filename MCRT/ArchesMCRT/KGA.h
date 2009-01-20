#ifndef KGA_H
#define KGA_H

using namespace std;

// this class is to generate data for k vs g and a vs g
// First , we have this k-g function for gas mixture
// Second, generate RgDmixture vs gUni table, so that given any Rg, we can look up for g
// Last, after looking up to get g, use k-g function to get k

class KGA{

public:
  
  double *gUni;
  double *RgDmixture;
  
  // calculate k vs gUni
  double *kgUni;

  // calcualte a vs gUni
  double *agUni;

  // calculate Rg
  double *fracA ;

  double *pwrg;
  
  KGA(const int &gNo_, const double &pwr);

  ~KGA();

  		       
  // generate RgDmixture vs gUni table
  void get_RgDmixture(const double *kgp,const double *kgpzone2,
		      const double *kgpzone3, const double *kgpzone4,
		      const double &klb, const double &kub,
		      const double &gExlb, const double &gExub,
		      const double &g2ndlast);
    
  // given any Rg, to get g
  double get_gDmixture(const double &Rg);

  // given any gg , calculate k from k-g function
  double get_kDmixture(const double *kgp,
		       const double &gg);

  // given any gg, calculate k from a-g function
  double get_aDmixture(const double *kgp,
		       const double &gg);

  double get_kDmixtureZone1(const double &klb);
  
  double get_kDmixtureZone2(const double *kgpzone2,
			    const double &gg);

  double get_kDmixtureZone3(const double *kgpzone3,
			    const double &gg);

  double get_kDmixtureZone4(const double *kgpzone4,
			    const double &gg);

  void get_RgDmixtureTable(const double *kgVoltest);

  
  double get_gDmixtureTable(const double &Rg, const double *kgVoltest);
  
  double get_kDmixtureTable(const double &gg,
			    const int &ggNo,
			    const double *kgVoltest);

  
  double get_kTDmixtureTable(const double &gg,
			     const int &ggNo,
			     const double *kgaVoltest);
  
  
  void get_RgTDmixtureTable(const double *kgaVoltest);

  double get_aTDmixtureTable(const double &gg,
			     const int &ggNo,
			     const double *kgaVoltest);
    
  double get_gTDmixtureTable(const double &Rg, const double *kgaVoltest);
  
  friend class MakeTableFunction;
  
private:
  
  int gNo; 


  
};


#endif
