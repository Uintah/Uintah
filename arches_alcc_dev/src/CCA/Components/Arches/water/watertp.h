#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cmath>    
#include <cstring> 

using namespace std;


/** @details  This StandAlone class is used to calculate thermaldynamic
  *           properties of water (only for liquid water and steams but 
  *           not for ice). The Temperature range is from 273.16K to 1200K 
  *           This range is enough for water-steam heat transfer
  */


class H2O_Properties{
    static const double Ttr = 273.16;    //Temperature of Triple point
    static const double Ptr = 611.2e-6;  //Pressure of Triple point (MPa)
    static const double Tc1 = 647.3000;  //Criticle Temperature (K)
    static const double Pc1 = 22.12000;  //Criticle Pressure (MPa)
    static const double Vc1 = 0.00317;   //Critical Specific Volume(m3/kg)
    static const double Rt = 461.51;     //gas constant, R, J/Kg.K)
    static const double Sa1 = (623.15/647.3000);
    static const double Sa2 = (863.15/647.3000);
    static const double Sa3 = (1200/647.3000);  //The original is (1073.15/Tc1)
    static const double Sat = (273.16/647.3000);
    static const double Ba1 = (16.535/22.12000);
    static const double Ba2 = (100/22.12000);
    static const double L = 7.160997524e0;
    static const double Rhoc1 = 317.70;  //critical density (kg/m3)
    static const double Etaxing = 55.071e-6; // dynamic viscosity (Pa.S=kg/(m.s))

public:    
    double Temperature;    
    double Pressure;
    double SV;
    double Enthalpy;
    double Steam_Fraction;   //mass fraction of the steam in the water-steam mixture

    //To get Enthalpy and SV given Pressure(Pa)and Temperature(K)
    //    void waterBaSa(double *steam_fraction,double Ba,double Sa,double *SV,double *Enthalpy);
    void Water_PT(double P, double T, double X);
    // to get Temperature(K) and SV(m3/kg) given Pressure(Pa) and Enthalpy(J/kg)
    void p_waterBah(double P,double H);
    //to calculate dynamic viscosity(Kg/(m.s))
    double Dynamic_Viscosity(double Tem, double Rho);
    //to calculate thermal conductivity, W/(m.K)
    double Conductivity(double Tem,double Rho);
    //To calculate Heat capacity(J/(Kg.K)
    double Heat_Cp(double Tem,double p,double steam_fraction);

    void set_PT(double P, double T); //set Temperature(K) and pressure(Pa)
    void set_PH(double P, double E);   //set Pressure(Pa) and Enthalpy(J/kg)
    void set_SteamX(double X);

private:

    double Kfun(double Sa);         //Kfunction, to get K boundary pressure based on Temperature
    double Lfun(double Sa);         //L-function, to get L boundary pressure given Temperature
    double Lpiefun(double Sa);      //derivate of Lfun
    double p_Kfun(double Ba);       //reverse function of Kfun
    double p_Lfun(double Ba);       //reverse function of Lfun

    //properties for The first block
    double water1_Sa(double Ba,double Enthalpy); //get Tem based on Pressure and Enthalpy
    double water1_SV(double Ba,double Sa);  //get SV given Tem and Pressure
    double water1_H(double Ba,double Sa);  //get Enthalpy given Temperaure and Pressure
	
    //properties for The second block
    double water2_Sa(double Ba,double Enthalpy,double Sal,double Bal,double Balpie); //p h to Tem
    double water2_SV(double Ba,double Sa,double Bal,double Balpie);  //p t to SV
    double water2_H(double Ba,double Sa,double Bal,double Balpie);  //p t to Enthalpy
	
    //properties for the third block
    double water3_1(double x,double Sa); //v Tem to Pressure
    double water3_SV(double Ba, double Sa); //竝 t to sv
    double water3_H(double x,double Sa);   //p t to Enthalpy
    double water3_Sa(double Ba,double Enthalpy,double Sak,double Sal);  // p h to Tem

    //properties for the fouth block
    double water4_1(double x,double Sa);  //v t to p
    double water4_SV(double Ba, double Sa); //p t to SV
    double water4_H(double x,double Sa); //v,t to Enthalpy
    double water4_Sa(double Ba,double h,double Sak);//p h to Temperature
};




void H2O_Properties::set_PT(double P, double T) //set Temperature(K) and pressure(Pa)
{
    Pressure = P;
    Temperature = T;
}

void H2O_Properties::set_PH(double P, double E)   //set Pressure(Pa) and Enthalpy(J/kg)
{
    Pressure = P;
    Enthalpy = E;
}
void H2O_Properties::set_SteamX(double X)
{
    Steam_Fraction = X;
}



// to calculate SV, Enthalpy given Temperature(K) and Pressure(MPa)
//void H2O_Properties::Water_PT(double *Steam_Fraction,double Pressure,double Temperature,double *SV,double *Enthalpy)
void H2O_Properties::Water_PT(double P, double T, double X){
    double Ba, Sa;
    double Bal,Balpie;
    double Bak,Sak;
    int I=0;
    double v1,v2,h1,h2;
    //double Steam_Fractionx;

    set_PT(P, T);
    set_SteamX(X);

    Ba=Pressure/Pc1/1e6;  
    Sa=Temperature/Tc1;
    Sak=p_Kfun(Ba);
    if (fabs(Sa-Sak)>0.1/Tc1)	Steam_Fraction=0;

    Bal=Lfun(Sa);
    Balpie=Lpiefun(Sa);
    Bak=Kfun(Sa);
    if((Steam_Fraction)==1||(Steam_Fraction)==0)
    {
        if(Sa>=Sat && Sa<=Sa1)
        {
            if(Ba>=0 && Ba<Bak)
                I=2;
            else  I=1;
        }
        else if(Sa>Sa1 && Sa<1)
        {
            if(Ba>=0 && Ba<=Bal)  I=2;
            else if(Ba>Bal && Ba<Bak) I=3;
            else if(Ba>Bak && Ba<Ba2) I=4;
        }
        else if(Sa>1 && Sa<Sa2)
        {
            if(Ba>=0 && Ba<=Bal) I=2;
            else if(Ba>Bal && Ba<=Ba2) I=3;
        }
        else if(Sa>Sa2 && Sa<Sa3)
        {
            if(Ba>=0 && Ba<=Ba2) I=2;
        }
        else //over range of 1200K
        {
            cout<<"Temperature over range!"<<"\n";
            exit(0);
        }
    }
    if((Steam_Fraction)<1&&(Steam_Fraction)>0)
    {
        if(Sa>=Sat&&Sa<=Sa1) I=6;
        else if(Sa>Sa1&&Sa<1) I=5;
    }
    {

        if(I==1)
        {
            SV=water1_SV(Ba,Sa);
            Enthalpy=water1_H(Ba,Sa);
            Steam_Fraction=0;
            //            cout<<"over the scope!"<<"\n";
            //            exit(0);
        }
        else if(I==2)
        {
            //cout("it is in the second block!\n");
            SV=water2_SV(Ba,Sa,Bal,Balpie);
            Enthalpy=water2_H(Ba,Sa,Bal,Balpie);
            Steam_Fraction=1;
        }
        else if(I==3)
        {
            //cout("it is in the third block!\n");
            SV=water3_SV(Ba,Sa);
            Enthalpy=water3_H(SV,Sa);
            Steam_Fraction=1;
        }
        else if(I==4)
        {
            //cout("it is in the fourth block!\n");
            SV=water4_SV(Ba,Sa);
            Enthalpy=water4_H(SV,Sa);
            Steam_Fraction=1;
        }
        //cout("bital=%f\tbitak=%f\n",Bal*Pc1,Bak*Pc1);
        //cout("kkkk:%f",water3_1(0.006211740/Vc1,Sa)*Pc1);
        else if(I==6)
        {
            v1=water1_SV(Ba,Sa);
            v2=water2_SV(Ba,Sa,Bal,Balpie);
            h1=water1_H(Ba,Sa);
            h2=water2_H(Ba,Sa,Bal,Balpie);
            SV=v1*(1-(Steam_Fraction))+v2*(Steam_Fraction);
            Enthalpy=h1*(1-(Steam_Fraction))+h2*(Steam_Fraction);
        }
        else if(I==5)
        {
            v1=water4_SV(Ba,Sa);
            v2=water3_SV(Ba,Sa);
            h1=water4_H(v1,Sa);
            h2=water3_H(v2,Sa);
            SV=v1*(1-(Steam_Fraction))+v2*(Steam_Fraction);
            Enthalpy=h1*(1-(Steam_Fraction))+h2*(Steam_Fraction);
        }
    }

    SV=(SV)*Vc1;
    Enthalpy=(Enthalpy)*Pc1*Vc1*1000;  // unit is kJ/kg
    Enthalpy *= 1000;   //convert to J/kg
}



// to get Temperature(K) and SV(m3/kg) given Pressure(Pa) and Enthalpy(J/kg)
void H2O_Properties::p_waterBah(double P,double H)
{
    double Ba;
    double Bal,Balpie,Sal;
    double Sak;
    double h41_1,h41_4,h43_3,h43_4,h32_3,h32_2,h12_1,h12_2;
    double v41_1,v41_4,v43_3,v43_4,v32_3,v32_2,v12_1,v12_2;
    int I=0;

    set_PH(P,H);
    Ba=Pressure/Pc1/1e6;  //to get dimensionless pressure
    Enthalpy=Enthalpy/(Pc1*Vc1*1000)/1000.;  //to convert to dimensionless 

    if(Ba<=Ba1)
    {
        Sak=p_Kfun(Ba);// to get K boundary Temperature (saturation temperature) given pressure
        Sal=p_Lfun(Ba);  //to get L boundary Temperature given pressure
        Bal=Lfun(Sal);	
        Balpie=Lpiefun(Sal);
        v12_1=water1_SV(Ba,Sak);				//boundary of 1 and 2 block
        v12_2=water2_SV(Ba,Sak,Bal,Balpie); 
        h12_1=water1_H(Ba,Sak);
        h12_2=water2_H(Ba,Sak,Bal,Balpie);
        //cout("v12_1=%f,\th12_1=%f\nv12_2=%f,\th12_2=%f\n",v12_1*Vc1,h12_1*Pc1*Vc1*1000,v12_2*Vc1,h12_2*Pc1*Vc1*1000);

        //cout("Sal=%f\tBal=%f\n",Sal,Bal);
        if(Enthalpy<h12_1) {
            //cout("It is in the first block!\n");
            Temperature=water1_Sa(Ba,Enthalpy);
            SV=water1_SV(Ba,Temperature);
            Steam_Fraction=0;
        }
        else if(Enthalpy>h12_2){
            //cout("It is in the second block!\n");
            Temperature=water2_Sa(Ba,Enthalpy,Sal,Bal,Balpie);
            SV=water2_SV(Ba,Temperature,Bal,Balpie);
            Steam_Fraction=1;
        }
        else 
        {
            Steam_Fraction=(Enthalpy-h12_1)/(h12_2-h12_1);
            Temperature=Sak;
            SV=Steam_Fraction*v12_2+(1-Steam_Fraction)*v12_1;
        }
    }

    else if(Ba>Ba1)
    {
        if(Ba<=1) Sak=p_Kfun(Ba);
        else if(Ba>1) Sak=1;
        Sal=p_Lfun(Ba);	//L boundary given pressure
        Bal=Lfun(Sal);	
        Balpie=Lpiefun(Sal);

        v41_4=water4_SV(Ba,Sa1);//boundary between 1 and 4
        v41_1=water1_SV(Ba,Sa1);
        h41_4=water4_H(v41_4,Sa1);
        h41_1=water1_H(Ba,Sa1);
        h41_4=(h41_4+h41_1)/2;
        h41_1=h41_4;

        v43_4=water4_SV(Ba,Sak);
        v43_3=water3_SV(Ba,Sak); //boundary between 3 and 4
        h43_4=water4_H(v43_4,Sak);
        h43_3=water3_H(v43_3,Sak);
        if(Sak>=1)
        {
            h43_4=(h43_4+h43_3)/2;
            h43_3=h43_4;
        }
	
        v32_3=water3_SV(Bal,Sal);
        v32_2=water2_SV(Bal,Sal,Bal,Balpie); //boundary between 3 and 2
        h32_3=water3_H(v32_3,Sal);
        h32_2=water2_H(Bal,Sal,Bal,Balpie);
        h32_3=(h32_3+h32_2)/2;
        h32_2=h32_3;

        if(Enthalpy<h41_1) {
            //cout<<"It is in the first block!\n"; 
            Temperature=water1_Sa(Ba,Enthalpy); 
            SV=water1_SV(Ba,Temperature);
            Steam_Fraction=0;
        }
        else if(Enthalpy>h41_4 && Enthalpy<h43_4) {
            //cout<<"It is in the fourth block!\n";
            Temperature=water4_Sa(Ba,Enthalpy,Sak); 
            SV=water4_SV(Ba,Temperature);
            Steam_Fraction=0;
        }
        else if(Enthalpy>h43_3 && Enthalpy<h32_3) {
            //cout<<"It is in the third block!\n";
            Temperature=water3_Sa(Ba,Enthalpy,Sak,Sal); 
            SV=water3_SV(Ba,Temperature);
            Steam_Fraction=1;
        }
        else if(Enthalpy>h32_2) {
            //cout<<"It is in the second block!\n";
            Temperature=water2_Sa(Ba,Enthalpy,Sal,Bal,Balpie);
            SV=water2_SV(Ba,Temperature,Bal,Balpie);
            Steam_Fraction=1;
        }
        else {
            Steam_Fraction=(Enthalpy-h43_4)/(h43_3-h43_4);
            Temperature=Sak;
            SV=(Steam_Fraction)*v43_3+(1-Steam_Fraction)*v43_4;
        }
    }

    Temperature *= Tc1;
    SV *= Vc1;
}


 //to get K boundary pressure based on Temperature
double H2O_Properties::Kfun(double Sa)
{
    double Bak; 
    double k[9]={-7.691234564e0,-2.608023696e1,-1.681706546e2,6.423285504e1,-1.189646225e2,\
                 4.167117320e0,2.097506760e1,1e9,6};
    int v;
    double mid=0;
    for(v=0;v<5;v++)
        mid=mid+k[v]*pow((1-Sa),v+1);
    Bak=exp((1/Sa)*mid/(1+k[5]*(1-Sa)+k[6]*(1-Sa)*(1-Sa))-(1-Sa)/(k[7]*(1-Sa)*(1-Sa)+k[8]));
    return(Bak);
}

//reverse function of Kfun
double H2O_Properties::p_Kfun(double Ba)
{
    double t1=Sat,t2=1,t; 
    double Ba_1=0;
    int i=0;
    while(fabs(Ba_1-Ba)>5e-10 && i<100)
    {
        t=(t1+t2)/2;
        Ba_1=Kfun(t);
        if(Ba_1>Ba) 
            t2=t;
        else
            t1=t;
        i++;
    }
    //cout("deta=%f\ni=%d\n",fabs(Ba_1-Ba),i);
    return(t);
}


//L-function, to get L boundary pressure given Temperature
double H2O_Properties::Lfun(double Sa)
{
    double Bal;
    Bal=((Sa2-Sa)*Ba1+(Sa-Sa1)*Ba2-L*(Sa2-Sa)*(Sa-Sa1))/(Sa2-Sa1);
    return(Bal);
}

//reverse function of Lfun
double H2O_Properties::p_Lfun(double Ba)
{
    double t1=Sa1,t2=Sa2,t;
    double Ba_1=0;
    int i=0;
    while(fabs(Ba_1-Ba)>5e-10 && i<100)
    {
        t=(t1+t2)/2;
        Ba_1=Lfun(t);
        if(Ba_1>Ba) 
            t2=t;
        else
            t1=t;
        i++;
    }
    //cout("deta=%f\ni=%d\n",fabs(Ba_1-Ba),i);
    return(t);
}



double H2O_Properties::Lpiefun(double Sa)
{
    double Balpie;
    Balpie=(Ba2-Ba1-L*(Sa2-2*Sa+Sa1))/(Sa2-Sa1);
    return(Balpie);
}

double H2O_Properties::water1_SV(double Ba,double Sa)
{
    double p,T,x; 
    double pi,tao,gamap=0;
    int I[35]={0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,4,5,8,8,21,23,29,30,31,32};
    int J[35]={0,-2,-1,0,1,2,3,4,5,-9,-7,-1,0,1,3,-3,0,1,3,17,-4,0,6,-5,-2,10,-8,-11,-6,-29,-31,-38,-39,-40,-41};
    double n[35]={0,0.14632971213167e0,-0.84548187169114e0,-0.37563603672040e1,0.33855169168385e1,\
                  -0.95791963387872e0,0.15772038513228e0,-0.16616417199501e-1,0.81214629983568e-3,0.28319080123804e-3,\
                  -0.60706301565874e-3,-0.18990068218419e-1,-0.32529748770505e-1,-0.21841717175414e-1,\
                  -0.52838357969930e-4,-0.47184321073267e-3,-0.30001780793026e-3,0.47661393906987e-4,\
                  -0.44141845330846e-5,-0.72694996297594e-15,-0.31679644845054e-4,-0.28270797985312e-5,\
                  -0.85205128120103e-9,-0.22425281908000e-5,-0.65171222895601e-6,-0.14341729937924e-12,\
                  -0.40516996860117e-6,-0.12734301741641e-8,-0.17424871230634e-9,-0.68762131295531e-18,\
                  0.14478307828521e-19,0.26335781662795e-22,-0.11947622640071e-22,0.18228094581404e-23,\
                  -0.93537087292458e-25};
    int i;
    p=Ba*Pc1;
    T=Sa*Tc1;
    pi=p/16.53;
    tao=1386/T;
    for(i=1;i<35;i++)
    {
        gamap=gamap-n[i]*I[i]*pow(7.1-pi,(I[i]-1))*pow(tao-1.222,J[i]);
    }
    gamap=gamap*pi;
    gamap=gamap*(461.526*T)/(p*1e6);
    x=gamap/Vc1;
    return(x);
}



double H2O_Properties::water1_H(double Ba,double Sa)
{
    double p,T,h; 
    double pi,tao,gamat=0;
    int I[35]={0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,4,5,8,8,21,23,29,30,31,32};
    int J[35]={0,-2,-1,0,1,2,3,4,5,-9,-7,-1,0,1,3,-3,0,1,3,17,-4,0,6,-5,-2,10,-8,-11,-6,-29,-31,-38,-39,-40,-41};
    double n[35]={0,0.14632971213167e0,-0.84548187169114e0,-0.37563603672040e1,0.33855169168385e1,\
                  -0.95791963387872e0,0.15772038513228e0,-0.16616417199501e-1,0.81214629983568e-3,0.28319080123804e-3,\
                  -0.60706301565874e-3,-0.18990068218419e-1,-0.32529748770505e-1,-0.21841717175414e-1,\
                  -0.52838357969930e-4,-0.47184321073267e-3,-0.30001780793026e-3,0.47661393906987e-4,\
                  -0.44141845330846e-5,-0.72694996297594e-15,-0.31679644845054e-4,-0.28270797985312e-5,\
                  -0.85205128120103e-9,-0.22425281908000e-5,-0.65171222895601e-6,-0.14341729937924e-12,\
                  -0.40516996860117e-6,-0.12734301741641e-8,-0.17424871230634e-9,-0.68762131295531e-18,\
                  0.14478307828521e-19,0.26335781662795e-22,-0.11947622640071e-22,0.18228094581404e-23,\
                  -0.93537087292458e-25};
    int i;
    p=Ba*Pc1;
    T=Sa*Tc1;
    pi=p/16.53;
    tao=1386/T;
    for(i=1;i<35;i++)
    {
        gamat=gamat+n[i]*pow(7.1-pi,I[i])*J[i]*pow(tao-1.222,(J[i]-1));
    }
    gamat=gamat*tao;
    gamat=gamat*(461.526*T);
    h=gamat/(Pc1*1e6*Vc1);
    return(h);
}


double H2O_Properties::water1_Sa(double Ba,double Enthalpy)
{
    double t1=0,t2,Sak,t; 
    double h_1=0;
    int i=0;
    if(Ba<Ba1)	Sak=p_Kfun(Ba);
    else Sak=Sa1;
    t2=Sak;
    while(fabs(h_1-Enthalpy)>5e-12 && i<200)
    {
        t=(t1+t2)/2;
        h_1=water1_H(Ba,t);
        if(h_1>Enthalpy) 
            t2=t;
        else
            t1=t;
        i++;
    }
    //cout<<"It is in the second block!\n";
    return(t);
}


 //p t to SV
double H2O_Properties::water2_SV(double Ba,double Sa,double Bal,double Balpie)
{
    double x; //SV
    double B_0=1.683599274e1;
    double B0[5]={2.856067796e1,-5.438923329e1,4.330662834e-1,-6.547711697e-1,8.565182058e-2};
    double B1[2]={6.670375918e-2,1.388983801e0};
    double B2[3]={8.390104328e-2,2.614670893e-2,-3.373439453e-2};
    double B3[2]={4.520918904e-1,1.069036614e-1};
    double B4[2]={-5.975336707e-1,-8.847535804e-2};
    double B5[3]={5.958051609e-1,-5.159303373e-1,2.075021122e-1};
    double B6[2]={1.190610271e-1,-9.867174132e-2};
    double B7[2]={1.683998803e-1,-5.809438001e-2};
    double B8[2]={6.552390126e-3,5.710218649e-4};
    double B9[7]={1.936587558e2,-1.388522425e3,4.126607219e3,-6.508211677e3,5.745984054e3,-2.693088365e3,5.235718623e2};
    double b=7.633333333e-1,b61=4.006073948e-1,b71=8.636081627e-2,b81=-8.532322921e-1,b82=3.460208861e-1;
    //double n[8]={2,3,2,2,3,2,2,2};
    //double z[8][3]={{13,18,18,25,32,12,24},{3,2,10,14,28,11,18,14},{0,1,0,0,24,0,0,0}};
    //double l[8]={0,0,0,0,0,1,1,2};
    //double x[8][2]={{0,0,0,0,0,14,19,54},{0,0,0,0,0,0,0,27}};
    //double miu[8]={1,2,3,4,5,6,7,8};
    double x1=0,x2=0,x3=0,x4=0;
    //double h1=0,h2=0,h3=0,h4=0;
    double x21=0; //为x2项的中间参数
    double i1=4.260321148e0;
    double X;
    int v;
    X=exp(b*(1-Sa));
    x1=i1*Sa/Ba;
    x2=pow(Ba,0)*(B1[0]*pow(X,13)+B1[1]*pow(X,3))+2*pow(Ba,1)*(B2[0]*pow(X,18)+B2[1]*pow(X,2)+B2[2]*X)\
        +3*pow(Ba,2)*(B3[0]*pow(X,18)+B3[1]*pow(X,10))+4*pow(Ba,3)*(B4[0]*pow(X,25)+B4[1]*pow(X,14))\
        +5*pow(Ba,4)*(B5[0]*pow(X,32)+B5[1]*pow(X,28)+B5[2]*pow(X,24));
    x3=(4*pow(Ba,-5)*(B6[0]*pow(X,12)+B6[1]*pow(X,11)))/pow((pow(Ba,-4)+b61*pow(X,14)),2)\
        +(5*pow(Ba,-6)*(B7[0]*pow(X,24)+B7[1]*pow(X,18)))/pow((pow(Ba,-5)+b71*pow(X,19)),2)\
        +(6*pow(Ba,-7)*(B8[0]*pow(X,24)+B8[1]*pow(X,14)))/pow((pow(Ba,-6)+b81*pow(X,54)+b82*pow(X,27)),2);
    for(v=0;v<7;v++)  
        x4=x4+B9[v]*pow(X,v);
    x4=11*pow((Ba/Bal),10)*x4;
    x=x1-x2-x3+x4;
    return(x);
}

double H2O_Properties::water2_H(double Ba,double Sa,double Bal,double Balpie)
{
    double h; 
    double B_0=1.683599274e1;
    double B0[5]={2.856067796e1,-5.438923329e1,4.330662834e-1,-6.547711697e-1,8.565182058e-2};
    double B1[2]={6.670375918e-2,1.388983801e0};
    double B2[3]={8.390104328e-2,2.614670893e-2,-3.373439453e-2};
    double B3[2]={4.520918904e-1,1.069036614e-1};
    double B4[2]={-5.975336707e-1,-8.847535804e-2};
    double B5[3]={5.958051609e-1,-5.159303373e-1,2.075021122e-1};
    double B6[2]={1.190610271e-1,-9.867174132e-2};
    double B7[2]={1.683998803e-1,-5.809438001e-2};
    double B8[2]={6.552390126e-3,5.710218649e-4};
    double B9[7]={1.936587558e2,-1.388522425e3,4.126607219e3,-6.508211677e3,5.745984054e3,-2.693088365e3,5.235718623e2};
    double b=7.633333333e-1,b61=4.006073948e-1,b71=8.636081627e-2,b81=-8.532322921e-1,b82=3.460208861e-1;
    //double n[8]={2,3,2,2,3,2,2,2};
    //double z[8][3]={{13,18,18,25,32,12,24},{3,2,10,14,28,11,18,14},{0,1,0,0,24,0,0,0}};
    //double l[8]={0,0,0,0,0,1,1,2};
    //double x[8][2]={{0,0,0,0,0,14,19,54},{0,0,0,0,0,0,0,27}};
    //double miu[8]={1,2,3,4,5,6,7,8};
    double x1=0,x2=0,x3=0,x4=0;
    double h1=0,h2=0,h3=0,h4=0;
    double x21=0; //为x2项的中间参数
    double i1=4.260321148e0;
    double X;
    int v;
    X=exp(b*(1-Sa));

    h1=0+B_0*Sa;

    for(v=0;v<5;v++)
        h1=h1-B0[v]*((v+1)-2)*pow(Sa,(v+1)-1);
	
    h2=pow(Ba,1)*(B1[0]*(1+13*b*Sa)*pow(X,13)+B1[1]*(1+3*b*Sa)*pow(X,3))\
        +pow(Ba,2)*(B2[0]*(1+18*b*Sa)*pow(X,18)+B2[1]*(1+2*b*Sa)*pow(X,2)+B2[2]*(1+b*Sa)*X)\
        +pow(Ba,3)*(B3[0]*(1+18*b*Sa)*pow(X,18)+B3[1]*(1+10*b*Sa)*pow(X,10))\
        +pow(Ba,4)*(B4[0]*(1+25*b*Sa)*pow(X,25)+B4[1]*(1+14*b*Sa)*pow(X,14))\
        +pow(Ba,5)*(B5[0]*(1+32*b*Sa)*pow(X,32)+B5[1]*(1+28*b*Sa)*pow(X,28)+B5[2]*(1+24*b*Sa)*pow(X,24));

    h3=(B6[0]*pow(X,12)*((1+12*b*Sa)-(b*Sa*14*b61*pow(X,14))/(pow(Ba,-4)+b61*pow(X,14)))\
        +B6[1]*pow(X,11)*((1+11*b*Sa)-(b*Sa*14*b61*pow(X,14))/(pow(Ba,-4)+b61*pow(X,14))))/(pow(Ba,-4)+b61*pow(X,14));
    h3=h3+(B7[0]*pow(X,24)*((1+24*b*Sa)-(b*Sa*19*b71*pow(X,19))/(pow(Ba,-5)+b71*pow(X,19)))\
           +B7[1]*pow(X,18)*((1+18*b*Sa)-(b*Sa*19*b71*pow(X,19))/(pow(Ba,-5)+b71*pow(X,19))))/(pow(Ba,-5)+b71*pow(X,19));
    h3=h3+(B8[0]*pow(X,24)*((1+24*b*Sa)-(b*Sa*(54*b81*pow(X,54)+27*b82*pow(X,27)))/(pow(Ba,-6)+b81*pow(X,54)+b82*pow(X,27)))\
           +B8[1]*pow(X,14)*((1+14*b*Sa)-(b*Sa*(54*b81*pow(X,54)+27*b82*pow(X,27)))/(pow(Ba,-6)+b81*pow(X,54)+b82*pow(X,27))))/(pow(Ba,-6)+b81*pow(X,54)+b82*pow(X,27));

    for(v=0;v<7;v++)
        h4=h4+B9[v]*pow(X,v)*(1+Sa*(10*Balpie/Bal+v*b));
	
    h4=h4*Ba*pow(Ba/Bal,10);
	
    h=h1-h2-h3+h4;
    return(h);
}


double H2O_Properties::water2_Sa(double Ba,double Enthalpy,double Sal,double Bal,double Balpie)
{
    double t1=0,t2=Sa3,t; 
    double h_1=0;
    int i=0;
    while(fabs(h_1-Enthalpy)>5e-12 && i<200)
    {
        t=(t1+t2)/2;
        h_1=water2_H(Ba,t,Bal,Balpie);
        if(h_1>Enthalpy) 
            t2=t;
        else
            t1=t;
        i++;
    }
    //cout<<"It is in the second block!\n";
    return(t);
}

double H2O_Properties::water3_1(double x,double Sa)
{
    double C0[13]={-6.83990000e0,-1.72260420e-2,-7.77175039e0,4.20460752e0,-2.76807038e0,\
                   2.10419707e0,-1.14649588e0,2.23138085e-1,1.16250363e-1,-8.20900544e-2,1.94129239e-2,-1.69470576e-3,-4.311577033e0};
    double C1[7]={7.08636085e-1,1.23679455e1,-1.20389004e1,5.40437422e0,-9.93865043e-1,6.27523182e-2,-7.74743016e0};
    double C2[8]={-4.29885092e0,4.31430538e1,-1.41619313e1,4.04172459e0,1.55546326e0,-1.66568935e0,3.24881158e-1,2.93655325e1};
    double C3[10]={7.94841842e-6,8.08859747e1,-8.36153380e1,3.58636517e1,7.51895954e0,-1.26160640e1,\
                   1.09717462e0,2.12145492e0,-5.46529566e-1,8.32875413e0};
    double C4[2]={2.75971776e-6,-5.09073985e-4};
    double C50=2.10636332e2;
    double C6[5]={5.528935335e-2,-2.336365955e-1,3.697071420e-1,-2.596415470e-1,6.828087013e-2};
    double C7[9]={-2.571600553e2,-1.518783715e2,2.220723208e1,-1.802039570e2,2.357096220e3,-1.462335698e4,4.542916630e4,-7.053556432e4,4.381571428e4};
    double x_mid[6]={0,0,0,0,0,0};
    int v;
    double Ba=0;

    for(v=2;v<12;v++)
        x_mid[0]=x_mid[0]+(1-v)*C0[v]*pow(x,-v);
    x_mid[0]=x_mid[0]+C0[1]+C0[12]*pow(x,-1);
    x_mid[0]=-x_mid[0];
	
    for(v=2;v<7;v++)
        x_mid[1]=x_mid[1]+(1-v)*C1[v-1]*pow(x,-v);
    x_mid[1]=x_mid[1]+C1[0]+C1[6]*pow(x,-1);
    x_mid[1]=x_mid[1]*(1-Sa);

    for(v=2;v<8;v++)
        x_mid[2]=x_mid[2]+(1-v)*C2[v-1]*pow(x,-v);
    x_mid[2]=x_mid[2]+C2[0]+C2[7]*pow(x,-1);
    x_mid[2]=x_mid[2]*(Sa-1)*(1-Sa);

    for(v=2;v<10;v++)
        x_mid[3]=x_mid[3]+(1-v)*C3[v-1]*pow(x,-v);
    x_mid[3]=x_mid[3]+C3[0]+C3[9]*pow(x,-1);
    x_mid[3]=x_mid[3]*pow((1-Sa),3);

    x_mid[4]=5*C4[1]*pow(x,-6)*pow(Sa,-23)*(Sa-1);

    for(v=0;v<5;v++)
        x_mid[5]=x_mid[5]+C6[v]*pow(Sa,(-2-v));
    x_mid[5]=-x_mid[5]*6*pow(x,5);

    for(v=0;v<6;v++)
        Ba=Ba+x_mid[v];
    return(Ba);
}

double H2O_Properties::water3_SV(double Ba, double Sa)
{
    double v1=0.001/Vc1,v2,v;
    double Ba_1=0;
    int i=0;
    double Sal,Bal=Ba,Balpie;
    Sal=p_Lfun(Ba);
    Balpie=Lpiefun(Sal);
    v2=water2_SV(Ba,Sal,Bal,Balpie);

    while(fabs(Ba_1-Ba)>5e-12 && i<200)
    {
        v=(v1+v2)/2;
        Ba_1=water3_1(v,Sa);
        if(Ba_1>Ba) 
            v1=v;
        else
            v2=v;
        i++;
    }
    return(v);
}

double H2O_Properties::water3_H(double x,double Sa)
{
    double C0[13]={-6.83990000e0,-1.72260420e-2,-7.77175039e0,4.20460752e0,-2.76807038e0,\
                   2.10419707e0,-1.14649588e0,2.23138085e-1,1.16250363e-1,-8.20900544e-2,1.94129239e-2,-1.69470576e-3,-4.311577033e0};
    double C1[7]={7.08636085e-1,1.23679455e1,-1.20389004e1,5.40437422e0,-9.93865043e-1,6.27523182e-2,-7.74743016e0};
    double C2[8]={-4.29885092e0,4.31430538e1,-1.41619313e1,4.04172459e0,1.55546326e0,-1.66568935e0,3.24881158e-1,2.93655325e1};
    double C3[10]={7.94841842e-6,8.08859747e1,-8.36153380e1,3.58636517e1,7.51895954e0,-1.26160640e1,\
                   1.09717462e0,2.12145492e0,-5.46529566e-1,8.32875413e0};
    double C4[2]={2.75971776e-6,-5.09073985e-4};
    double C50=2.10636332e2;
    double C6[5]={5.528935335e-2,-2.336365955e-1,3.697071420e-1,-2.596415470e-1,6.828087013e-2};
    double C7[9]={-2.571600553e2,-1.518783715e2,2.220723208e1,-1.802039570e2,2.357096220e3,-1.462335698e4,4.542916630e4,-7.053556432e4,4.381571428e4};
    double h=0,hm[6]={0,0,0,0,0,0};
    int v;
	
    for(v=2;v<12;v++)
        hm[0]=hm[0]+v*C0[v]*pow(x,1-v);
    for(v=2;v<7;v++)
        hm[0]=hm[0]-C1[v-1]*pow(x,1-v);
    hm[0]=hm[0]+C0[0]-C0[12]-C50-C1[0]*x+(C0[12]-C1[6])*log(x);

    for(v=2;v<7;v++)
        hm[1]=hm[1]+(v-1)*C1[v-1]*pow(x,1-v);
    for(v=2;v<8;v++)
        hm[1]=hm[1]-2*C2[v-1]*pow(x,1-v);
    hm[1]=hm[1]-C1[6]-C50-(C1[0]+2*C2[0])*x-2*C2[7]*log(x);
    hm[1]=hm[1]*(Sa-1);

    for(v=2;v<8;v++)
        hm[2]=hm[2]+(v-2)*C2[v-1]*pow(x,1-v);
    for(v=2;v<10;v++)
        hm[2]=hm[2]-3*C3[v-1]*pow(x,1-v);
    hm[2]=hm[2]-C2[7]-(2*C2[0]+3*C3[0])*x-(C2[7]+3*C3[9])*log(x);
    hm[2]=hm[2]*pow(Sa-1,2);

    for(v=2;v<10;v++)
        hm[3]=hm[3]+(v-3)*C3[v-1]*pow(x,1-v);
    hm[3]=hm[3]-C3[9]-3*C3[0]*x-2*C3[9]*log(x);
    hm[3]=hm[3]*pow(Sa-1,3);

    for(v=0;v<5;v++)
        hm[4]=hm[4]+pow(x,6)*(v-3)*C6[v]*pow(Sa,-(2+v));
    hm[4]=hm[4]+(23*C4[0]+28*C4[1]*pow(x,-5))*pow(Sa,-22);
    hm[4]=hm[4]-(24*C4[0]+29*C4[1]*pow(x,-5))*pow(Sa,-23);
	
    for(v=0;v<9;v++)
        hm[5]=hm[5]-C7[v]*(1+v*Sa)*pow(Sa-1,v);

    for(v=0;v<6;v++)
        h=h+hm[v];
    return(h);
}

double H2O_Properties::water3_Sa(double Ba,double Enthalpy,double Sak,double Sal)
{
    double t1=Sak,t2=Sal,t; 
    double h_1=0,v3;
    int i=0;
    while(fabs(h_1-Enthalpy)>5e-12 && i<200)
    {
        t=(t1+t2)/2;
        v3=water3_SV(Ba,t);
        h_1=water3_H(v3,t);
        if(h_1>Enthalpy) 
            t2=t;
        else
            t1=t;
        i++;
    }
    //cout<<"It is in the third block!\n";
    return(t);
}

double H2O_Properties::water4_1(double x,double Sa)
{
    double D3[5]={-1.717616747e0,3.526389875e0,-2.690899373e0,9.070982605e-1,-1.138791156e-1};
    double D4[5]={1.301023613e0,-2.642777743e0,1.996765362e0,-6.661557013e-1,8.270860589e-2};
    double D5[3]={3.426663535e-4,-1.236521258e-3,1.155018309e-3};
    int v;
    double Ba=0;
    double y;
    y=(1-Sa)/(1-Sa1);
    for(v=0;v<5;v++)
        Ba=Ba+v*D3[v]*pow(y,3)*pow(x,-v-1)+v*D4[v]*pow(y,4)*pow(x,-v-1);
    for(v=0;v<3;v++)
        Ba=Ba-pow(y,32)*v*D5[v]*pow(x,v-1);
    Ba=Ba+water3_1(x,Sa);
    return(Ba);
}

double H2O_Properties::water4_SV(double Ba, double Sa)
{
    double v1=0.001/Vc1,v2=1,v; 
    double Ba_1=0;
    int i=0;
    while(fabs(Ba_1-Ba)>5e-12 && i<200)
    {
        v=(v1+v2)/2;
        Ba_1=water4_1(v,Sa);
        if(Ba_1>Ba) 
            v1=v;
        else
            v2=v;
        i++;
    }
    return(v);
}

double H2O_Properties::water4_H(double x,double Sa)
{
    double D3[5]={-1.717616747e0,3.526389875e0,-2.690899373e0,9.070982605e-1,-1.138791156e-1};
    double D4[5]={1.301023613e0,-2.642777743e0,1.996765362e0,-6.661557013e-1,8.270860589e-2};
    double D5[3]={3.426663535e-4,-1.236521258e-3,1.155018309e-3};
    double h,h1=0,h2=0,y;
    int v;
    y=(1-Sa)/(1-Sa1);
    for(v=0;v<5;v++)
        h1=h1+D3[v]*((1-3+v)*y+3/(1-Sa1))*pow(y,3-1)*pow(x,-v)+D4[v]*((1-4+v)*y+4/(1-Sa1))*pow(y,4-1)*pow(x,-v);
    for(v=0;v<3;v++)
        h2=h2+pow(y,31)*D5[v]*((31+v)*y-32/(1-Sa1))*pow(x,v);
    h=water3_H(x,Sa)+h1-h2;
    return(h);
}

double H2O_Properties::water4_Sa(double Ba,double Enthalpy,double Sak)
{
    double t1=Sa1,t2=Sak,t; 
    double h_1=0,v4;
    int i=0;
    while(fabs(h_1-Enthalpy)>5e-12 && i<200)
    {
        t=(t1+t2)/2;
        v4=water4_SV(Ba,t);
        h_1=water4_H(v4,t);
        if(h_1>Enthalpy) 
            t2=t;
        else
            t1=t;
        i++;
    }
    //cout<<"It is in the fourth block!\n";
    return(t);
}

//to calculate dynamic viscosity(Kg/(m.s))
double H2O_Properties::Dynamic_Viscosity(double Tem, double Rho)
{
    double Tp=Tem/Tc1,Rhop=Rho/Rhoc1;
    double eta0,eta1=0,eta2=1,eta;
    double H[4]={1.000000,0.978197,0.579829,-0.202354};
    double Hij[6][7]={{0.5132047,0.2151778,-0.2818107,0.1778064,-0.04176610,0,0},\
                      {0.3205656,0.7317883,-1.070786,0.4605040,0,-0.01578386,0},{0,1.241044,-1.263184,\
                                                                                 0.2340379,0,0,0},{0,1.476783,0,-0.4924179,0.1600435,0,-0.003629481},\
                      {-0.7782567,0,0,0,0,0,0},{0.1885447,0,0,0,0,0,0}};
    double mid=0;
    int i,j;
    for(i=0;i<4;i++)
        mid=mid+(H[i]/pow(Tp,i));
    eta0=sqrt(Tp)/mid;
    for(i=0;i<6;i++)
    {
        for(j=0;j<7;j++)
            eta1=eta1+Rhop*Hij[i][j]*pow((1/Tp-1),i)*pow((Rhop-1),j);
    }
    eta1=exp(eta1);
    eta=eta0*eta1*eta2;
    return(eta*Etaxing);
}


	
//to calculate thermal conductivity, W/(m.K)
double H2O_Properties::Conductivity(double Tem,double Rho)
{
    double la0=0,lap=0,lad=0,lambda;
    double a[4]={1.02811e-2,2.99621e-2,1.56146e-2,-4.22464e-3};
    double b0=-3.97070e-1,b1=4.00302e-1,b2=1.06000;
    double B1=-1.71587e-1,B2=2.39219;
    double C1=6.42857e-1,C2=-4.11717,C3=-6.17937,C4=3.08976e-3,C5=8.22994e-2,C6=1.00932e1;
    double d1=7.01309e-2,d2=1.18520e-2,d3=1.69937e-3,d4=-1.02000;
    int i;
    double dtxing,Q,R,Ss;
    Rho=Rho/Rhoc1;
    dtxing=fabs(Tem/Tc1-1.0)+C4;
    Q=2.0+C5*pow(dtxing,-0.6);
    R=Q+1.0;
    if(Tem/Tc1>=1) Ss=pow(dtxing,-1.0);
    else Ss=C6*pow(dtxing,-0.6);
    for(i=0;i<4;i++)
        la0=la0+sqrt(Tem/Tc1)*a[i]*pow(Tem/Tc1,i);
    lap=b0+b1*Rho+b2*exp(B1*pow((Rho+B2),2));
    lad=(d1*pow((Tc1/Tem),10)+d2)*pow(Rho,1.8)*exp(C1*(1-pow(Rho,28)));
    lad=lad+d3*Ss*pow(Rho,Q)*exp(Q/R*(1-pow(Rho,R)));
    lad=lad+d4*exp(C2*pow(Tem/Tc1,1.5)+C3/pow(Rho,5));
    lambda=la0+lap+lad;
    return(lambda);
}


//to calculate heat capacity, J/(kg.K)
double H2O_Properties::Heat_Cp(double Temp, double Pre,double SteamX)
{
    double h1,h2,v1,v2;
    double cp,t1,t2;
    t1=Temp-0.1;
    t2=Temp+0.1;
    Water_PT(Pre, t1, SteamX);
    Water_PT(Pre, t2, SteamX);
    cp=(h2-h1)/0.2;  //kJ/(kg.K)
    return(cp*1000);   //convert to J/(kg.K)
    cout<<fixed<<setprecision(4)<<"cp="<<cp<<"\n";
}




















