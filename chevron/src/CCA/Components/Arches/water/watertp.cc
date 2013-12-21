#include "watertp.h"
//this program is used to calculate the characteristics of the water


using namespace std;
main()
{
    class H2O_Properties Water;
    double in_T,in_P;
    double nd,lam,cp;
    double in_SV, in_H, in_SteamX; 
    int I;
    cout<<"                 ********************************************************\n";
    cout<<"                 *      Author: Yuxin Wu, ICSE, University of Utah      *\n";
    cout<<"                 *                   wu.yuxin@utah.edu                  *\n";
    cout<<"                 *            for water properties calculation          *\n";
    cout<<"                 *           based on Temperature(T), Pressure(P)       *\n";
    cout<<"                 *           Specific Volum(SV) and Enthalpy(E)         *\n";
    cout<<"                 ********************************************************\n";
    cout<<"\n\n";
    cout<<"                 ========================================================\n";
    cout<<"                 =       PLEASE INPUT 0 IF you WANT TO EXIT             =\n";
    cout<<"                 =       Press 1 to get E and SV from T and P           =\n";
    cout<<"                 =       Press 2 to get T and SV from E and P           =\n";
    cout<<"                 =       All Units are in SI Unit                       =\n";
    cout<<"                 ========================================================\n";
    cout<<"please input your choice:";
    cin>>I;

    while(I==1 || I==2)
    {
        if(I==0) exit(0);
        if(I==1)
        {
            cout<<"please input the temperature(K):";
            cin>>in_T;
            cout<<"please input the pressure(Pa) :";
            cin>>in_P;
            cout<<"please input the steam fraction:";
            cin>>in_SteamX;
            //            set_PT(Pressure, Temperature);
            //            set_SteamX(Steam_Fraction);
            Water.Water_PT(in_P, in_T, in_SteamX);
            if(Water.Steam_Fraction==0)
                cout<<"It's pure water\n";
            else if(Water.Steam_Fraction==1)
                cout<<"It's steam\n";
            else if(Water.Steam_Fraction>0 && Water.Steam_Fraction<1)
                cout<<"It's steam-water mixture! Steam_Fraction="<<Water.Steam_Fraction<<"\n";
            cout<<"temperature:"<<Water.Temperature-273.15<<"cc"<<"\n";
            cout<<"pressure:"<<Water.Pressure<<"Pa"<<"\n";
            cout<<"Specific Volume:"<<Water.SV<<"m3/kg"<<"\n";
            cout<<"enthalpy:"<<Water.Enthalpy<<"J/kg"<<"\n";
        }
        if(I==2)
        {
            cout<<"please input the pressure(Pa):";
            cin>>in_P;
            cout<<"please input the ehthalpy(J/kg):";
            cin>>in_H;
            in_SteamX;
            Water.p_waterBah(in_P, in_H);
            if(Water.Steam_Fraction==0)
                cout<<"Pure water!\n";
            else if(Water.Steam_Fraction==1)
                cout<<"Steam! \n";
            else if(Water.Steam_Fraction>0 && Water.Steam_Fraction<1)
                cout<<"steam-water mixture! Steam_Fraction="<< Water.Steam_Fraction<<"\n";
            cout<<"Pressure:"<<Water.Pressure<<"Pa"<<"\n";
            cout<<"Enthalpy:"<<Water.Enthalpy<<"J/kg"<<"\n";
            cout<<"Temperature:"<<Water.Temperature-273.15<<"cc"<<"\n";
            cout<<"Specic volume:"<<Water.SV<<"m3/kg"<<"\n";	
        }
        nd=Water.Dynamic_Viscosity(Water.Temperature,1/Water.SV);
	cout<<"dynamic viscosity:"<<nd<<"kg/(m.s)\n";
	lam=Water.Conductivity(Water.Temperature,1/Water.SV);
	cout<<"Conductivity:"<<lam<<"W/(m.k)\n";
	//Specific heat J/(kg.K)
	cp=Water.Heat_Cp(Water.Temperature,Water.Pressure,Water.Steam_Fraction);
	cout<<"Specific heat Cp="<<cp<<"J/(kg.K)\n";
        cout<<"\n\nplease input your choice:";
        cin>>I;
    }
}

