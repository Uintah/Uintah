# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import numpy as np  #TODO this will run faster if you import only specific utilities.

from mixing_table import ClassicTable as ct

class Coal(object):
    def __init__(self, coal_filename):
        """Read the coal input file & save the necessary coal properties."""
        # read in inputs from coal file
        tree = ET.parse(coal_filename)
        root = tree.getroot()
        input_variable = root.find('./Coal_Properties')
        self.as_received = np.array(eval(input_variable.find('as_received').text))
        #[C, H, O, N, S, Char, Ash, Moisture] (should sum to 1.0)
        # 0  1  2  3  4  5     6    7  
        self.particle_sizes = np.array(eval(input_variable.find('particle_sizes').text))
        self.particle_weights = np.array(eval(input_variable.find('particle_weights').text))
        self.HHV = eval(input_variable.find('HHV').text)  # [J/g]
        self.use_HHV_est = input_variable.find('use_HHV_est').text  # should the code guess a new HHV
        self.compute_scaling_const = input_variable.find('compute_scaling_const').text  # should the code compute the scaling constants
        self.density = eval(input_variable.find('particle_density').text)  # [kg/m^3] 
        self.particle_temperature = eval(input_variable.find('particle_temperature').text)  # [K] 
        self.raw_coal_dh_formation = eval(input_variable.find('raw_coal_enthalpy').text)/1000   # [J/g]  
        self.ash_dh_formation = eval(input_variable.find('ash_enthalpy').text)/1000   # [J/g]  
        self.char_dh_formation = eval(input_variable.find('char_enthalpy').text)/1000   # [J/g]   
        # compute useful quantaties
        total_dry = np.sum(self.as_received[:7])  # total dry mass
        self.total_dry=total_dry
        total_daf = np.sum(self.as_received[:5])  # total dry, ash-free mass (daf)
        self.total_daf=total_daf
        mass_dry = self.as_received[:7] / total_dry  # elemental mass frac. (dry)
        self.mass_dry=mass_dry
        ash_mass_frac = mass_dry[6]  # fraction of dry coal that is ash
        self.ash_mass_frac=ash_mass_frac
        char_mass_frac = mass_dry[5]  # fraction of dry coal that is char
        self.char_mass_frac=char_mass_frac
        coal_mass_frac = 1.0 - ash_mass_frac - char_mass_frac # fraction of dry coal that is raw coal
        self.coal_mass_frac=coal_mass_frac
        self.slurry_mass_frac=0
        mass_daf = self.as_received[:5] / total_daf  # elemental mass frac. (daf)
        self.mass_daf=mass_daf
        # assign as received HHV
        use_HHV_est=self.use_HHV_est
        if use_HHV_est=='yes':
            HHV_est1,HHV_est2=self.HHV_est(self.mass_daf)  # these are estimates of HHV from M.Sciazko / Fuel 114 (2013) 2-9  
            HHV_ar=-HHV_est1
        else:
            HHV_ar = -self.HHV  # [J/g]  # HHV as received  
        self.HHV_ar=HHV_ar
        self.formation_enthalpy()
        self.sensible_enthalpy()
        self.inlet_enthalpy()

    def formation_enthalpy(self):
        """Calculate the enthalpy of formation for the coal."""
        dh_combustion=self.HHV_ar/(1-self.as_received[5]-self.as_received[6]-self.as_received[7])        
        # Heat of formation of the products
        element_MW = np.array([12.0107, 1.00794, 15.9994, 14.0067, 32.065])  # [g/mol] C, H, O, N, S
        basis = 1000.0  # 1000 grams of coal
        mass_daf_basis = basis * self.mass_daf # grams of each element in the coals
        moles_daf_basis = mass_daf_basis / element_MW  # moles of each element in the coal
        product_name = ['CO2', 'H2O(liquid)', 'O2', 'N2', 'SO2']  # product species names
        product_dHf = np.array([-393509., -285840., 0., 0., -296840.])  # enthalpy of formation at 298 K [J/mol] CO2, H2O, O2, N2, SO2
        product_stoich = np.array([1., 0.5, 0.5, 0.5, 1.])  # stoich. coefficients for product reactions, C+O2=>CO2 2H+0.5O2=>H2O O=>0.5O2 N=>0.5N2 S+O2=>SO2
        heat_per_basis = product_dHf * moles_daf_basis * product_stoich  # [J]
        sum_heat_per_basis = np.sum(heat_per_basis)  # [J] for 1000g of coal
        dh_products = sum_heat_per_basis / basis  # [J/g]
        # coal heat of formation  (dh_products = dh_formation + dh_reaction) -> dh_formation=dh_products-dh_reaction
        dh_formation_daf = dh_products - dh_combustion  # J/g heat of formation of dry, ash-free coal
        self.dh_formation_daf=dh_formation_daf #  J/g
        if (np.abs(self.dh_formation_daf-self.raw_coal_dh_formation) > 0.1 ):
            print "Warning enthalpy of formation for coal in coal input file"
            print "is different than what was computed in formation_enthalpy()"
            print "input file: ",self.raw_coal_dh_formation
            print "computed: ",self.dh_formation_daf
            print "script is using value from coal input file!!!!"
            

    def sensible_enthalpy(self):
        """Calculate the sensible enthalpy."""
        # sensible enthalpy from Merrick 1983
        #element_MW = np.array([12.0107, 1.00794, 15.9994, 14.0067, 32.065])  # [g/mol] C, H, O, N, S
        element_MW = np.array([ 12., 1., 16., 14., 32.])  # [g/mol] C, H, O, N, S
        T_ref=298.15
        T_particle=self.particle_temperature
        Tvec=np.linspace(T_ref,T_particle,1000)
        Rgas = 8314.3  # J/kg/K
        MW_avg = 0.0  # Mean atomic weight of coal
        for i in range(5):
            MW_avg += self.mass_daf[i]/element_MW[i];
        MW_avg = 1/MW_avg      
        self.hsens_daf=self.calc_hcint(T_particle,MW_avg)/1000 - self.calc_hcint(298.15,MW_avg)/1000
        self.hsens_char=self.calc_hhint(T_particle,MW_avg)/1000 - self.calc_hhint(298.15,MW_avg)/1000
        self.hsens_ash=self.calc_haint(T_particle)/1000 - self.calc_haint(298.15)/1000

    def g1(self,z1,z2):
        """z function for heat capacity calculation."""
        sol = 380.0/(np.exp(z1)-1.0) + 3600.0/(np.exp(z2)-1.0);
        return sol
        
    def calc_hcint(self,Tp,MW_avg):
        """calc coal enthalpy. J/kg"""
        Rgas = 8314.3 # J/K/kmol
        z1 = 380.0/Tp
        z2 = 1800.0/Tp
        sol = (Rgas/MW_avg)*self.g1(z1,z2)
        return sol
        
    def calc_hhint(self,Tp,MW_avg):
        """calc char enthalpy. J/kg"""
        Rgas = 8314.3 # J/K/kmol
        z1 = 380.0/Tp
        z2 = 1800.0/Tp
        sol = (Rgas/12.0)*self.g1(z1,z2)
        return sol
        
    def calc_haint(self,Tp):
        """calc ash enthalpy. J/kg"""
        sol = (593.0 + 0.293*Tp)*Tp
        return sol
                
    def HHV_est(self,comp):
        """Provde an estimate for the heat of combustion of coal in J/g on a dry ash-free base, fraction are in wt.%."""
        # these are estimates of HHV from M.Sciazko / Fuel 114 (2013) 2-9
        C=comp[0]*100
        H=comp[1]*100
        O=comp[2]*100
        N=comp[3]*100
        S=comp[4]*100
        HHV9=341*C+1323*H+68.5-119.4*(O+N)
        HHV1=338.3*C+1443*(H-O/8)+94.2*S
        return HHV9,HHV1

    def inlet_enthalpy(self):        
        self.h_particles=(self.raw_coal_dh_formation+self.hsens_daf)*self.coal_mass_frac+(self.ash_dh_formation+self.hsens_ash)*self.ash_mass_frac+(self.char_dh_formation+self.hsens_char)*self.char_mass_frac  # J/g particle enthalpy
        self.dh_daf=self.raw_coal_dh_formation+self.hsens_daf
        self.dh_ash=self.ash_dh_formation+self.hsens_ash
        self.dh_char=self.char_dh_formation+self.hsens_char  
        
    def create_table_input_file(self,table_base_name):
        """Generate an the input file that makes the lookup table."""
        print "Creating table input file"        
        text_file = open(table_base_name, "w")

        text_file.writelines('  2,                                         !NSAY..(SAY(I),I=1,NSAY) follows:\n')
        text_file.writelines('                   ARCHES %%%%%%%%%%%%% HOT FLOW\n')
        text_file.writelines('    >>>>>>>>>>     OXYCOAL Utah FURNACE UQ case 7   <<<<<<<<<<\n')
        text_file.writelines('TRUE,TRUE,FALS,TRUE                      !LCALF,LCALET,LCO,LCALH\n')
        text_file.writelines('TRUE,FALS,TRUE,FALS                      !LCOAL,LPRMIX,LCREE,LNASA\n')
        text_file.writelines('FALS                                     !LEQLS\n')
        text_file.writelines('  101325.                                !PRES\n')
        text_file.writelines('  0.0018546438, 1.0 , 0.0                !FLOWPR,FPR,ETAPR\n') # kg/s flow rate of primary
        text_file.writelines('  0.002079, 0.0 , 0.0                    !FLOWSC,FSC,ETASC\n') # kg/s secondary rate of primary
        text_file.writelines('  1.0                                    !PLOAD \n')
        text_file.writelines('  273.0, -0.5                            !TREF, HLMIN\n')  # heat loss reference temperature, minimum heat loss
        text_file.writelines('  24,   30,  30                          !NC(coal gas mix frac),NTX(heat loss),NTZ(mix frac)\n')       
        text_file.writelines('                                         !                       (Blank line)\n')
        text_file.writelines('ELEMENTS\n')
        text_file.writelines('THERMO                                       !The react. sect. is formatted\n')
        text_file.writelines('REACTANTS 1\n')
        text_file.writelines('    300.00                                   !TMP (unformatted)\n')  # reactant 1 temperature
        text_file.writelines('O 2.       0.       0.       0.     O2       0.001            G\n')  # reactant 1 element: species
        text_file.writelines('C 1.     O 2.       0.       0.     CO2      0.9990           G\n')  # reactant 1 element: species
        text_file.writelines('                                          !                       (Blank line)\n')
        text_file.writelines('REACTANTS 2\n')
        text_file.writelines('    488.70                                   !TMP (unformatted)\n')  # reactant 2 temperature
        text_file.writelines('O 2.       0.       0.       0.     O2       0.9990           G\n')  # reactant 2 element: species
        text_file.writelines('C 1.     O 2.       0.       0.     CO2      0.0010           G\n')  # reactant 2 element: species
        text_file.writelines('                                          !                       (Blank line)\n')
        text_file.writelines('COAL\n')
        text_file.writelines('    %1.1f                                    !TMP (unformatted)\n' % self.particle_temperature)# incoming temperature of coal particles "0"
        text_file.writelines(' %1.6E, %1.6E, %1.6E, !HC0(J),HH0(J),HA0(J)\n' % (self.raw_coal_dh_formation*1000,self.char_dh_formation*1000,self.ash_dh_formation*1000))  # Enthalpy at "0": rawcoal, char, ash (converted to J/kg)
        text_file.writelines(' %1.6E, %1.6f,                   !HW0(J),TNBP\n' % (0,300))  # Enthalpy at "0": slurry (converted to J/kg), temperature normal boiling point for slurry 
        text_file.writelines(' %1.6E, %1.6E, %1.6E,   !OMEGAC(J),OMEGAH(J),OMEGAA(J)\n' % (self.coal_mass_frac,self.char_mass_frac,self.ash_mass_frac))  # mass fraction: raw coal(OmegaC) + char(OmegaH) + ash(OmegaA) + slurry(OmegaW) = 1
        text_file.writelines(' %1.6E,                               !OMEGAW(J)\n' % self.slurry_mass_frac)
        text_file.writelines(' %1.6E, %1.6E, %1.6E,   !(WIC(J,K) K = 1,3)\n' % (self.mass_daf[0],self.mass_daf[1],self.mass_daf[2]))  # mass fraction for elements C + H + O + N + S = 1(order is the same as the thermo file)
        text_file.writelines(' %1.6E, %1.6E,                 !(WIC(J,K) K = 4,NLM)            \n' % (self.mass_daf[3],self.mass_daf[4]))         
        text_file.writelines(' H2O                                         !SLRCMP\n') # Slurry component 
        text_file.writelines('                                             !(Blank line)\n')
        text_file.close()
        print "Generating table"
        print "WARNING!!!!!! USER MUST SPECIFY THE FOLLOWING PARAMETERS"
        print "IN THE FILE: ",table_base_name
        print "REACTANTS 1"
        print "REACTANTS 2"
        print "PRES (system pressure)"
        print "FPR,ETAPR (mix frac primary, coal gas mix frac primary)"
        print "FSC,ETASC (mix frac secondary, coal gas mix frac secondary)"
        print "TREF,HLMIN (reference temperature for sensible enthalpy, and minimum heat loss" 


class BC(object):
    def __init__(self, bc_filename, face_name):
        """Read the user input files for each boundary condition."""
        tree = ET.parse(bc_filename)
        root = tree.getroot()
        self.bc_name = face_name
        input_variable = root.find('./CoalFeedRate')
        CoalFeedRate = input_variable.find('value').text
        self.lPartMassFlowInlet = input_variable.find('useMassFlowInlet')
        #if self.lPartMassFlowInlet==None:
           #self.cfr_var = "PartMassFlowInlet" 
        #else:
        self.cfr_var = input_variable.find('var').text

        self.cfr = float(CoalFeedRate)
        sndi = input_variable.find('small_number_density_inlet')
        wbc = input_variable.find('wall_boundary_condition')
        if sndi is None:
            self.small_number_density_inlet="false"
        else: 
            self.small_number_density_inlet="true"
        if wbc is None:
            self.wall_boundary_condition="false"
        else: 
            self.wall_boundary_condition="true"    
        string="./gas_phase/"
        gas_phase = root.findall(string)
        self.gas_names, self.gas_vars, self.gas_values,  self.swirl_no = [], [], [], []
        for phase in gas_phase:
            name = phase.tag
            var = phase.find('var').text
            value = phase.find('value').text
            self.gas_names.append(name)
            self.gas_vars.append(var)
            try:
                self.gas_values.append(float(value))
            except:
                self.gas_values.append(value)

            if var == 'Swirl' :
                self.swirl_no.append( phase.find('swirl_no').text )

    def mdot_modify(self,coal,correction_factor):
        """Modify mass flow rates if moisture is assumed to immediately enter the gas phase."""
        if (self.cfr > 0) and coal.as_received[7]>0:
            print "MASS FLOW RATES HAVE BEEN MODIFIED"
            print "Coal moisture was assumed to be in the"
            print "vapor phase and was moved to the gas phase."
            print "original coal feed rate [kg/s]=",self.cfr            
            moisture=self.cfr*coal.as_received[7]
            self.cfr=(self.cfr-moisture)*correction_factor
            print "new coal feed rate [kg/s]=",self.cfr  
            for i in range(np.size(self.gas_names)):
                if self.gas_names[i]=='mass_inlet':
                    print "original gas feed rate [kg/s]=",self.gas_values[i] 
                    self.gas_values[i]=self.gas_values[i]+moisture
                    print "new gas feed rate [kg/s]=",self.gas_values[i] 


class Face(object):
    def __init__(self, ups_filename):
        """Parse the ups file and return all relevant face information."""
        tree = ET.parse(ups_filename)
        root = tree.getroot()
        string = "./Grid/BoundaryConditions/"
        boundary_conditions = root.findall(string)
        count = 0
        self.input_files, self.names, self.new_input_file_names = [], [], []
        self.area, self.orientation, self.orientation_letter = [], [], []
        self.weight_constant=0
        for my_face in boundary_conditions:
            name = my_face.get('name')
            include = my_face.find('include')
            input_file = include.get('href')
            new_input_file = input_file.split('.xml')[0]
            self.input_files.append(input_file)
            self.names.append(name)
            geom = my_face.get('circle')
            if bool(geom):
                orientation = geom
                radius = my_face.get('radius')
                area = np.pi * float(radius) * float(radius)
                self.area.append(area)
                self.orientation.append(orientation)
            geom = my_face.get('annulus')
            if bool(geom):
                orientation = geom
                inner_radius = my_face.get('inner_radius')
                outer_radius = my_face.get('outer_radius')
                area = ( np.pi * float(outer_radius) * float(outer_radius) -
                         np.pi * float(inner_radius) * float(inner_radius) )
                self.area.append(area)
                self.orientation.append(orientation)
            geom = my_face.get('rectangle')
            if bool(geom):
                orientation = geom   # x+ x- y+ y- etc..
                lower = my_face.get('lower')
                lower=np.array(map(float, lower.split(' ')))  # this means there must be one space between input arguments
                upper = my_face.get('upper')
                upper=np.array(map(float, upper.split(' ')))
                dx=np.abs(lower[0]-upper[0])
                dy=np.abs(lower[1]-upper[1])
                dz=np.abs(lower[2]-upper[2])   
                if dx==0:
                    area = dy*dz
                if dy==0:
                    area = dx*dz
                if dz==0:
                    area = dx*dy                    
                self.area.append(area)
                self.orientation.append(orientation)
            geom = my_face.get('side')
            if bool(geom):
                orientation = geom
                area = 10000000000
                self.area.append(area)
                self.orientation.append(orientation)
            orientation_letter = orientation 
            new_input = new_input_file + '.' + str(count) + '.' + orientation_letter + '.xml'
            self.orientation_letter.append(orientation_letter)
            self.new_input_file_names.append(new_input)
            count += 1
    def gas_velocity(self, bc, myTable, counti):
        """Compute gas phase velocity from gas phase density and flow rate."""
        # First, get density of the gas stream
        count = 0
        mdot = 0
        for ind in bc.gas_names:
            if ind == 'mixture_fraction_2':
                mixture_fraction_2=bc.gas_values[count]
            elif ind == 'coal_gas_mix_frac':
                coal_gas_mix_frac=bc.gas_values[count]
            elif ind == 'heat_loss':
                heat_loss=bc.gas_values[count]
            elif ind == 'mass_inlet':
                mdot=bc.gas_values[count]
            count=count+1
        if 'mixture_fraction_2' in locals():
            f = mixture_fraction_2 / (1.0 - coal_gas_mix_frac)
            x0 = np.array([coal_gas_mix_frac, heat_loss, f])  # eta, hl , f
        else :
            print "Assuming a 2-D table"
            x0 = np.array([coal_gas_mix_frac, heat_loss])  # eta, hl , f
        value = myTable.interpolate(x0)
        cter=0
        for name in myTable.dep_names:
            if name=='density':
                density = value[cter]
            #if name=='temperature':
            #    print value[cter]
            cter=cter+1
        # Second, compute velocity of the gas stream
        velocity = mdot / (density * self.area[counti])  # kg/s * m3/kg * 1/m2 [=] m/s
        return velocity
    def gas_velocity_no_lookup(self, bc,  counti, density):
        count = 0
        mdot = 0
        for ind in bc.gas_names:
            if ind == 'mass_inlet':
                mdot=bc.gas_values[count]
            count=count+1
        velocity = mdot / (density * self.area[counti])  # kg/s * m3/kg * 1/m2 [=] m/s
        return velocity
        # Modify new UPS file
    def rewrite_ups(self, ups_filename, file_directory):
      """Change the input file names (depending on orientation) and resave."""
      tree = ET.parse(ups_filename)
      root = tree.getroot()
      count = 0
      for i in self.names:
          string = "./Grid/BoundaryConditions/Face/[@name='" + i + "']"
          my_face = root.find(string)
          include = my_face.find('include')
          include.set('href',file_directory+self.new_input_file_names[count])
          count += 1        
      """change scaling constants"""
      scaling_const = root.find("./CFD/ARCHES/DQMOM/Weights/scaling_const")
      scaling_const.text=str(self.weight_constant) 
      count = 0
      for i in self.scaling_constants:      
          scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='"+self.scaling_names[count]+"']/scaling_const")
          scaling_const.text=str(i) 
          count += 1   
      """change initial values"""
      counter=0
      self.init_weights=[1e3,1e3,1e3]
      for i in self.init_weights:    
          init_value = root.find("./CFD/ARCHES/DQMOM/Weights/initialization/env_constant/[@qn=\""+str(counter)+"\"]")
          init_value.set("value",str(i))
          counter=counter+1    
      count = 0  
      for i in self.scaling_names:   
          counter=0
          for j in self.init_weights: 
              init_value = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='"+self.scaling_names[count]+"']/initialization/env_constant/[@qn=\""+str(counter)+"\"]")
              init_value.set("value",str(self.init_ics[count,counter]))
              counter += 1    
          count += 1       
      tree.write(ups_filename, xml_declaration=True, encoding="utf-8")

    def DQMOM_BCs(self, coal, bc, ups_filename, gas_vel, counti):
        """Create DQMOM BC for the new ups file."""
        tree = ET.parse(ups_filename)
        root = tree.getroot()
        ################ Compute weights and weighted abscissas
        ### assume delta distributions for all abscissa except particle size, raw coal mass, ash mass.
        ### NON-Velocity internal coordinates
        particle_diameter = coal.particle_sizes  # [m] for 1 particle
        particle_volume = (np.pi/6.) * np.power(particle_diameter,3)  # [m^3] for 1 particle
        particle_mass = particle_volume * coal.density  # [kg] for 1 particle
        coal_mass = coal.coal_mass_frac * particle_mass  # [kg of raw coal] for 1 particle
        ash_mass = coal.ash_mass_frac * particle_mass  # [kg of ash] for 1 particle
        char_mass = coal.char_mass_frac * particle_mass  # [kg of char] for 1 particle
        #particle_enthalpy = coal.h_particles*1000.*coal_mass  # [J] - daf for 1 particle  (converted from [J / g] to [J / kg])
        particle_enthalpy = coal.h_particles*1000.*particle_mass  # [J] - dry for 1 particle  (converted from [J / g] to [J / kg])
        mass_fracs=coal.particle_weights # [-] mass fractions of particles with a given diameter  
        QN_mass_flow_rate=mass_fracs*bc.cfr # kg/s for each particle size      
        totalDryMassFlowRate=sum(QN_mass_flow_rate) # kg/s for each particle size
        if gas_vel==0:
            number_density=np.zeros(np.size(mass_fracs))
        else:
            number_density=QN_mass_flow_rate*(1/particle_mass)*(1/gas_vel)*(1/self.area[counti])  # kg/s * particle/kg * (s/m)gas * 1/m^2 = kg/m^3 = [# of particles / m^3] 
        ############ Read in scaling constants from base ups file
        if coal.compute_scaling_const=='true':
            w_qn_sc = np.mean(number_density)
            length_qn_sc = np.mean(particle_diameter)
            RCmass_qn_sc =  np.mean(coal_mass)
            Charmass_qn_sc = np.mean(coal_mass)
            pE_qn_sc = np.mean(particle_enthalpy)
            ux_qn_sc = 1
            uy_qn_sc = 1
            uz_qn_sc = 1
            if self.weight_constant<w_qn_sc:
                self.weight_constant=w_qn_sc
        else:
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Weights/scaling_const")
            w_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='length']/scaling_const")
            length_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='RCmass']/scaling_const")
            RCmass_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='Charmass']/scaling_const")
            Charmass_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='pE']/scaling_const")
            pE_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='ux']/scaling_const")
            ux_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='uy']/scaling_const")
            uy_qn_sc = float(scaling_const.text)
            scaling_const = root.find("./CFD/ARCHES/DQMOM/Ic/[@label='uz']/scaling_const")
            uz_qn_sc = float(scaling_const.text)
            self.weight_constant=w_qn_sc

        self.scaling_constants=[length_qn_sc,RCmass_qn_sc,Charmass_qn_sc,pE_qn_sc,ux_qn_sc,uy_qn_sc,uz_qn_sc]
        self.scaling_names=['length','RCmass','Charmass','pE','ux','uy','uz']      
        self.init_ics=np.zeros((7,np.size(particle_diameter))) 
        ################ Compute weights and weighted abscissas     
        if w_qn_sc==0:
            w_qn=number_density*0  # weights 
        else:            
            w_qn=number_density/w_qn_sc  # weights 
        length_qn = w_qn * (particle_diameter / length_qn_sc)  # length
        RCmass_qn = w_qn * (coal_mass / RCmass_qn_sc) 
        Charmass_qn = w_qn * (char_mass / Charmass_qn_sc)  
        pE_qn = w_qn * (particle_enthalpy / pE_qn_sc) 
        self.init_ics[0,]=particle_diameter # length
        self.init_ics[1,]=coal_mass # RCmass
        self.init_ics[2,]=char_mass # Charmass
        self.init_ics[3,]=particle_enthalpy # pE
        if bc.lPartMassFlowInlet==None:
            self.init_ics[4,]=[0,0,0] # ux
            self.init_ics[5,]=[0,0,0] # uy
            self.init_ics[6,]=[0,0,0] # uz  
        ## Velocity internal coordinates
            if self.orientation[counti] == 'x-':
                ux_qn = w_qn * (gas_vel / ux_qn_sc)  # vel x
                uy_qn = w_qn * (0/uy_qn_sc)  # vel y
                uz_qn = w_qn * (0/uz_qn_sc)  # vel z
            if self.orientation[counti]=='x+':
                ux_qn = w_qn * (-gas_vel / ux_qn_sc)  # vel x
                uy_qn = w_qn * (0/uy_qn_sc)  # vel y
                uz_qn = w_qn * (0/uz_qn_sc)  # vel z
            if self.orientation[counti]=='y-':
                ux_qn = w_qn * (0/ux_qn_sc)  # vel x
                uy_qn = w_qn * (gas_vel / uy_qn_sc) # vel y
                uz_qn = w_qn * (0/uz_qn_sc)  # vel z
            if self.orientation[counti]=='y+':
                ux_qn = w_qn * (0/ux_qn_sc)  # vel x
                uy_qn = w_qn * (-gas_vel / uy_qn_sc) # vel y
                uz_qn = w_qn * (0/uz_qn_sc)  # vel z
            if self.orientation[counti]=='z-':
                ux_qn = w_qn * (0/ux_qn_sc)  # vel x
                uy_qn = w_qn * (0/uy_qn_sc)  # vel y
                uz_qn = w_qn * (gas_vel / uz_qn_sc)  # vel z
            if self.orientation[counti]=='z+':
                ux_qn = w_qn * (0/ux_qn_sc)  # vel x
                uy_qn = w_qn * (0/uy_qn_sc)  # vel y
                uz_qn = w_qn * (-gas_vel / uz_qn_sc)  # vel z
        if bc.lPartMassFlowInlet==None:
            DQMOM = {'labels':{'w_qn','length_qn','ux_qn','uy_qn','uz_qn','RCmass_qn','Charmass_qn','pE_qn'}}
            DQMOM['values'] = {'w_qn':w_qn,'length_qn':length_qn,'ux_qn':ux_qn,'uy_qn':uy_qn,'uz_qn':uz_qn,'RCmass_qn':RCmass_qn,'Charmass_qn':Charmass_qn,'pE_qn':pE_qn}
        else: 
            DQMOM = {'labels':{'w_qn','length_qn','PartMassFlowInlet','RCmass_qn','Charmass_qn','pE_qn'}}
            DQMOM['values'] = {'w_qn':w_qn,'length_qn':length_qn,'PartMassFlowInlet':totalDryMassFlowRate,'RCmass_qn':RCmass_qn,'Charmass_qn':Charmass_qn,'pE_qn':pE_qn}  # length commented out since we don't use it currently

        DQMOM['NUM_QNs'] = np.size(w_qn)
        DQMOM['var_tags'] = []

        for i in DQMOM['labels']:    
            if bc.wall_boundary_condition=="true" and (i=="vel_qn" or i=="ux_qn" or i=="uy_qn" or i=="uz_qn"):
                DQMOM['var_tags'].append("Dirichlet")
            else:
                DQMOM['var_tags'].append(bc.cfr_var)
        return DQMOM

        for i in range(len(DQMOM['labels'])):
            DQMOM['var_tags'].append(bc.cfr_var)
        return DQMOM
        
    def gas_phase_BCs(self,bc, gas_vel, counti):
        """Create gas phase BC for new ups file."""
        gas = dict(labels=bc.gas_names, var_tags=bc.gas_vars,
                   values=bc.gas_values, swirl_no=bc.swirl_no)    
        # if users specifies VelocityInlet with mass_inlet tag, then we will put in the velocity instead of mdot.
        countL=0
        for j in gas['labels']:
            if j=='mass_inlet' and gas['var_tags'][countL]=='VelocityInlet':
                gas['labels'][countL]='velocity_inlet'+'_'+bc.bc_name
                #j='velocity_inlet'+str(countL)
                print "User specified VelocityInlet for a mass_inlet var. Boundary condition is being changed to a velocity_inlet."
                if self.orientation[counti] == 'x-':
                    gas['values'][countL]=[gas_vel,0,0]
                if self.orientation[counti]=='x+':
                    gas['values'][countL]=[-gas_vel,0,0]
                if self.orientation[counti]=='y-':
                    gas['values'][countL]=[0,gas_vel,0]
                if self.orientation[counti]=='y+':
                    gas['values'][countL]=[0,-gas_vel,0]
                if self.orientation[counti]=='z-':
                    gas['values'][countL]=[0,0,gas_vel]
                if self.orientation[counti]=='z+':
                    gas['values'][countL]=[0,0,-gas_vel]
            if j=='mass_inlet': # if you are using generic labels this will change their names 
                gas['labels'][countL]='mass_inlet'+'_'+bc.bc_name
            if j=='wall':  # if you are using generic labels this will change their names 
                gas['labels'][countL]='wall'+'_'+bc.bc_name     
            if j=='outlet':  # if you are using generic labels this will change their names 
                gas['labels'][countL]='outlet'+'_'+bc.bc_name                    
            countL=countL+1                     
        return gas        


        ## Function for writing new sub-ups file 
    def write_bc_file(self, gas ,DQMOM, counti, file_directory):
        """Write the bc file."""
        Uintah_Include = ET.Element('Uintah_Include')
        # loop over object 1
        count = 0
        for j in gas['labels']:
            BCType = ET.SubElement(Uintah_Include, 'BCType')
            BCType.set("id", "all")
            BCType.set("label", j)
            BCType.set("var", gas['var_tags'][count])
            value = ET.SubElement(BCType, 'value')
            value.text = str(gas['values'][count])
            if gas['var_tags'][count]== 'Swirl' :
                value = ET.SubElement(BCType, 'swirl_no')  # currently this works only if swirl is the first BC type!!!!!
                value.text = str(gas['swirl_no'][count])
            count += 1
        # loop over object 2
        count = 0
        for j in DQMOM['labels']:
            for i in range(DQMOM['NUM_QNs']):
               BCType = ET.SubElement(Uintah_Include, 'BCType')
               BCType.set("id","all")
               BCType.set("var",DQMOM['var_tags'][count])
               value = ET.SubElement(BCType, 'value')
               if  j ==  "PartMassFlowInlet":
                  BCType.set("label", j+`counti`)
                  BCType.set("var","PartMassFlowInlet")
                  value.text = str(DQMOM['values'][j])
                  break
               else:   
                  BCType.set("label", j+str(i))
                  value.text = str(DQMOM['values'][j][i])
                  BCType.set("var",DQMOM['var_tags'][count])

            count += 1
        rough_string = ET.tostring(Uintah_Include)
        reparsed = MD.parseString(rough_string)
        s = reparsed.toprettyxml(indent="  ", encoding="ISO-8859-1")
        text_file = open(file_directory+self.new_input_file_names[counti], "w")
        text_file.write(s)
        text_file.close()




