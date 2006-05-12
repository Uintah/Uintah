/**************************************************************************
// Program : MPMMaterialsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           MPM materials
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.event.*;
import java.text.DecimalFormat;
import java.util.Vector;
import java.awt.Point;

//**************************************************************************
// Class   : MPMMaterialsPanel
//**************************************************************************
public class MPMMaterialsPanel extends JPanel {

  // Static variables

  // Data
  private int d_numMat = 0;
  private UintahInputPanel d_parent = null;

  // Local components
  private WholeNumberField numMatEntry = null;
  private JTabbedPane mpmMatTabbedPane = null;
  private Vector mpmMaterialInputPanel = null;
  

  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public MPMMaterialsPanel(UintahInputPanel parent) {

    // Initialize local variables
    d_numMat = 2;
    d_parent = parent;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Label and entry for number of materials
    JPanel numMatPanel = new JPanel(new GridLayout(1,0));
    JLabel numMatLabel = new JLabel("Number of MPM Materials");
    numMatEntry = new WholeNumberField(d_numMat, 2);
    numMatPanel.add(numMatLabel);
    numMatPanel.add(numMatEntry);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(numMatLabel, gbc);
    add(numMatPanel);
    
    // Create a text field listener
    TextFieldListener textListener = new TextFieldListener();
    numMatEntry.getDocument().addDocumentListener(textListener);
    
    // Create the tabbed pane
    mpmMatTabbedPane = new JTabbedPane();

    // Create the panels for each material
    mpmMaterialInputPanel = new Vector();
    for (int ii = 0; ii < d_numMat; ++ii) {
      MPMMaterialInputPanel matPanel = new MPMMaterialInputPanel(ii);
      mpmMaterialInputPanel.addElement(matPanel);
      String matID = new String("Material "+String.valueOf(ii));
      mpmMatTabbedPane.addTab(matID, null, matPanel, null);
    }
    mpmMatTabbedPane.setSelectedIndex(0);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(mpmMatTabbedPane, gbc);
    add(mpmMatTabbedPane);

    // Create the save button
    saveButton = new JButton("Save");
    saveButton.setActionCommand("save");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 2, 1, 1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    saveButton.addActionListener(buttonListener);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to changed text 
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class TextFieldListener implements DocumentListener {
    public void insertUpdate(DocumentEvent e) {
      int new_numMat = numMatEntry.getValue();
      if (new_numMat == 0) return;
      if (new_numMat < d_numMat) {
        for (int ii = d_numMat-1; ii >= new_numMat; --ii) {
          mpmMaterialInputPanel.removeElementAt(ii);
          mpmMatTabbedPane.remove(ii);
        }
        d_numMat = new_numMat;
      } else {
        for (int ii = d_numMat; ii < new_numMat; ++ii) {
          MPMMaterialInputPanel matPanel = new MPMMaterialInputPanel(ii);
          mpmMaterialInputPanel.addElement(matPanel);
          String matID = new String("Material "+String.valueOf(ii));
          mpmMatTabbedPane.addTab(matID, null, matPanel, null);
        }
        d_numMat = new_numMat;
      }
    }
    public void removeUpdate(DocumentEvent e) {
    }
    public void changedUpdate(DocumentEvent e) {
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "save") {
        
        // Create filewriter and printwriter
        File outputFile = new File("test.ups");
        try {
          FileWriter fw = new FileWriter(outputFile);
          PrintWriter pw = new PrintWriter(fw);

          String tab = new String("    ");
          for (int ii = 0; ii < d_numMat; ++ii) {
            MPMMaterialInputPanel matPanel = 
              (MPMMaterialInputPanel) mpmMaterialInputPanel.elementAt(ii);
            matPanel.writeUintah(pw, tab);
          }

          pw.close();
          fw.close();
        } catch (Exception event) {
          System.out.println("Could not write to file "+outputFile.getName());
        }
      }
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // MPM material input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class MPMMaterialInputPanel extends JPanel 
                                      implements ItemListener {

    // Data and components
    private boolean d_isRigid = false;
    private String d_burnModel = null;
    private String d_constModel = null;

    private JTextField matNameEntry = null;
    private DecimalField densityEntry = null;
    private DecimalField thermalCondEntry = null;
    private DecimalField spHeatEntry = null;
    private DecimalField roomTempEntry = null;
    private DecimalField meltTempEntry = null;

    private JComboBox burnModelComB = null;
    private JComboBox constModelComB = null;
    private JCheckBox isRigidCB = null;
    private DecimalField bulkEntry = null;
    private DecimalField shearEntry = null;
    private DecimalField cteEntry = null;

    private JPanel constModelPanel = null;
    private ElasticPlasticPanel elasticPlasticPanel = null;
    private ViscoSCRAMPanel viscoSCRAMPanel = null;

    public MPMMaterialInputPanel(int matIndex) {

      // Initialize
      d_isRigid = false;
      d_constModel = new String("rigid");
      d_burnModel = new String("null");

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Add the first panel
      JPanel panel1 = new JPanel(new GridLayout(6,0));

      JLabel matNameLabel = new JLabel("Material Name");
      String matName = new String("Material "+ String.valueOf(matIndex));
      matNameEntry = new JTextField(matName, 20);
      panel1.add(matNameLabel);
      panel1.add(matNameEntry);

      JLabel densityLabel = new JLabel("Density");
      densityEntry = new DecimalField(8900.0, 7);
      panel1.add(densityLabel);
      panel1.add(densityEntry);

      JLabel thermalCondLabel = new JLabel("Thermal Conductivity");
      thermalCondEntry = new DecimalField(390.0, 7);
      panel1.add(thermalCondLabel);
      panel1.add(thermalCondEntry);

      JLabel spHeatLabel = new JLabel("Specific Heat");
      spHeatEntry = new DecimalField(410.0, 7);
      panel1.add(spHeatLabel);
      panel1.add(spHeatEntry);

      JLabel roomTempLabel = new JLabel("Room Temperature");
      roomTempEntry = new DecimalField(298.0, 7);
      panel1.add(roomTempLabel);
      panel1.add(roomTempEntry);

      JLabel meltTempLabel = new JLabel("Melt Temperature");
      meltTempEntry = new DecimalField(1400.0, 7);
      panel1.add(meltTempLabel);
      panel1.add(meltTempEntry);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 1, 1, 5);
      gb.setConstraints(panel1, gbc);
      add(panel1); 

      // Add the second panel
      JPanel panel2 = new JPanel(new GridLayout(6,2));

      JLabel burnModelLabel = new JLabel("Burn Model");
      burnModelComB = new JComboBox();
      burnModelComB.addItem("None");
      burnModelComB.addItem("Simple Burn");
      burnModelComB.addItem("Pressure Burn");
      burnModelComB.addItem("Ignition and Combustion");
      panel2.add(burnModelLabel); panel2.add(burnModelComB); 

      JLabel constModelLabel = new JLabel("Constitutive Model");
      constModelComB = new JComboBox();
      constModelComB.addItem("Rigid");
      constModelComB.addItem("Hypoelastic");
      constModelComB.addItem("Compressible Neo-Hookean");
      constModelComB.addItem("Elastic-Plastic");
      constModelComB.addItem("ViscoSCRAM");
      panel2.add(constModelLabel); panel2.add(constModelComB); 

      JLabel bulkLabel = new JLabel("Bulk Modulus");
      bulkEntry = new DecimalField(130.0e9, 9, true);
      panel2.add(bulkLabel);
      panel2.add(bulkEntry);

      JLabel shearLabel = new JLabel("Shear Modulus");
      shearEntry = new DecimalField(46.0e9, 9, true);
      panel2.add(shearLabel);
      panel2.add(shearEntry);

      JLabel cteLabel = new JLabel("Coefficient of Thermal Expansion");
      cteEntry = new DecimalField(1.0e-5, 9, true);
      panel2.add(cteLabel);
      panel2.add(cteEntry);

      isRigidCB = new JCheckBox("Treat Material As Rigid ");
      isRigidCB.setSelected(false);
      panel2.add(isRigidCB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 1, 1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2); 

      // Add the Third panel for the constituive models
      constModelPanel = new JPanel();
      elasticPlasticPanel = new ElasticPlasticPanel();
      viscoSCRAMPanel = new ViscoSCRAMPanel();
      constModelPanel.add(elasticPlasticPanel);
      constModelPanel.add(viscoSCRAMPanel);
      elasticPlasticPanel.setVisible(false);
      viscoSCRAMPanel.setVisible(false);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 1, 0, 1, 2, 5);
      gb.setConstraints(constModelPanel, gbc);
      add(constModelPanel); 

      // Create and add the listeners
      burnModelComB.addItemListener(this);
      constModelComB.addItemListener(this);

      CheckBoxListener checkBoxListener = new CheckBoxListener();
      isRigidCB.addItemListener(checkBoxListener);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Listens for item picked in combo box and takes action as required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void itemStateChanged(ItemEvent e) {
        
      // Get the object that has been selected
      Object source = e.getItemSelectable();

      // Get the item that has been selected
      String item = String.valueOf(e.getItem());

      if (source == burnModelComB) {
        if (item.equals(new String("None"))) {
          d_burnModel = "null";
        } else if (item.equals(new String("Simple Burn"))) {
          d_burnModel = "simple";
        } else if (item.equals(new String("Pressure Burn"))) {
          d_burnModel = "pressure";
        } else if (item.equals(new String("Ignition and Combustion"))) {
          d_burnModel = "IgnitionCombustion";
        }
      } else {
        if (item == "Rigid") {
          d_constModel = "rigid";
          elasticPlasticPanel.setVisible(false);
          viscoSCRAMPanel.setVisible(false);
          isRigidCB.setSelected(false);
        } else if (item == "Hypoelastic") {
          d_constModel = "hypoelastic";
          elasticPlasticPanel.setVisible(false);
          viscoSCRAMPanel.setVisible(false);
        } else if (item == "Compressible Neo-Hookean") {
          d_constModel = "comp_neo_hook";
          elasticPlasticPanel.setVisible(false);
          viscoSCRAMPanel.setVisible(false);
        } else if (item == "Elastic-Plastic") {
          d_constModel = "elastic_plastic";
          elasticPlasticPanel.setVisible(true);
          viscoSCRAMPanel.setVisible(false);
          validate();
        } else if (item == "ViscoSCRAM") {
          d_constModel = "visco_scram";
          elasticPlasticPanel.setVisible(false);
          viscoSCRAMPanel.setVisible(true);
          validate();
        }
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : CheckBoxListener
    // Purpose : Listens for item seleceted in check box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class CheckBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
       if (e.getStateChange() == ItemEvent.SELECTED) {
         d_isRigid = true;
       } else {
         d_isRigid = false;
       }
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw, String tab) {
      
      if (pw == null) return;

      // Write the data
      String tab1 = new String(tab+"  ");
      String tab2 = new String(tab1+"  ");
      pw.println(tab);
      pw.println(tab+"<material name = \""+matNameEntry.getText()+"\">");
      pw.println(tab1);
      pw.println(tab1+"<is_rigid> "+d_isRigid+" </is_rigid>");
      pw.println(tab1+"<density> "+densityEntry.getValue()+
                      " </density>");
      pw.println(tab1+"<thermal_conductivity> "+thermalCondEntry.getValue()+
                      " </thermal_conductivity>");
      pw.println(tab1+"<specific_heat> "+spHeatEntry.getValue()+
                      " </specific_heat>");
      pw.println(tab1+"<room_temp> "+roomTempEntry.getValue()+
                      " </room_temp>");
      pw.println(tab1+"<melt_temp> "+meltTempEntry.getValue()+
                      " </melt_temp>");

      pw.println(tab1);
      pw.println(tab1+"<constitutive_model type=\""+d_constModel+"\">");
      pw.println(tab2);
      pw.println(tab2+"<bulk_modulus> "+bulkEntry.getValue()+
                      " </bulk_modulus>");
      pw.println(tab2+"<shear_modulus> "+shearEntry.getValue()+
                      " </shear_modulus>");
      pw.println(tab2+"<coeff_thermal_expansion> "+cteEntry.getValue()+
                      " </coeff_thermal_expansion>");

      pw.println(tab2);
      if(d_constModel == "elastic_plastic") {
        elasticPlasticPanel.writeUintah(pw, tab2);
      }
      pw.println(tab1+"</constitutive_model>");
      pw.println(tab+"</material>");

    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Elastic-Plastic Material input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ElasticPlasticPanel extends JPanel 
                                    implements ActionListener {

    // Data and components
    private boolean d_isothermal = false;
    private boolean d_doMelting = false;
    private boolean d_evolvePorosity = false;
    private boolean d_evolveDamage = false;
    private boolean d_checkTEPLA = false;
    private boolean d_stressTriax = false;
    private boolean d_spHeatComp = false;

    private String d_eos = null;
    private String d_flowStress = null;
    private String d_yieldCond = null;
    private String d_shear = null;
    private String d_melt = null;
    private String d_spHeat = null;
    private String d_damage = null;
    private String d_stability = null;

    private JCheckBox isothermalCB = null;
    private JCheckBox doMeltingCB = null;
    private JCheckBox evolvePorosityCB = null;
    private JCheckBox evolveDamageCB = null;
    private JCheckBox checkTEPLACB = null;
    private JCheckBox stressTriaxCB = null;
    private JCheckBox spHeatCompCB = null;

    private DecimalField toleranceEntry = null;
    private DecimalField taylorQuinneyEntry = null;
    private DecimalField critStressEntry = null;

    private JComboBox eosComB = null;
    private JComboBox flowStressComB = null;
    private JComboBox yieldCondComB = null;
    private JComboBox shearComB = null;
    private JComboBox meltComB = null;
    private JComboBox spHeatComB = null;
    private JComboBox damageComB = null;
    private JComboBox stabilityComB = null;

    private MieGruneisenFrame mieGruneisenFrame = null;
    private JohnsonCookFlowFrame johnsonCookFlowFrame = null;
    private HancockMacKenzieDamageFrame hancockMacKenzieDamageFrame = null;

    public ElasticPlasticPanel() {

      // Initialize
      d_isothermal = false;
      d_doMelting = false;
      d_evolvePorosity = false;
      d_evolveDamage = false;
      d_checkTEPLA = false;
      d_stressTriax = false;
      d_spHeatComp = false;

      d_eos = new String("mie_gruneisen");
      d_flowStress = new String("johnson_cook");
      d_yieldCond = new String("vonMises");
      d_shear = new String("constant_shear");
      d_melt = new String("constant_Tm");
      d_spHeat = new String("constant_Cp");
      d_damage = new String("hancock_mackenzie");
      d_stability = new String("none");

      // Create a gridbaglayout and constraints
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);

      // Add the panel for the flag check boxes
      JPanel panel1 = new JPanel(new GridLayout(4,0));

      isothermalCB = new JCheckBox("Isothermal");
      doMeltingCB = new JCheckBox("Do Melting");
      spHeatCompCB = new JCheckBox("Compute Specific Heat");
      evolvePorosityCB = new JCheckBox("Evolve Porosity");
      evolveDamageCB = new JCheckBox("Evolve Damage");
      checkTEPLACB = new JCheckBox("Check TEPLA Failure Crit.");
      stressTriaxCB = new JCheckBox("Check Max. Stress Failure Crit.");

      isothermalCB.setSelected(false);
      doMeltingCB.setSelected(false);
      spHeatCompCB.setSelected(false);
      evolvePorosityCB.setSelected(false);
      evolveDamageCB.setSelected(false);
      checkTEPLACB.setSelected(false);
      stressTriaxCB.setSelected(false);

      panel1.add(isothermalCB);
      panel1.add(doMeltingCB);
      panel1.add(spHeatCompCB);
      panel1.add(evolvePorosityCB);
      panel1.add(evolveDamageCB);
      panel1.add(checkTEPLACB);
      panel1.add(stressTriaxCB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 0, 1, 1, 5);
      gb.setConstraints(panel1, gbc);
      add(panel1); 

      // Add the panel for the text fields and combo boxes
      JPanel panel2 = new JPanel(new GridLayout(11,0));

      JLabel toleranceLabel = new JLabel("Tolerance");
      JLabel taylorQuinneyLabel = new JLabel("Taylor-Quinney Coefficient");
      JLabel critStressLabel = new JLabel("Max. Triax. Stress at Failure");

      toleranceEntry = new DecimalField(1.0e-12, 9, true);
      taylorQuinneyEntry = new DecimalField(0.9, 3);
      critStressEntry = new DecimalField(5.0e9, 9, true);

      panel2.add(toleranceLabel);
      panel2.add(toleranceEntry);
      panel2.add(taylorQuinneyLabel);
      panel2.add(taylorQuinneyEntry);
      panel2.add(critStressLabel);
      panel2.add(critStressEntry);

      JLabel eosLabel = new JLabel("Equation of State Model");
      JLabel flowStressLabel = new JLabel("Flow Stress Model");
      JLabel yieldCondLabel = new JLabel("Yield Condition Model");
      JLabel shearLabel = new JLabel("Shear Modulus Model");
      JLabel meltLabel = new JLabel("Melt Temperature Model");
      JLabel spHeatLabel = new JLabel("Specific Heat Model");
      JLabel damageLabel = new JLabel("Damage Model");
      JLabel stabilityLabel = new JLabel("Material Stability Model");

      eosComB = new JComboBox();
      flowStressComB = new JComboBox();
      yieldCondComB = new JComboBox();
      shearComB = new JComboBox();
      meltComB = new JComboBox();
      spHeatComB = new JComboBox();
      damageComB = new JComboBox();
      stabilityComB = new JComboBox();

      eosComB.addItem("Mie-Gruneisen");
      eosComB.addItem("None");

      flowStressComB.addItem("Johnson-Cook");
      flowStressComB.addItem("Mechanical Threshold Stress");
      flowStressComB.addItem("Preston-Tonks-Wallace");
      flowStressComB.addItem("Zerilli-Armstrong");
      flowStressComB.addItem("Steinberg-Cochran-Guinan");
      flowStressComB.addItem("Linear");

      yieldCondComB.addItem("von Mises");
      yieldCondComB.addItem("Gurson-Tvergaard-Needleman");

      shearComB.addItem("Constant");
      shearComB.addItem("Chen-Gray");
      shearComB.addItem("Steinberg-Guinan");
      shearComB.addItem("Nadal-Le Poac");

      meltComB.addItem("Constant");
      meltComB.addItem("Steinberg-Guinan");
      meltComB.addItem("Burakovsky-Preston-Silbar");

      spHeatComB.addItem("Constant");
      spHeatComB.addItem("Copper Model");
      spHeatComB.addItem("Steel Model");
      spHeatComB.addItem("Aluminum Model");

      damageComB.addItem("Hancock-MacKenzie");
      damageComB.addItem("Johnson-Cook");

      stabilityComB.addItem("None");
      stabilityComB.addItem("Drucker Stability");
      stabilityComB.addItem("Acoustic Tensor Stability");
      stabilityComB.addItem("Drucker + Acoustic Tensor");

      panel2.add(eosLabel);
      panel2.add(eosComB);
      panel2.add(flowStressLabel);
      panel2.add(flowStressComB);
      panel2.add(yieldCondLabel);
      panel2.add(yieldCondComB);
      panel2.add(shearLabel);
      panel2.add(shearComB);
      panel2.add(meltLabel);
      panel2.add(meltComB);
      panel2.add(spHeatLabel);
      panel2.add(spHeatComB);
      panel2.add(damageLabel);
      panel2.add(damageComB);
      panel2.add(stabilityLabel);
      panel2.add(stabilityComB);

      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                               1.0, 1.0, 0, 1, 1, 1, 5);
      gb.setConstraints(panel2, gbc);
      add(panel2); 

      // Add frames for the models
      mieGruneisenFrame = new MieGruneisenFrame();
      johnsonCookFlowFrame = new JohnsonCookFlowFrame();
      hancockMacKenzieDamageFrame = new HancockMacKenzieDamageFrame();

      mieGruneisenFrame.pack();
      johnsonCookFlowFrame.pack();
      hancockMacKenzieDamageFrame.pack();

      mieGruneisenFrame.setVisible(false);
      johnsonCookFlowFrame.setVisible(false);
      hancockMacKenzieDamageFrame.setVisible(false);

      // Create and add the listeners
      CheckBoxListener checkBoxListener = new CheckBoxListener();
      isothermalCB.addItemListener(checkBoxListener);
      doMeltingCB.addItemListener(checkBoxListener);
      spHeatCompCB.addItemListener(checkBoxListener);
      evolvePorosityCB.addItemListener(checkBoxListener);
      evolveDamageCB.addItemListener(checkBoxListener);
      checkTEPLACB.addItemListener(checkBoxListener);
      stressTriaxCB.addItemListener(checkBoxListener);

      eosComB.addActionListener(this);
      flowStressComB.addActionListener(this);
      yieldCondComB.addActionListener(this);
      shearComB.addActionListener(this);
      meltComB.addActionListener(this);
      spHeatComB.addActionListener(this);
      damageComB.addActionListener(this);
      stabilityComB.addActionListener(this);

    }

    //-----------------------------------------------------------------------
    // Listens for item selected in combo box and takes action as needed.
    //-----------------------------------------------------------------------
    public void actionPerformed(ActionEvent e) {

      // Find the object that has been selected
      JComboBox source = (JComboBox) e.getSource();

      // Get the item that has been selected
      String item = (String) source.getSelectedItem();

      Point location = getParent().getLocation();

      validate();

      if (source == eosComB) {
        if (item == "Mie-Gruneisen") {
          d_eos = "mie_gruneisen";
          mieGruneisenFrame.setLocation(location);
          mieGruneisenFrame.setVisible(true);
        } else if (item == "None") {
          d_eos = "none";
          mieGruneisenFrame.setVisible(false);
        }
      } else if (source == flowStressComB) {
        if (item == "Johnson-Cook") {
          d_flowStress = "johnson_cook";
          johnsonCookFlowFrame.setLocation(location);
          johnsonCookFlowFrame.setVisible(true);
        } else if (item == "Mechanical Threshold Stress") {
          johnsonCookFlowFrame.setVisible(false);
        } else if (item == "Preston-Tonks-Wallace") {
          johnsonCookFlowFrame.setVisible(false);
        } else if (item == "Zerilli-Armstrong") {
          johnsonCookFlowFrame.setVisible(false);
        } else if (item == "Steinberg-Cochran-Guinan") {
          johnsonCookFlowFrame.setVisible(false);
        } else if (item == "Linear") {
          johnsonCookFlowFrame.setVisible(false);
        }
      } else if (source == yieldCondComB) {
        if (item == "von Mises") {
          d_yieldCond = "vonMises";
        } else if (item == "Gurson-Tvergaard-Needleman") {
        }
      } else if (source == shearComB) {
        if (item == "Constant") {
          d_shear = "constant_shear";
        } else if (item == "Chen-Gray") {
        } else if (item == "Steinberg-Guinan") {
        } else if (item == "Nadal-Le Poac") {
        }
      } else if (source == meltComB) {
        if (item == "Constant") {
          d_melt = "constant_Tm";
        } else if (item == "Steinberg-Guinan") {
        } else if (item == "Burakovsky-Preston-Silbar") {
        }
      } else if (source == spHeatComB) {
        if (item == "Constant") {
          d_spHeat = "constant_Cp";
        } else if (item == "Copper Model") {
        } else if (item == "Steel Model") {
        } else if (item == "Aluminum Model") {
        }
      } else if (source == damageComB) {
        if (item == "Johnson-Cook") {
          d_damage = "johnson_cook";
          hancockMacKenzieDamageFrame.setVisible(false);
        } else if (item == "Hancock-MacKenzie") {
          d_damage = "hancock_mackenzie";
          hancockMacKenzieDamageFrame.setLocation(location);
          hancockMacKenzieDamageFrame.setVisible(true);
        }
      } else if (source == stabilityComB) {
        if (item == "None") {
          d_stability = "none";
        } else if (item == "Drucker Stability") {
        } else if (item == "Acoustic Tensor Stability") {
        } else if (item == "Drucker + Acoustic Tensor") {
        }
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : CheckBoxListener
    // Purpose : Listens for item seleceted in check box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class CheckBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
        // Find the object that has changed
        Object source = e.getItemSelectable();

        if (source == isothermalCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_isothermal = true;
          } else {
            d_isothermal = false;
          }
        } else if (source == doMeltingCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_doMelting = true;
          } else {
            d_doMelting = false;
          }
        } else if (source == spHeatCompCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_spHeatComp = true;
          } else {
            d_spHeatComp = false;
          }
        } else if (source == evolvePorosityCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_evolvePorosity = true;
          } else {
            d_evolvePorosity = false;
          }
        } else if (source == evolveDamageCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_evolveDamage = true;
          } else {
            d_evolveDamage = false;
          }
        } else if (source == checkTEPLACB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_checkTEPLA = true;
          } else {
            d_checkTEPLA = false;
          }
        } else if (source == stressTriaxCB) {
          if (e.getStateChange() == ItemEvent.SELECTED) {
            d_stressTriax = true;
          } else {
            d_stressTriax = false;
          }
        }
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw, String tab1) {

        if (pw == null) return;

        String tab2 = new String(tab1+"  ");
        pw.println(tab1+"<isothermal> "+d_isothermal+" </isothermal>");
        pw.println(tab1+"<do_melting> "+d_doMelting+" </do_melting>");
        pw.println(tab1+"<evolve_porosity> "+d_evolvePorosity+
                        " </evolve_porosity>");
        pw.println(tab1+"<evolve_damage> "+d_evolveDamage+
                        " </evolve_damage>");
        pw.println(tab1+"<check_TEPLA_failure_criterion> "+d_checkTEPLA+
                        " </check_TEPLA_failure_criterion>");
        pw.println(tab1+"<check_max_stress_failure> "+d_stressTriax+
                        " </check_max_stress_failure>");
        pw.println(tab1+"<compute_specific_heat> "+d_spHeatComp+
                        " </compute_specific_heat>");

        pw.println(tab1);
        pw.println(tab1+"<tolerance> "+toleranceEntry.getValue()+
                        " </tolerance>");
        pw.println(tab1+"<taylor_quinney_coeff> "+taylorQuinneyEntry.getValue()+
                        " </taylor_quinney_coeff>");
        pw.println(tab1+"<critical_stress> "+critStressEntry.getValue()+
                        " </critical_stress>");

        pw.println(tab1);
        pw.println(tab1+"<equation_of state type = \""+d_eos+"\">");
        if (d_eos.equals(new String("mie_gruneisen"))) {
          mieGruneisenFrame.writeUintah(pw, tab2);
        }
        pw.println(tab1+"</equation_of state>");

        pw.println(tab1);
        pw.println(tab1+"<plasticity_model type = \""+d_flowStress+"\">");
        if (d_flowStress.equals(new String("johnson_cook"))) {
          johnsonCookFlowFrame.writeUintah(pw, tab2);
        }
        pw.println(tab1+"</plasticity_model>");

        pw.println(tab1);
        pw.println(tab1+"<yield_condition type = \""+d_yieldCond+"\">");
        pw.println(tab1+"</yield_condition>");

        pw.println(tab1);
        pw.println(tab1+"<shear_modulus_model type = \""+d_shear+"\">");
        pw.println(tab1+"</shear_modulus_model>");

        pw.println(tab1);
        pw.println(tab1+"<melting_temp_model type = \""+d_melt+"\">");
        pw.println(tab1+"</melting_temp_model>");

        pw.println(tab1);
        pw.println(tab1+"<specific_heat_model type = \""+d_spHeat+"\">");
        pw.println(tab1+"</specific_heat_model>");

        pw.println(tab1);
        pw.println(tab1+"<damage_model type = \""+d_damage+"\">");
        if (d_damage.equals(new String("hancock_mackenzie"))) {
          hancockMacKenzieDamageFrame.writeUintah(pw, tab2);
        }
        pw.println(tab1+"</damage_model>");

        pw.println(tab1);
        pw.println(tab1+"<stability_check type = \""+d_stability+"\">");
        pw.println(tab1+"</stability_check>");
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : MieGruneisenFrame
    // Purpose : Takes inputs for Mie-Gruneisen EOS
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class MieGruneisenFrame extends JFrame 
                                    implements ActionListener,
                                               DocumentListener {

      double d_c0 = 0.0;
      double d_gamma0 = 0.0;
      double d_salpha = 0.0;

      DecimalField c0Entry = null;
      DecimalField gamma0Entry = null;
      DecimalField salphaEntry = null;

      JButton closeButton = null;

      public MieGruneisenFrame() {

        // Set the title
        setLocation(100,100);
        setTitle("Mie-Gruneisen EOS Model");

        // Create a gridbaglayout and constraints
        GridBagLayout gb = new GridBagLayout();
        GridBagConstraints gbc = new GridBagConstraints();
        getContentPane().setLayout(gb);

        // Create a panel for the inputs
        JPanel panel = new JPanel(new GridLayout(3,0));

        JLabel c0Label = new JLabel("C_0");
        JLabel gamma0Label = new JLabel("gamma_0");
        JLabel salphaLabel = new JLabel("S_alpha");

        d_c0 = 3940.0;
        d_gamma0 = 2.02;
        d_salpha = 1.489;

        c0Entry = new DecimalField(d_c0, 6);
        gamma0Entry = new DecimalField(d_gamma0, 6);
        salphaEntry = new DecimalField(d_salpha, 6);
        c0Entry.getDocument().addDocumentListener(this);
        gamma0Entry.getDocument().addDocumentListener(this);
        salphaEntry.getDocument().addDocumentListener(this);

        panel.add(c0Label);
        panel.add(c0Entry);
        panel.add(gamma0Label);
        panel.add(gamma0Entry);
        panel.add(salphaLabel);
        panel.add(salphaEntry);

        UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
        gb.setConstraints(panel, gbc);
        getContentPane().add(panel);

        // Create the close button
        closeButton = new JButton("Close");
        closeButton.setActionCommand("close");
        closeButton.addActionListener(this);
        UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
        gb.setConstraints(closeButton, gbc);
        getContentPane().add(closeButton);
      }

      // Respond to changed text
      public void insertUpdate(DocumentEvent e) {
        d_c0 = c0Entry.getValue();
        d_gamma0 = gamma0Entry.getValue();
        d_salpha = salphaEntry.getValue();
      }

      public void removeUpdate(DocumentEvent e) {
      }

      public void changedUpdate(DocumentEvent e) {
      }

      public void actionPerformed(ActionEvent e) {
        if (e.getActionCommand() == "close") {
          setVisible(false);
        }
      }

      public void writeUintah(PrintWriter pw, String tab) {

        if (pw == null) return;

        pw.println(tab+"<C_0> "+d_c0+" </C_0>");
        pw.println(tab+"<Gamma_0> "+d_gamma0+" </Gamma_0>");
        pw.println(tab+"<S_alpha> "+d_salpha+" </S_alpha>");
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : JohnsonCookFlowFrame
    // Purpose : Takes inputs for Johnson-Cook flow model
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class JohnsonCookFlowFrame extends JFrame 
                                       implements ActionListener {

      DecimalField c_AEntry = null;
      DecimalField c_BEntry = null;
      DecimalField c_CEntry = null;
      DecimalField c_nEntry = null;
      DecimalField c_mEntry = null;
      DecimalField c_epdot0Entry = null;
      DecimalField c_TrEntry = null;
      DecimalField c_TmEntry = null;

      JButton closeButton = null;

      public JohnsonCookFlowFrame() {

        // Set the title
        setLocation(150,150);
        setTitle("Johnson-Cook Flow Model");

        // Create a gridbaglayout and constraints
        GridBagLayout gb = new GridBagLayout();
        GridBagConstraints gbc = new GridBagConstraints();
        getContentPane().setLayout(gb);

        // Create a panel for the inputs
        JPanel panel = new JPanel(new GridLayout(8,0));

        JLabel c_ALabel = new JLabel("A");
        JLabel c_BLabel = new JLabel("B");
        JLabel c_CLabel = new JLabel("C");
        JLabel c_nLabel = new JLabel("n");
        JLabel c_mLabel = new JLabel("m");
        JLabel c_epdot0Label = new JLabel("epdot_0");
        JLabel c_TrLabel = new JLabel("T_r");
        JLabel c_TmLabel = new JLabel("T_m");

        c_AEntry = new DecimalField(90.0e6, 9, true);
        c_BEntry = new DecimalField(292.0e6, 9, true);
        c_CEntry = new DecimalField(0.025, 6);
        c_nEntry = new DecimalField(0.31, 6);
        c_mEntry = new DecimalField(1.09, 6);
        c_epdot0Entry = new DecimalField(1.0, 9, true);
        c_TrEntry = new DecimalField(298.0, 6);
        c_TmEntry = new DecimalField(1793.0, 6);

        panel.add(c_ALabel);
        panel.add(c_AEntry);
        panel.add(c_BLabel);
        panel.add(c_BEntry);
        panel.add(c_CLabel);
        panel.add(c_CEntry);
        panel.add(c_nLabel);
        panel.add(c_nEntry);
        panel.add(c_mLabel);
        panel.add(c_mEntry);
        panel.add(c_epdot0Label);
        panel.add(c_epdot0Entry);
        panel.add(c_TrLabel);
        panel.add(c_TrEntry);
        panel.add(c_TmLabel);
        panel.add(c_TmEntry);

        UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
        gb.setConstraints(panel, gbc);
        getContentPane().add(panel);

        // Create the close button
        closeButton = new JButton("Close");
        closeButton.setActionCommand("close");
        closeButton.addActionListener(this);
        UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
        gb.setConstraints(closeButton, gbc);
        getContentPane().add(closeButton);
      }

      public void actionPerformed(ActionEvent e) {
        if (e.getActionCommand() == "close") {
          setVisible(false);
        }
      }

      public void writeUintah(PrintWriter pw, String tab) {

        if (pw == null) return;

        pw.println(tab+"<A> "+c_AEntry.getValue()+" </A>");
        pw.println(tab+"<B> "+c_BEntry.getValue()+" </B>");
        pw.println(tab+"<C> "+c_CEntry.getValue()+" </C>");
        pw.println(tab+"<n> "+c_nEntry.getValue()+" </n>");
        pw.println(tab+"<m> "+c_mEntry.getValue()+" </m>");
        pw.println(tab+"<epdot_0> "+c_epdot0Entry.getValue()+" </epdot_0>");
        pw.println(tab+"<Tr> "+c_TrEntry.getValue()+" </Tr>");
        pw.println(tab+"<Tm> "+c_TrEntry.getValue()+" </Tm>");
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : Hancock-MacKenzieDamageFrame
    // Purpose : Takes inputs for Hancok MacKenzie damage model
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class HancockMacKenzieDamageFrame extends JFrame 
                                              implements ActionListener {

      DecimalField d0Entry = null;
      DecimalField dCritEntry = null;

      JButton closeButton = null;

      public HancockMacKenzieDamageFrame() {

        // Set the title
        setLocation(200,200);
        setTitle("Hancock-MacKenzie Damage Model");

        // Create a gridbaglayout and constraints
        GridBagLayout gb = new GridBagLayout();
        GridBagConstraints gbc = new GridBagConstraints();
        getContentPane().setLayout(gb);

        // Create a panel for the inputs
        JPanel panel = new JPanel(new GridLayout(2,0));

        JLabel d0Label = new JLabel("D_0");
        JLabel dCritLabel = new JLabel("D_crit");

        d0Entry = new DecimalField(0.0001, 6);
        dCritEntry = new DecimalField(0.7, 6);

        panel.add(d0Label);
        panel.add(d0Entry);
        panel.add(dCritLabel);
        panel.add(dCritEntry);

        UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
        gb.setConstraints(panel, gbc);
        getContentPane().add(panel);

        // Create the close button
        closeButton = new JButton("Close");
        closeButton.setActionCommand("close");
        closeButton.addActionListener(this);
        UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
        gb.setConstraints(closeButton, gbc);
        getContentPane().add(closeButton);
      }

      public void actionPerformed(ActionEvent e) {
        if (e.getActionCommand() == "close") {
          setVisible(false);
        }
      }

      public void writeUintah(PrintWriter pw, String tab) {

        if (pw == null) return;

        pw.println(tab+"<D0> "+d0Entry.getValue()+" </D0>");
        pw.println(tab+"<Dc> "+dCritEntry.getValue()+" </Dc>");
      }
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ViscoSCRAM Material input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ViscoSCRAMPanel extends JPanel {

    // Data and components
    public ViscoSCRAMPanel() {

    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : ComboBoxListener
    // Purpose : Listens for item picked in combo box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class ComboBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
      }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Class   : CheckBoxListener
    // Purpose : Listens for item seleceted in check box and takes action as
    //           required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    private class CheckBoxListener implements ItemListener {
      public void itemStateChanged(ItemEvent e) {
        
      }
    }

    //--------------------------------------------------------------------
    /** Write the contents out in Uintah format */
    //--------------------------------------------------------------------
    public void writeUintah(PrintWriter pw) {
      
      if (pw == null) return;

      // Write the data
    }
  }
}
