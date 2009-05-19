/**************************************************************************
// Program : InputPartDistPanel.java
// Purpose : Create a panel that contains widgets to 
//           1) read particle distribution information from 
//                a) the screen
//                b) a file
//           2) save that information to a file.
//           3) calculate the actual number of particles of each size
//              fraction in 100 spherical particles.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.text.DecimalFormat;

//**************************************************************************
// Class   : InputPartDistPanel
// Purpose : The following components are realized in the panel :
//             1) A gridbag layout that contains the components.
//             2) A button that calls a routine to read the data from a
//                file and display (so that it may be edited).
//             3) A button that saves the data .. no check is put on whether
//                the data have been saved or not so that if the user does
//                not save the data by pressing this button it will be lost.
//             4) A set of text input components that take all the available
//                information for each type of particle.
//**************************************************************************
public class InputPartDistPanel extends JPanel {

  // Static variables

  // Data
  private ParticleSize d_partSizeDist = null;
  private ParticleSizeDistInputPanel d_parent = null;

  private boolean alreadyCalculated = false;

  // Components that are realized
  private JButton readButton = null;
  private JButton calcButton = null;
  private JButton saveButton = null;
  public JTextField matNameEntry = null;
  public DecimalField volFracEntry = null;
  public DecimalField size11Entry = null;
  public DecimalField frac11Entry = null;
  public DecimalField size12Entry = null;
  public DecimalField frac12Entry = null;
  public DecimalField size13Entry = null;
  public DecimalField frac13Entry = null;
  public DecimalField size14Entry = null;
  public DecimalField frac14Entry = null;
  public DecimalField size15Entry = null;
  public DecimalField frac15Entry = null;
  public DecimalField size16Entry = null;
  public DecimalField frac16Entry = null;
  
  public InputPartDistPanel(ParticleSize partSizeDist,
                            ParticleSizeDistInputPanel parent) {

    // Initialize local variables
    d_partSizeDist = partSizeDist;
    d_parent = parent;

    // set the size
    //setLocation(500,50);

    // Set the flags to be false
    alreadyCalculated = false;

    // There are six panels that contain various components
    //  1) A panel containing a button "Read From File".
    //  2) A panel containing three text input components.
    //  3) A panel containing a table for size distribution inputs
    //  4) A panel containing the save button.

    // Create the panels
    JPanel panel1 = new JPanel(new GridLayout(1,0));
    JPanel panel2 = new JPanel(new GridLayout(4,0));
    JPanel panel4 = new JPanel(new GridLayout(8,0));
    JPanel panel6 = new JPanel(new GridLayout(1,0));

    // Create the components for each panel
    // Panel 1
    readButton = new JButton("Read from File");
    readButton.setActionCommand("read");
    panel1.add(readButton);

    // Panel 2
    JLabel label21 = new JLabel("Composite Material Name");
    JLabel label23 = new JLabel("Vol. Frac. of Particles in Composite (%)");
    matNameEntry = new JTextField(d_partSizeDist.compositeName, 20);
    volFracEntry = new DecimalField(100.0,5);
    panel2.add(label21);
    panel2.add(matNameEntry);
    panel2.add(label23);
    panel2.add(volFracEntry);

    // Panel 4
    JLabel label41 = new JLabel("Size Distribution");
    JLabel label42 = new JLabel("of Particles");
    JLabel label43 = new JLabel("Size <=");
    JLabel label44 = new JLabel("Fraction (volume %)");
    size11Entry = new DecimalField(d_partSizeDist.sizeInp[0], 4);
    frac11Entry = new DecimalField(d_partSizeDist.volFracInp[0], 5);
    size12Entry = new DecimalField(d_partSizeDist.sizeInp[1], 4);
    frac12Entry = new DecimalField(d_partSizeDist.volFracInp[0]+
                                   d_partSizeDist.volFracInp[1], 5);
    size13Entry = new DecimalField(0.0, 4);
    frac13Entry = new DecimalField(0.0, 5);
    size14Entry = new DecimalField(0.0, 4);
    frac14Entry = new DecimalField(0.0, 5);
    size15Entry = new DecimalField(0.0, 4);
    frac15Entry = new DecimalField(0.0, 5);
    size16Entry = new DecimalField(0.0, 4);
    frac16Entry = new DecimalField(0.0, 5);
    panel4.add(label41);
    panel4.add(label42);
    panel4.add(label43);
    panel4.add(label44);
    panel4.add(size11Entry);
    panel4.add(frac11Entry);
    panel4.add(size12Entry);
    panel4.add(frac12Entry);
    panel4.add(size13Entry);
    panel4.add(frac13Entry);
    panel4.add(size14Entry);
    panel4.add(frac14Entry);
    panel4.add(size15Entry);
    panel4.add(frac15Entry);
    panel4.add(size16Entry);
    panel4.add(frac16Entry);

    // Panel 6
    calcButton = new JButton("Calculate Particle Distribution");
    calcButton.setActionCommand("calc");
    saveButton = new JButton("Save to File");
    saveButton.setActionCommand("save");
    panel6.add(calcButton);
    panel6.add(saveButton);

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,0, 1,1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,1, 1,1, 5);
    gb.setConstraints(panel2, gbc);
    add(panel2);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,2, 1,1, 5);
    gb.setConstraints(panel4, gbc);
    add(panel4);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,4, 1,1, 5);
    gb.setConstraints(panel6, gbc);
    add(panel6);
    
    // Create and add listeners for the input components
    TextFieldListener textFieldListener = new TextFieldListener();
    matNameEntry.getDocument().addDocumentListener(textFieldListener);
    volFracEntry.getDocument().addDocumentListener(textFieldListener);
    size11Entry.getDocument().addDocumentListener(textFieldListener);
    frac11Entry.getDocument().addDocumentListener(textFieldListener);
    size12Entry.getDocument().addDocumentListener(textFieldListener);
    frac12Entry.getDocument().addDocumentListener(textFieldListener);
    size13Entry.getDocument().addDocumentListener(textFieldListener);
    frac13Entry.getDocument().addDocumentListener(textFieldListener);
    size14Entry.getDocument().addDocumentListener(textFieldListener);
    frac14Entry.getDocument().addDocumentListener(textFieldListener);
    size15Entry.getDocument().addDocumentListener(textFieldListener);
    frac15Entry.getDocument().addDocumentListener(textFieldListener);
    size16Entry.getDocument().addDocumentListener(textFieldListener);
    frac16Entry.getDocument().addDocumentListener(textFieldListener);
    
    ButtonListener buttonListener = new ButtonListener();
    readButton.addActionListener(buttonListener);
    calcButton.addActionListener(buttonListener);
    saveButton.addActionListener(buttonListener);
  }

  // Respond to changed text
  class TextFieldListener implements DocumentListener {
    public void insertUpdate(DocumentEvent e) {
      updatePartSizeDistFromInput();
    }
    public void removeUpdate(DocumentEvent e) {
      updatePartSizeDistFromInput();
    }
    public void changedUpdate(DocumentEvent e) {
      updatePartSizeDistFromInput();
    }
  }

  // Respond to button pressed (inner class button listener)
  class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "read") {
        readFromFile();
      }
      else if (e.getActionCommand() == "save") {
        if (!alreadyCalculated) calcParticleDist();
        saveToFile();
      }
      else if (e.getActionCommand() == "calc") {
        calcParticleDist();
        alreadyCalculated = true;
      }
    }
  }

  // Read the particle size distribution and other data from a text file.
  private void readFromFile() {

    // Get the name of the file 
    File file = null;
    JFileChooser fc = new JFileChooser(new File(".."));
    int returnVal = fc.showOpenDialog(this);
    if (returnVal == JFileChooser.APPROVE_OPTION) {
      file = fc.getSelectedFile();
    }

    if (file == null) return;

    try {
      FileReader fr = new FileReader(file);
      StreamTokenizer st = new StreamTokenizer(fr);
      // st.parseNumbers();
      st.commentChar('#');
      st.quoteChar('"');
      int count = 0;
      while ( st.nextToken() != StreamTokenizer.TT_EOF) {
        count++;
        switch (count) {
        case 1: 
          matNameEntry.setText(st.sval);
          break;
        case 2: 
          volFracEntry.setText(String.valueOf(st.nval));
          break;
        case 3: 
          size11Entry.setValue(String.valueOf(st.nval));
          break;
        case 4: 
          frac11Entry.setText(String.valueOf(st.nval));
          break;
        case 5: 
          size12Entry.setValue(String.valueOf(st.nval));
          break;
        case 6: 
          frac12Entry.setText(String.valueOf(st.nval));
          break;
        case 7: 
          size13Entry.setValue(String.valueOf(st.nval));
          break;
        case 8: 
          frac13Entry.setText(String.valueOf(st.nval));
          break;
        case 9: 
          size14Entry.setValue(String.valueOf(st.nval));
          break;
        case 10: 
          frac14Entry.setText(String.valueOf(st.nval));
          break;
        case 11: 
          size15Entry.setValue(String.valueOf(st.nval));
          break;
        case 12: 
          frac15Entry.setText(String.valueOf(st.nval));
          break;
        case 13: 
          size16Entry.setValue(String.valueOf(st.nval));
          break;
        case 14: 
          frac16Entry.setText(String.valueOf(st.nval));
          break;
        }
      }
      fr.close();
    } catch (Exception e) {
      System.out.println("Could not read "+file.getName());
    }

    // Populate d_partSizeDist
    updatePartSizeDistFromInput();

    // Update the histogram display
    d_parent.refreshDisplayPartDistFrame();
  }

  //  Update the d_partSizeDist data structure
  private void updatePartSizeDistFromInput() {

    String matName = matNameEntry.getText(); 
    double volFracOfParticles = volFracEntry.getValue();

    d_partSizeDist.compositeName = matName;
    d_partSizeDist.volFracInComposite = volFracOfParticles;

    // Read the size and vol % data
    int nofSizes = 0;
    try {

      // Line 1
      double size = size11Entry.getValue(); 
      double volFrac = frac11Entry.getValue(); 
      if (size != 0) {
        d_partSizeDist.sizeInp[nofSizes] = size;
        d_partSizeDist.volFracInp[nofSizes] = volFrac;
        ++nofSizes;
      }

      // Line 2
      size = size12Entry.getValue(); 
      volFrac = frac12Entry.getValue(); 
      if (size != 0) {
        d_partSizeDist.sizeInp[nofSizes] = size;
        d_partSizeDist.volFracInp[nofSizes] = 
          volFrac - frac11Entry.getValue();
        ++nofSizes;
      }

      // Line 3
      size = size13Entry.getValue(); 
      volFrac = frac13Entry.getValue(); 
      if (size != 0) {
        d_partSizeDist.sizeInp[nofSizes] = size;
        d_partSizeDist.volFracInp[nofSizes] = 
          volFrac - frac12Entry.getValue();
        ++nofSizes;
      }

      // Line 4
      size = size14Entry.getValue(); 
      volFrac = frac14Entry.getValue(); 
      if (size != 0) {
        d_partSizeDist.sizeInp[nofSizes] = size;
        d_partSizeDist.volFracInp[nofSizes] = 
          volFrac - frac13Entry.getValue();
        ++nofSizes;
      }

      // Line 5
      size = size15Entry.getValue(); 
      volFrac = frac15Entry.getValue(); 
      if (size != 0) {
        d_partSizeDist.sizeInp[nofSizes] = size;
        d_partSizeDist.volFracInp[nofSizes] = 
          volFrac - frac14Entry.getValue();
        ++nofSizes;
      }

      // Line 6
      size = size16Entry.getValue(); 
      volFrac = frac16Entry.getValue(); 
      if (size != 0) {
        d_partSizeDist.sizeInp[nofSizes] = size;
        d_partSizeDist.volFracInp[nofSizes] = 
          volFrac - frac15Entry.getValue();
        ++nofSizes;
      }

      d_partSizeDist.nofSizesInp = nofSizes;

    } catch (Exception e) {
      System.out.println("Error reading size/vol % data from entry");
    }
  }

  // Save the particle size distribution and other data to a text file.
  private void saveToFile() {

    // Update the data structure
    updatePartSizeDistFromInput();

    // Get the name of the file 
    File file = null;
    JFileChooser fc = new JFileChooser(new File(".."));
    int returnVal = fc.showSaveDialog(this);
    if (returnVal == JFileChooser.APPROVE_OPTION) {
      file = fc.getSelectedFile();
    }

    if (file == null) return;

    try {
      // Create a FileWriter and an associated printwriter
      FileWriter fw = new FileWriter(file);
      PrintWriter pw = new PrintWriter(fw);

      // Write the data
      pw.println("# Particle Size Distribution Data");
      pw.println("# Material Name");
      pw.println("\""+matNameEntry.getText()+"\"");
      pw.println("# Volume Fraction of Particles in Composite");
      pw.println(volFracEntry.getText());
      pw.println("# Size Distribution");
      pw.println("# Size (<= )   % (volume)");
      pw.println(size11Entry.getText()+" "+frac11Entry.getText());
      pw.println(size12Entry.getText()+" "+frac12Entry.getText());
      pw.println(size13Entry.getText()+" "+frac13Entry.getText());
      pw.println(size14Entry.getText()+" "+frac14Entry.getText());
      pw.println(size15Entry.getText()+" "+frac15Entry.getText());
      pw.println(size16Entry.getText()+" "+frac16Entry.getText());
      pw.close();
      fw.close();
    } catch (Exception e) {
      System.out.println("Could not write "+file.getName());
    }

    // Update the histogram display
    d_parent.refreshDisplayPartDistFrame();
  }

  // Calculate the particle distribution to be used in the generation
  // of random particle locations for the Generalized Method of Cells
  // and other applications
  private void calcParticleDist() {

    // The max number of calculated particle sizes is 10
    int NUM_SIZES_MAX = 11;
    int LARGE_INT = 100000;

    // Read the (almost) continuous distribution
    int nofSizesInp = d_partSizeDist.nofSizesInp;
    if (nofSizesInp == 0) return;

    double[] sizeInp = new double[nofSizesInp];
    double[] volFracInp = new double[nofSizesInp]; 
    for (int ii = 0; ii < nofSizesInp; ii++) {
      sizeInp[ii] = d_partSizeDist.sizeInp[ii];
      volFracInp[ii] = d_partSizeDist.volFracInp[ii];
    }
    
    // If the distribution contains only one size then the material
    // is monodispersed and we don't need any further information
    if (nofSizesInp == 1) {

      // Copy the rest of the data into d_PartSizeDist
      System.out.println("Composite");
      System.out.println("Size .. Number ");
      int totBalls = LARGE_INT;
      d_partSizeDist.nofSizesCalc = 1;
      d_partSizeDist.sizeCalc[0] = d_partSizeDist.sizeInp[0];
      d_partSizeDist.freq2DCalc[0] = totBalls;
      d_partSizeDist.freq3DCalc[0] = totBalls;
      System.out.println(sizeInp[0]+"    "+totBalls);
      System.out.println(" Total   "+totBalls);
      return;
    }

    // Compute a range of mean sizes and mean volume fractions
    double[] meanSizeCalc = new double[NUM_SIZES_MAX];
    double[] meanVolFracCalc = new double[NUM_SIZES_MAX];
    if (nofSizesInp > 0) {
      double minSize = sizeInp[0];
      double maxSize = sizeInp[nofSizesInp-1];
      double sizeIncr = (maxSize-minSize)/(NUM_SIZES_MAX-1);
      double[] sizeCalc = new double[NUM_SIZES_MAX];
      if (volFracInp[0] > 0.0) {
        sizeCalc[0] = 0.5*minSize;
      } else {
        sizeCalc[0] = minSize;
      }
      for (int ii = 1; ii < NUM_SIZES_MAX; ++ii) {
        sizeCalc[ii] = minSize + ii*sizeIncr;
      }
      for (int ii = 0; ii < NUM_SIZES_MAX-1; ++ii) {
        double size_start = sizeCalc[ii];
        double size_end = sizeCalc[ii+1];
        meanSizeCalc[ii] = 0.5*(size_start + size_end);

        double intp = 0.0;
        for (int jj = 0; jj < nofSizesInp; ++jj) {
          size_start = 0.0;
          if (jj > 0) size_start = sizeInp[jj-1];
          size_end = sizeInp[jj];
          double tt = (meanSizeCalc[ii]-size_start)/(size_end - size_start);
          if (tt >= 0.0 && tt <= 1.0) {
            if (jj > 0) {
              intp = (1.0-tt)*volFracInp[jj-1]+tt*volFracInp[jj];
            } else {
              intp = tt*volFracInp[jj];
            }
            break;
          }
        }
        meanVolFracCalc[ii] = intp/100.0;
      }
    }

    // Convert the volume fraction into a number
    // Assume that the total volume is a large value
    // Formula is
    // n_i = vf_i * V / d_1^2  - 2D
    // n_i = vf_i * V / d_1^3  - 3D
    double totalVol = 1000.0*Math.pow(sizeInp[nofSizesInp-1],3);
    int nofSizesCalc = 0;
    int[] nofBalls2D = new int[NUM_SIZES_MAX];
    int[] nofBalls3D = new int[NUM_SIZES_MAX];
    double[] ballDia = new double[NUM_SIZES_MAX];
    if (nofSizesInp > 0) {
      for (int ii = 0; ii < NUM_SIZES_MAX-1; ++ii) {
        nofBalls2D[nofSizesCalc] = (int)
          Math.ceil(meanVolFracCalc[ii]*totalVol/Math.pow(meanSizeCalc[ii],2));
        nofBalls3D[nofSizesCalc] = (int)
          Math.ceil(meanVolFracCalc[ii]*totalVol/Math.pow(meanSizeCalc[ii],3));
        ballDia[nofSizesCalc] = meanSizeCalc[ii]; 
        ++nofSizesCalc;
      }
      d_partSizeDist.nofSizesCalc = nofSizesCalc;
    }

    double[] vol2D = new double[nofSizesCalc];
    double[] vol3D = new double[nofSizesCalc];
    int n2D, n3D;
    double dia;
    double totVol2D = 0.0; double totVol3D = 0.0;
    for (int ii = 0; ii < nofSizesCalc; ++ii) {
      n2D = nofBalls2D[ii];
      n3D = nofBalls3D[ii];
      dia = ballDia[ii];
      vol2D[ii] = dia*dia*n2D;
      vol3D[ii] = dia*dia*dia*n3D;
      totVol2D += vol2D[ii];
      totVol3D += vol3D[ii];
    }

    // Copy the rest of the data into d_PartSizeDist
    DecimalFormat df = new DecimalFormat("##0.0##E0");
    System.out.println("Composite");
    System.out.println("Size "+
                       "... Number (2D) .. Vol.Frac (2D)"+
                       "... Number (3D) .. Vol.Frac (3D)");
    int totBalls2D = 0;
    int totBalls3D = 0;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      d_partSizeDist.sizeCalc[ii] = ballDia[ii];
      d_partSizeDist.freq2DCalc[ii] = nofBalls2D[ii];
      d_partSizeDist.freq3DCalc[ii] = nofBalls3D[ii];
      d_partSizeDist.volFrac2DCalc[ii] = 100.0*vol2D[ii]/totVol2D;
      d_partSizeDist.volFrac3DCalc[ii] = 100.0*vol3D[ii]/totVol3D;
      totBalls2D += nofBalls2D[ii];
      totBalls3D += nofBalls3D[ii];
      System.out.println(df.format(ballDia[ii])+"    "+
                         nofBalls2D[ii]+"     "+ 
                         df.format(d_partSizeDist.volFrac2DCalc[ii])+"      "+
                         nofBalls3D[ii]+"     "+ 
                         df.format(d_partSizeDist.volFrac3DCalc[ii]));
    }
    System.out.println(" Total:  2D = "+totBalls2D+ " 3D = "+totBalls3D);

    // Update the histogram display
    d_parent.refreshDisplayPartDistFrame();
  }

  // Get the particle size distribution data
  public ParticleSize getParticleSizeDist() {
    return d_partSizeDist;
  }
}
