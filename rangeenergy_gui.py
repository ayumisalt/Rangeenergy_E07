# -*- coding: utf-8 -*-

# python3系必須
from PyQt5.QtWidgets import (QApplication, QWidget,
                             QGridLayout, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QMessageBox, 
                             QComboBox,QTextEdit, QCheckBox)
import subprocess
import os
from datetime import time, tzinfo, datetime, date
import rangeenergy as reen
import nuclide
import kinema
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import fsolve
import math

TEXT_RANGE = "Range [$#mu$ m]"
TEXT_RANGE_MGCM2 = "Range [mg/cm2]"
TEXT_BETAGAMMA = "$#beta#gamma$"
TEXT_MOMENTUM = "Momentum [MeV/c]"
TEXT_KE = "Kinetic energy [MeV]"
TEXT_EOU = "Energy [MeV/u]"
TEXT_DE = "Stopping power dE MeV cm$^2$/g"
TEXT_STRAGGLING_RATE_RANGE = "Range straggling rate [in range]"
TEXT_STRAGGLING_RATE_KE = "KE straggling rate [$#delta$KE/KE]"
TEXT_STRAGGLING_ABS_KE = "KE straggling $#delta$KE [MeV]"

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.Particle = QLineEdit()
        self.Particle.setFixedWidth(100)
        self.KineticEnergy = QLineEdit()
        self.KineticEnergy.setFixedWidth(100)
        self.Range = QLineEdit()
        self.Range.setFixedWidth(100)
        self.Momentum = QLineEdit()
        self.Momentum.setFixedWidth(100)
        self.Density = QLineEdit()
        self.Density.setFixedWidth(100)
        self.Mass = QLineEdit()
        self.Mass.setFixedWidth(100)
        self.RangeStraggling = QLineEdit()
        self.RangeStraggling.setFixedWidth(100)
        self.BetaGamma = QLineEdit()
        self.BetaGamma.setFixedWidth(100)

        self.GraphHorizontal = QComboBox(self)
        self.GraphHorizontal.addItem(TEXT_BETAGAMMA)
        self.GraphHorizontal.addItem(TEXT_RANGE)
        self.GraphHorizontal.addItem(TEXT_MOMENTUM)
        self.GraphHorizontal.addItem(TEXT_KE)
        self.GraphHorizontal.addItem(TEXT_EOU)
        self.GraphHorizontal.addItem(TEXT_DE)

        self.GraphVertical = QComboBox(self)
        self.GraphVertical.addItem(TEXT_DE)
        self.GraphVertical.addItem(TEXT_BETAGAMMA)
        self.GraphVertical.addItem(TEXT_RANGE)
        self.GraphVertical.addItem(TEXT_MOMENTUM)
        self.GraphVertical.addItem(TEXT_KE)
        self.GraphVertical.addItem(TEXT_STRAGGLING_RATE_RANGE)
        self.GraphVertical.addItem(TEXT_STRAGGLING_RATE_KE)
        self.GraphVertical.addItem(TEXT_STRAGGLING_ABS_KE)
        self.GraphVertical.addItem(TEXT_RANGE_MGCM2)

        self.Particle.setText("p")
        self.KineticEnergy.setText("20")
        self.Range.setText("2000")
        self.Momentum.setText("200")
        self.Density.setText("3.619")
        self.Mass.setText("938.272")

        self.calc_e2oButton = QPushButton("KE -> Range, Mom")
        self.calc_e2oButton.clicked.connect(self.process_calc_e2o)
        self.calc_r2oButton = QPushButton("Range -> KE, Mom")
        self.calc_r2oButton.clicked.connect(self.process_calc_r2o)
        self.calc_m2oButton = QPushButton("Mom -> Range, KE")
        self.calc_m2oButton.clicked.connect(self.process_calc_m2o)
        self.calc_er2dButton = QPushButton("KE, Range -> Density")
        self.calc_er2dButton.clicked.connect(self.process_calc_energyrange2density)
        self.calc_lifetimeButton = QPushButton("RangeHF, MomAtDecay -> LifeTime")
        self.calc_lifetimeButton.clicked.connect(self.process_calc_lifetime)
        self.calc_showdeButton = QPushButton("Show Graph")
        self.calc_showdeButton.clicked.connect(self.process_calc_de)

        self.resultList = QTextEdit()
        self.resultList.setMaximumHeight(100)
        self.resultList.setText("")

        self.logCheckBoxX = QCheckBox()
        self.logCheckBoxX.setText("Set log X")
        self.logCheckBoxX.setChecked(True)
        self.logCheckBoxY = QCheckBox()
        self.logCheckBoxY.setText("Set log Y")
        self.logCheckBoxY.setChecked(False)
        self.showPDGData = QCheckBox()
        self.showPDGData.setText("Show PDG")
        self.showPDGData.setChecked(False)
        self.XRange = QLineEdit()
        self.XRange.setFixedWidth(150)
        self.YRange = QLineEdit()
        self.YRange.setFixedWidth(150)

        lineLayout = QGridLayout()
        lineLayout.addWidget(QLabel("Particle [p, d, t, He4 etc]"), 0, 0)
        lineLayout.addWidget(self.Particle, 0, 1)
        lineLayout.addWidget(QLabel("KineticEnergy [MeV]"), 2, 0)
        lineLayout.addWidget(self.KineticEnergy, 2, 1)
        lineLayout.addWidget(QLabel("Range [um]"), 3, 0)
        lineLayout.addWidget(self.Range, 3, 1)
        lineLayout.addWidget(QLabel("Momentum [MeV/c]"), 4, 0)
        lineLayout.addWidget(self.Momentum, 4, 1)
        lineLayout.addWidget(QLabel("Mass [MeV/c2]"), 5, 0)
        lineLayout.addWidget(self.Mass, 5, 1)
        lineLayout.addWidget(QLabel("Density [g/cm3]"), 6, 0)
        lineLayout.addWidget(self.Density, 6, 1)
        lineLayout.addWidget(QLabel("Range straggling [um]"), 7, 0)
        lineLayout.addWidget(self.RangeStraggling, 7, 1)
        lineLayout.addWidget(QLabel("Beta gamma"), 8, 0)
        lineLayout.addWidget(self.BetaGamma, 8, 1)
        lineLayout.addWidget(self.resultList, 9, 0, 1, 2)

        axisLayout = QGridLayout()
        axisLayout.addWidget(self.logCheckBoxX, 0, 0)
        axisLayout.addWidget(self.logCheckBoxY, 1, 0)
        axisLayout.addWidget(self.showPDGData, 2, 0)
        axisLayout.addWidget(self.XRange, 0, 1)
        axisLayout.addWidget(self.YRange, 1, 1)
        axisLayout.addWidget(QLabel("X range"), 0, 2)
        axisLayout.addWidget(QLabel("Y range"), 1, 2)
        self.change_model = QPushButton("✓Mishina  ATIMA")
        self.change_model.clicked.connect(self.process_change_model)
        axisLayout.addWidget(self.change_model, 2, 1)

        buttonLayout = QGridLayout()
        buttonLayout.addWidget(self.calc_e2oButton, 0, 0)
        buttonLayout.addWidget(self.calc_r2oButton, 1, 0)
        buttonLayout.addWidget(self.calc_m2oButton, 2, 0)
        buttonLayout.addWidget(self.calc_er2dButton, 3, 0)
        buttonLayout.addWidget(self.calc_lifetimeButton, 4, 0)
        buttonLayout.addWidget(self.calc_showdeButton, 5, 0)
        buttonLayout.addWidget(self.GraphHorizontal, 6, 0)
        buttonLayout.addWidget(self.GraphVertical, 7, 0)
        buttonLayout.addLayout(axisLayout, 8, 0)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(lineLayout)
        mainLayout.addLayout(buttonLayout)


        self.setLayout(mainLayout)
        self.setWindowTitle("Range, KE, Mom calculation {0}".format(datetime.now().strftime("%c")))

    def process_change_model(self):
        if self.change_model.text()=="✓Mishina  ATIMA":
            reen.RANGEENERGY_MODEL = "ATIMA"
            self.change_model.setText("Mishina  ✓ATIMA")
        else:
            reen.RANGEENERGY_MODEL = "Mishina"
            self.change_model.setText("✓Mishina  ATIMA")

    def process_calc_massdiff(self, particles):
        if "-" in particles or "+" in particles:
            mass = 0
            for i, particle in enumerate(particles):
                if particle == "-" or particle == "+":continue
                nucleus = nuclide.search(particle)
                mass += nucleus['M'] * (1 if i == 0 or particles[i - 1] == "+" else -1)
            self.Mass.setText(f"{mass:.3f}")
            return True
        else: return False

    def process_calc_e2o(self):
        try:
            particles = self.Particle.text().split()
            if self.process_calc_massdiff(particles):return
            nucleus = nuclide.search(particles[0])
            mass = nucleus['M']
            self.Mass.setText(f"{mass:.3f}")

            ke = eval(self.KineticEnergy.text())
            mom = kinema.ke2mom(mass, ke)
            density = float(self.Density.text())

            if nucleus['Z'] != 0:
                range = reen.RangeFromKE(mass, ke, nucleus['Z'], density)
            else:
                range = -1

            self.Range.setText(f"{range:.3f}")
            self.Momentum.setText(f"{mom:.3f}")
            self.BetaGamma.setText(f"{mom/mass:.6f}")

            rstr = reen.RangeStragglingFromRange(mass, range, nucleus['Z'], density)
            self.RangeStraggling.setText(f"{rstr:.3f}")
        except:
            print(sys.exc_info())
            return

    def process_calc_r2o(self):
        try:
            particles = self.Particle.text().split()
            if self.process_calc_massdiff(particles):return
            nucleus = nuclide.search(particles[0])
            if nucleus['Z'] == 0:
                self.KineticEnergy.setText("Error")
                self.Momentum.setText("Error")
                return
                
            mass = nucleus['M']
            self.Mass.setText(f"{mass:.3f}")

            range = eval(self.Range.text())
            density = float(self.Density.text())

            ke = reen.KEfromRange(mass, range, nucleus['Z'], density)
            mom = kinema.ke2mom(mass, ke)

            self.KineticEnergy.setText(f"{ke:.3f}")
            self.Momentum.setText(f"{mom:.3f}")
            self.BetaGamma.setText(f"{mom/mass:.6f}")

            rstr = reen.RangeStragglingFromRange(mass, range, nucleus['Z'], density)
            self.RangeStraggling.setText(f"{rstr:.3f}")
        except:
            print(sys.exc_info())
            return

    def process_calc_m2o(self):
        try:
            particles = self.Particle.text().split()
            if self.process_calc_massdiff(particles):return
            nucleus = nuclide.search(particles[0])
            mass = nucleus['M']
            self.Mass.setText(f"{mass:.3f}")

            mom = eval(self.Momentum.text())
            ke = kinema.mom2ke(mass, mom)
            density = float(self.Density.text())

            if nucleus['Z'] != 0:
                range = reen.RangeFromKE(mass, ke, nucleus['Z'], density)
            else:
                range = -1

            self.BetaGamma.setText(f"{mom/mass:.6f}")
            self.KineticEnergy.setText(f"{ke:.3f}")
            self.Range.setText(f"{range:.3f}")

            rstr = reen.RangeStragglingFromRange(mass, range, nucleus['Z'], density)
            self.RangeStraggling.setText(f"{rstr:.3f}")
        except:
            print(sys.exc_info())
            return

    def process_calc_energyrange2density(self):
        try:
            particles = self.Particle.text().split()
            nucleus = nuclide.search(particles[0])
            mass = nucleus['M']
            self.Mass.setText(f"{mass:.3f}")

            range = eval(self.Range.text())
            ke = eval(self.KineticEnergy.text())
            mom = kinema.ke2mom(mass, ke)
            self.Momentum.setText(f"{mom:.3f}")
            self.BetaGamma.setText(f"{mom/mass:.6f}")

            density = reen.DensityFromKERange(mass, ke, range, nucleus['Z'])

            self.Density.setText(f"{density:.3f}")
        except:
            print(sys.exc_info())
            return

    def process_calc_lifetime(self):
        try:
            density = float(self.Density.text())
            range_HF = float(self.Range.text().split()[0])
            nucleus = nuclide.search(self.Particle.text().split()[0])
            if nucleus['Z'] == 0:
                self.resultList.setText("Particle has no charge.")
                return
            
            mass = nucleus['M']
            self.Mass.setText(f"{mass:.3f}")
            self.KineticEnergy.setText("")

            mom_invisible = float(self.Momentum.text())
            ke_invisible = kinema.mom2ke(mass, mom_invisible)
                
            range_invisible = reen.RangeFromKE(mass, ke_invisible, nucleus['Z'], density)
            beta_invisible = mom_invisible / math.sqrt(ke_invisible ** 2 + mass ** 2)

            total_range = range_HF + range_invisible
            ke_total = reen.KEfromRange(mass, total_range, nucleus['Z'], density)
            mom_total = kinema.ke2mom(mass, ke_total)
            beta_total = mom_total / math.sqrt(ke_total ** 2 + mass ** 2)

            Nsegment = 100
            range_seg = range_HF / Nsegment
            life_expected = 0
            light_speed = 299_792_458_000_000 # um
            beta_list = []

            for i in range(Nsegment):
                range_step = total_range - i * range_seg
                ke_seg = reen.KEfromRange(mass, range_step, nucleus['Z'], density)
                mom_seg = kinema.ke2mom(mass, ke_seg)
                beta_seg = mom_seg / math.sqrt(ke_seg ** 2 + mass ** 2)
                beta_list.append(beta_seg)

            for i in range(Nsegment - 1):
                beta_seg = (beta_list[i] + beta_list[i + 1]) / 2
                dt = range_seg / (beta_seg * light_speed)
                life_expected += dt
                print(beta_seg,dt)

            result = "Invisible range: {0:.3f} [um]\n".format(range_invisible)
            result += "Total range   : {0:.3f} [um]\n".format(total_range)
            result += "Life time     : {0:.3f} [ps]\n".format(life_expected * 1e12)
            self.resultList.setText(result)

        except:
            print(sys.exc_info())
            return

    def process_calc_de(self):
        try:
            matplotlib.rcParams.update({'font.size': 16})
            plt.rcParams["figure.figsize"] = [12, 9]

            particles = self.Particle.text().split()

            for particle in particles:
                nucleus = nuclide.search(particle)
                mass = nucleus['M']
                density = float(self.Density.text())

                x = []
                y = []
                xlabel = self.GraphHorizontal.currentText()
                ylabel = self.GraphVertical.currentText()
                text = "#{} {}\n".format(xlabel.replace('$','').replace(' ',''), 
                                         ylabel.replace('$','').replace(' ',''))
                for i in range(-30, 101):

                    try:
                        rangec = 10 ** (i * 0.05)
                        kec = reen.KEfromRange(mass, rangec, nucleus['Z'], density)
                        mom = kinema.ke2mom(mass, kec)

                        if not ((xlabel == TEXT_BETAGAMMA or xlabel == TEXT_RANGE or xlabel == TEXT_MOMENTUM or xlabel == TEXT_KE or xlabel == TEXT_EOU) 
                                and (ylabel == TEXT_BETAGAMMA or ylabel == TEXT_RANGE or ylabel == TEXT_RANGE_MGCM2 or ylabel == TEXT_MOMENTUM or ylabel == TEXT_KE)):
                            range0 = 10 ** (i * 0.05 - 0.025)
                            range1 = 10 ** (i * 0.05 + 0.025)
                            ke0 = reen.KEfromRange(mass, range0, nucleus['Z'], density)
                            ke1 = reen.KEfromRange(mass, range1, nucleus['Z'], density)
                            range_straggling = reen.RangeStragglingFromRange(mass, rangec, nucleus['Z'], density)
                            range_straggling_range = range_straggling / rangec
                            kep = reen.KEfromRange(mass, rangec + range_straggling, nucleus['Z'], density)
                            kem = reen.KEfromRange(mass, rangec - range_straggling, nucleus['Z'], density)
                            range_straggling_ke = (kep - kem) * 0.5 / kec
                            dE = (ke1 - ke0) / (range1 - range0) * 10000 / density

                        if xlabel == TEXT_BETAGAMMA:
                            x.append(mom / mass)
                        elif xlabel == TEXT_RANGE :
                            x.append(rangec)
                        elif xlabel == TEXT_MOMENTUM :
                            x.append(mom)
                        elif xlabel == TEXT_KE:
                            x.append(kec)
                        elif xlabel == TEXT_EOU:
                            x.append(kec*nucleus["M"]/931.4941)
                        elif xlabel == TEXT_DE:
                            x.append(dE)

                        if ylabel == TEXT_BETAGAMMA:
                            y.append(mom / mass)
                        elif ylabel == TEXT_RANGE :
                            y.append(rangec)
                        elif ylabel == TEXT_RANGE_MGCM2:
                            y.append(rangec/10000*density*1000)
                        elif ylabel == TEXT_MOMENTUM :
                            y.append(mom)
                        elif ylabel == TEXT_KE:
                            y.append(kec)
                        elif ylabel == TEXT_DE:
                            y.append(dE)
                        elif ylabel == TEXT_STRAGGLING_RATE_RANGE:
                            y.append(range_straggling_range)
                        elif ylabel == TEXT_STRAGGLING_RATE_KE:
                            y.append(range_straggling_ke)
                        elif ylabel == TEXT_STRAGGLING_ABS_KE:
                            y.append((kep - kem) * 0.5)
                        
                        text += "{:.6f} {:.6f}\n".format(x[-1], y[-1])
                    except:pass

                plt.plot(x, y, label=particle)

            if self.showPDGData.isChecked():
                nucleus = nuclide.search("mu-")
                mass = nucleus["M"]
                data = np.loadtxt("stopping_power_muon.txt")
                x = []
                y = []
                if xlabel == TEXT_BETAGAMMA:
                    x = data.T[1] / mass
                elif xlabel == TEXT_MOMENTUM :
                    x = data.T[1]
                elif xlabel == TEXT_KE:
                    x = data.T[0]
                elif xlabel == TEXT_DE:
                    x = data.T[7]
                if ylabel == TEXT_BETAGAMMA:
                    y = data.T[1] / mass
                elif ylabel == TEXT_MOMENTUM :
                    y = data.T[1]
                elif ylabel == TEXT_KE:
                    y = data.T[0]
                elif ylabel == TEXT_DE:
                    y = data.T[7]
                if len(x) > 0 and len(y) > 0:
                    plt.plot(x, y, label="Muon PDG")

                nucleus = nuclide.search("e-")
                mass = nucleus["M"]
                data = np.loadtxt("stopping_power_electron.txt")
                x = []
                y = []
                if xlabel == TEXT_BETAGAMMA:
                    x = kinema.ke2mom(mass, data.T[0]) / mass
                elif xlabel == TEXT_MOMENTUM :
                    x = kinema.ke2mom(mass, data.T[0])
                elif xlabel == TEXT_KE:
                    x = data.T[0]
                elif xlabel == TEXT_DE:
                    x = data.T[3]
                if ylabel == TEXT_BETAGAMMA:
                    y = kinema.ke2mom(mass, data.T[0]) / mass
                elif ylabel == TEXT_MOMENTUM :
                    y = kinema.ke2mom(mass, data.T[0])
                elif ylabel == TEXT_KE:
                    y = data.T[0]
                elif ylabel == TEXT_DE:
                    y = data.T[3]

                if len(x) > 0 and len(y) > 0:
                    plt.plot(x, y, label="Electron PDG")


            plt.grid(which="both")
            if self.logCheckBoxX.isChecked():
                plt.xscale("log")
            if self.logCheckBoxY.isChecked():
                plt.yscale("log")
            if ylabel == TEXT_STRAGGLING_RATE_RANGE:
                plt.ylim(0.0,0.08)
            if ylabel == TEXT_STRAGGLING_RATE_KE:
                plt.ylim(0.0,0.08)

            x_range = self.XRange.text().split()
            y_range = self.YRange.text().split()
            if len(x_range):
                plt.xlim(float(x_range[0]),float(x_range[1]))
            if len(y_range):
                plt.ylim(float(y_range[0]),float(y_range[1]))

            #if len(particles) > 1:
            plt.legend()
            #plt.title("Graph for {}".format(particles))
            plt.xlabel(xlabel.replace('#','\\'))
            plt.ylabel(ylabel.replace('#','\\'))
            plt.show()
            self.resultList.setText(text)

        except:
            print(sys.exc_info())
            return


if __name__ == '__main__':
    reen.RANGEENERGY_MODEL = "Mishina"

    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()

    main_window.show()
    sys.exit(app.exec_())
