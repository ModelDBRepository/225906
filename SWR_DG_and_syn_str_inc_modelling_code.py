# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:07:00 2016

@author: Conor D. Cox
Notes:
    this requires brian2 and python 2.7, though it might run on python 3 the print functions are all wrong.
    WARNING WARNING WARNING
    This requires a number of GB of ram equal to the number of seconds you are simulating. If your computer doesn't have that
    it will crash this can be ameliorated by changing the recording function down around  428ish. You can turn
    those off and it will require less RAM.
    The output of this, other than dozens of figures is the output array. You can save it as you see fit
    This code representes figures 5 and 6. To swap between panels 5 and 6, turn
    net_loop on and off and increase number of runs. to try different parts of panel 6
    uncomment the line at 441 and comment out the line that says DG freq then switch the loop to run a rrange of dg_freq.
    email me for any problems:
    cdcox1@gmail.com
"""

# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from brian2 import *
import copy

def make_targetting_array():
    targetting_count_array=[]
    j=1
    for i in range(1100):
        if i%11!=0:
            targetting_count_array.append(j)
            j+=1
        else:
            targetting_count_array.append(0)
    targetting_count_array=np.array(targetting_count_array)
    return targetting_count_array
  

def randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons):
    check=np.random.normal(scale=1000)
    target=np.round(check/10)
    temp_tar=real_i+target
    if ((i+target)<0) or ((temp_tar)>=len(targetting_count_array)):
        checker=999999999
        return checker,check,12345678
    checker=targetting_count_array[temp_tar]
    if checker>number_of_neurons:
        checker=999999999
        return checker,check,12345678
    return checker,check,temp_tar
    
def randomize_and_edge_checkint(real_i,i,targetting_count_array):
    check=np.random.normal(scale=100)
    target=np.round(check/10)
    temp_tar=real_i+target
    if ((real_i+target)<0) or ((temp_tar)>=len(targetting_count_array)):
        checker=999999999
        return checker,check,12345678
    checker=targetting_count_array[temp_tar]
    if checker>number_of_ca1_neurons:
        checker=999999999
        return checker,check,12345678
    return checker,check,temp_tar

  
def excite_to_excite_wire_it_up(number_of_neurons,exkij,exkijsd,inkj,inkijsd):
    connectivity=[]
    connectivityin=[]
    delayex=[]
    delayin=[]
    targetting_count_array=make_targetting_array()
    for i in range(1,number_of_neurons+1):
        in_count=0
        numb_in_connect=np.int(np.round(np.random.normal(loc=inkj,scale=inkijsd)))
        if exkij!=0:
            numb_connect=np.int(np.round(np.random.normal(loc=exkij,scale=exkijsd)))
        else:
            numb_connect=0
        real_i=np.where(targetting_count_array==i)[0][0]
        for j in range(1,numb_connect):  
            checker,check,temp_tar=randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons)
            if checker==999999999:
                continue
            while temp_tar%11==0 or checker==i:
                if in_count<numb_in_connect and checker!=i and temp_tar!=0:
                    connectivityin.append([i,np.int(temp_tar/11)])
                    delayin.append(np.abs(np.round(check*um/speed,decimals=4)))
                    in_count+=1
                checker,check,temp_tar=randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons)
                if checker==999999999:
                    continue
            if checker==999999999:
                continue
            connectivity.append([i,checker])
            delayex.append(np.abs(np.round(check*um/speed,decimals=4)))
        while in_count<numb_in_connect:
            checker,check,temp_tar=randomize_and_edge_check(real_i,i,targetting_count_array,number_of_neurons)
            if checker==999999999:
                continue
            if temp_tar%11==0 and temp_tar!=0:
                connectivityin.append([i,np.int(temp_tar/11)])
                delayin.append(np.abs(np.round(check*um/speed,decimals=4)))
                in_count+=1
    connectivity=np.array(connectivity)
    connectivityin=np.array(connectivityin)
    return connectivity,connectivityin,delayex,delayin
    
def ca1in_wire_it_up(number_of_ca1_neurons,exkij,exkijsd,inkj,inkijsd):
    speed2=100*um/ms
    connectivity=[]
    connectivityin=[]
    delayex=[]
    delayin=[]
    targetting_count_array=make_targetting_array()
    for i in range(1,int(number_of_ca1_neurons/10)+1):
        in_count=0
        numb_in_in_connect=np.int(np.round(np.random.normal(loc=inkj,scale=inkijsd)))
        numb_2_ex_connect=np.int(np.round(np.random.normal(loc=exkij,scale=exkijsd)))
        real_i=np.where(targetting_count_array==i)[0][0]
        for j in range(1,numb_2_ex_connect):  
            checker,check,temp_tar=randomize_and_edge_checkint(i*11,i,targetting_count_array)
            if checker==999999999:
                continue
            while temp_tar%11==0 or checker==i:
                if in_count<numb_in_in_connect and checker!=i and temp_tar!=0:
                    connectivityin.append([i,np.int(temp_tar/11)])
                    delayin.append(np.abs(np.round(check*um/speed2,decimals=4)))
                    in_count+=1
                checker,check,temp_tar=randomize_and_edge_checkint(i*11,i,targetting_count_array)
                if checker==999999999:
                    continue
            if checker==999999999:
                continue
            connectivity.append([i,checker])
            delayex.append(np.abs(np.round(check*um/speed2,decimals=4)))
        while in_count<numb_in_in_connect:
            checker,check,temp_tar=randomize_and_edge_checkint(i*11,i,targetting_count_array)
            if checker==999999999:
                continue
            if temp_tar%11==0 and temp_tar!=0:
                connectivityin.append([i,np.int(temp_tar/11)])
                delayin.append(np.abs(np.round(check*um/speed2,decimals=4)))
                in_count+=1
    connectivity=np.array(connectivity)
    connectivityin=np.array(connectivityin)
    return connectivity,connectivityin,delayex,delayin

def in2ex_wire_it_up(number_of_ca3_neurons,ca3_inpykij,ca3_inpykij_sd):
    #wires up CA3 in2ex
    targetting_count_array=make_targetting_array()
    connectivity=[]
    delay=[]
    for i in range(1,101):
        numb_connect=np.int(np.round(np.random.normal(loc=ca3_inpykij,scale=ca3_inpykij_sd)))
        cell_pos=i*11
        for j in range(1,numb_connect):  
            step=np.random.uniform(-1.5*100,1.5*100)
            cell_step=step/10
            target_cell=np.int(np.round(cell_pos+cell_step))

            if target_cell<0 or (target_cell>=len(targetting_count_array)):
                continue
            real_target=targetting_count_array[target_cell]
            while real_target==0:
                step=np.random.uniform(-1.5*100,1.5*100)
                cell_step=step/10
                target_cell=np.int(np.round(cell_pos+cell_step))

                if target_cell<0 or (target_cell>=len(targetting_count_array)):
                    continue
                real_target=targetting_count_array[target_cell] 
            connectivity.append([i,real_target])
            delay.append((np.abs(np.round(step*um/speed,decimals=4))))
    connectivity=np.array(connectivity)
    return connectivity,delay
    
def randomize_and_edge_checkSC(real_i,i,targetting_count_array,number_of_neurons):
    check=np.random.normal(scale=1200)
    target=np.round(check/10)
    temp_tar=real_i+target
    if ((i+target)<0) or ((temp_tar)>=len(targetting_count_array)):
        checker=999999999
        return checker,check,12345678
    checker=targetting_count_array[temp_tar]
    if checker>number_of_neurons:
        checker=999999999
        return checker,check,12345678
    return checker,check,temp_tar
    
def schafer_collaterals(number_of_neurons,exkij,exkijsd,inkj,inkijsd):
    connectivity=[]
    connectivityin=[]
    delayex=[]
    delayin=[]
    targetting_count_array=make_targetting_array()
    for i in range(1,number_of_neurons+1):
        if exkij!=0:
            numb_connect=np.int(np.round(np.random.normal(loc=exkij,scale=exkijsd)))
        else:
            numb_connect=0
        real_i=np.where(targetting_count_array==i)[0][0]
        for j in range(1,numb_connect):  
            checker,check,temp_tar=randomize_and_edge_checkSC(real_i,i,targetting_count_array,number_of_neurons)
            if checker==999999999:
                continue
            if temp_tar%11==0 and temp_tar!=0:
                dupe_rate=13
                for k in range(dupe_rate):
                    connectivityin.append([i,np.int(temp_tar/11)])
                    delayin.append(np.abs(np.round(check*um/speed,decimals=4)))
            else:
                dupe_rate=np.abs(np.round(13+np.random.normal(scale=13)))
                for k in range(int(dupe_rate)):
                    connectivity.append([i,checker])
                    delayex.append(np.abs(np.round(check*um/speed,decimals=4)))
    connectivity=np.array(connectivity)
    connectivityin=np.array(connectivityin)
    delayex=delayex+1000*um/speed
    delayin=delayin+1000*um/speed
    return connectivity,connectivityin,delayex,delayin    
    
def generate_dentate_gyrus(number_of_cells,frequency_of_spiking,length_of_simulation):
    frequency=frequency_of_spiking/1000
    firing_array=np.random.rand(number_of_cells,np.int(length_of_simulation/ms))
    firing_array=firing_array<frequency
    indicies,times=np.where(firing_array)
    dentate = SpikeGeneratorGroup(number_of_cells, indicies, times*ms)
    return dentate

def calc_line_crossings(net):
    count_list=[]
    for i in range(1001):
        a=net['statemonitor'].Issyn[i][:-1]/nA-net['statemonitor'].Issyn[i][1:]/nA
        zero_crossings = np.where(np.diff(np.signbit(a)))[0]
        count_list.append(len(zero_crossings))
    return np.mean(count_list)
    
def threshold_based_counter(net):
    counts_list=[]
    for i in range(1001):
        set_list=net['statemonitor'].Issyn[i]/nA<-5
        reset_list=net['statemonitor'].Issyn[i]/nA>-2
        counterr=0
        flip=0
        for sn in range(len(set_list)):
            if set_list[sn] and flip==0:
                flip=1
                counterr+=1
            if reset_list[sn] and flip==1:
                flip=0
        counts_list.append(counterr)
    return(np.mean(counts_list))
        
def main(output_array,innnnn,net_loop):    

    Pinksy_rinzel_eqs='''
    
    dVs/dt=(-gLs*(Vs-VL)-gNa*(Minfs**2)*hs*(Vs-VNa)-gKdr*ns*(Vs-VK)+(gc/pp)*(Vd-Vs)+(Ip0-Iinssyn)/pp)/Cm : volt
    dVd/dt=(-gLd*(Vd-VL)-ICad-gKahp*qd*(Vd-VK)-gKC*cd*chid*(Vd-VK)+(gc*(Vs-Vd))/(1.0-pp)-Isyn/(1.0-pp))/Cm : volt
    dCad/dt=  -0.13*ICad/uamp/ms*scaler-0.075*Cad/ms : 1
    dhs/dt=  alphahs-(alphahs+betahs)*hs : 1
    dns/dt=  alphans-(alphans+betans)*ns : 1
    dsd/dt=  alphasd-(alphasd+betasd)*sd : 1
    dcd/dt=  alphacd-(alphacd+betacd)*cd : 1
    dqd/dt=  alphaqd-(alphaqd+betaqd)*qd : 1
    ICad=     gCa*sd*sd*(Vd-VCa) : amp
    alphams=  0.32*(-46.9-Vs/mV)/(exp((-46.9-Vs/mV)/4.0)-1.000001)/ms : Hz
    betams=   0.28*(Vs/mV+19.9)/(exp((Vs/mV+19.9)/5.0)-1.000001)/ms : Hz
    Minfs=    alphams/(alphams+betams) : 1
    alphans=  0.016*(-24.9-Vs/mV)/(exp((-24.9-Vs/mV)/5.0)-1.000001)/ms : Hz
    betans=   0.25*exp(-1.0-0.025*Vs/mV)/ms : Hz
    alphahs=  0.128*exp((-43.0-Vs/mV)/18.0)/ms : Hz
    betahs=   4.0/(1.00001+exp((-20.0-Vs/mV)/5.0))/ms : Hz
    alphasd=  1.6/(1.0001+exp(-0.072*(Vd/mV-5.0)))/ms : Hz
    betasd=   0.02*(Vd/mV+8.9)/(exp((Vd/mV+8.9)/5.0)-1.0)/ms : Hz
    alphacd=((Vd/mV<=-10)*exp((Vd/mV+50.0)/11-(Vd/mV+53.5)/27)/18.975+(Vd/mV>-10)*2.0*exp((-53.5-Vd/mV)/27.0))/ms  : Hz
    betacd=   ((Vd/mV<=-10)*(2.0*exp((-53.5-Vd/mV)/27.0)-alphacd*ms)+(Vd/mV>-10)*0)/ms : Hz
    alphaqd=  clip(0.00002*Cad,0,0.01)/ms : Hz
    betaqd=   0.001/ms : Hz
    chid=     clip(Cad/250.0,0,1.0) : 1
    INMDA=gNMDA*Si*(1+0.28*exp(-0.062*(Vs/mV)))**(-1)*(Vd-Vsyn) : amp
    Hxs=0<=((Vs/mV+50)*1) : 1
    Hxw=0<=((Vs/mV+40)*1) : 1
    dSi/dt=(Hxs-Si/150)/second : 1
    Isyn=Issyn+INMDA+Iapp : amp
    Issyn=gbarsyn*clip(ssyn,0,7000)*(Vd-Vsyn): amp
    dssyn/dt=-ssyn/tausyn: 1
    dinssyn/dt=-inssyn/tauisyn: 1
    Iinssyn=gbarsyn*clip(inssyn,0,7000)*(Vs-VK): amp
    dIapp/dt=0*amp/ms : amp
    counter:1
    '''
    #
    
    CA1_pyramidal = NeuronGroup(number_of_ca1_neurons+1, Pinksy_rinzel_eqs,threshold='Vs>-20*mV',refractory='Vs>-60*mV',method='euler')
    CA1_pyramidal.Vs='(randn()*60*.05-60)*mV'
    CA1_pyramidal.Vd='(randn()*65*.05-65)*mV'
    CA1_pyramidal.hs=.999
    CA1_pyramidal.ns=.0001
    CA1_pyramidal.sd=.009
    CA1_pyramidal.cd=.007
    CA1_pyramidal.qd=.01
    CA1_pyramidal.Cad=.20
    CA1_pyramidal.Si=0
    CA1_pyramidal.ssyn=0
    CA1_pyramidal.inssyn=0
    


    
    buzsaki_eqs = '''
    dv/dt = (-gNai*m**3*h*(v-VNa)-gKi*n**4*(v-VK)-gL*(v-VL2)-Issyn-Iinssyn-Iapp2)/Cmi : volt
    m = alpha_m/(alpha_m+beta_m) : 1
    alpha_m = -0.1/mV*(v+35*mV)/(exp(-0.1/mV*(v+35*mV))-1)/ms : Hz
    beta_m = 4*exp(-(v+60*mV)/(18*mV))/ms : Hz
    dh/dt = 5*(alpha_h*(1-h)-beta_h*h) : 1
    alpha_h = 0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
    beta_h = 1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
    dn/dt = 5*(alpha_n*(1-n)-beta_n*n) : 1
    alpha_n = -0.01/mV*(v+34*mV)/(exp(-0.1/mV*(v+34*mV))-1)/ms : Hz
    beta_n = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
    Issyn=gbarsyn*(clip(ssyn,0,7000))*(v-Vsyn): amp
    dssyn/dt=-ssyn/tausyn: 1
    dinssyn/dt=-inssyn/taguisyn: 1
    Iinssyn=gbarsyn*clip(inssyn,0,7000)*(v-VK):amp
    
    '''
    
    Ca1_interneuron=NeuronGroup(np.round(number_of_ca1_neurons/10+1),buzsaki_eqs,threshold='v>-20*mV',refractory='v>-60*mV',method='euler')
    Ca1_interneuron.v='(randn()*60*.05-60)*mV'
    Ca1_interneuron.h=.999
    Ca1_interneuron.n=.0001
    Ca1_interneuron.ssyn=0
    
    ca1_pypy_kij=0
    ca1_pypy_kij_sd=ca1_pypy_kij/20
    ca1_pyin_kij=20
    ca1_pyin_kij_sd=ca1_pyin_kij/20
    connectivity2ex,connectivity2in,delayex,delayin=excite_to_excite_wire_it_up(number_of_ca1_neurons,ca1_pypy_kij,ca1_pypy_kij_sd,ca1_pyin_kij,ca1_pyin_kij_sd)
    ca1_inpykij=400
    ca1_inpykij_sd=ca1_inpykij/20
    ca1in2inkij=100
    ca1in2iniij_sd=ca1in2inkij/20
    connectivityin2ex,connectivityin2in,delayinex,delayinin=ca1in_wire_it_up(number_of_ca1_neurons,ca1_inpykij,ca1_inpykij_sd,ca1in2inkij,ca1in2iniij_sd)
    
    Ca1_in_in=Synapses(Ca1_interneuron,Ca1_interneuron,pre='inssyn+=5')
    Ca1_in_in.connect(i=list(connectivityin2in[0:,0]),j=list(connectivityin2in[0:,1]))
    Ca1_in_in.delay=delayinin[0:]
    
    Ca1_py_in=Synapses(CA1_pyramidal,Ca1_interneuron,pre='ssyn+=2.5')
    
    Ca1_py_in.connect(i=list(connectivity2in[0:,0]),j=list(connectivity2in[0:,1]))
    Ca1_py_in.delay=delayin[0:]
    
    Ca1_in_py=Synapses(Ca1_interneuron,CA1_pyramidal,pre='inssyn+=15')
    Ca1_in_py.connect(i=list(connectivityin2ex[0:,0]),j=list(connectivityin2ex[0:,1]))
    Ca1_in_py.delay=delayinex[0:]
    print 'ca1'
    
    number_of_ca3_neurons=1000

    
    CA3_pyramidal = NeuronGroup(number_of_ca3_neurons+1, Pinksy_rinzel_eqs,threshold='Vs>-20*mV',refractory='Vs>-60*mV',method='euler')
    CA3_pyramidal.Vs='(randn()*60*.05-60)*mV'
    CA3_pyramidal.Vd='(randn()*65*.05-65)*mV'
    CA3_pyramidal.hs=.999
    CA3_pyramidal.ns=.0001
    CA3_pyramidal.sd=.009
    CA3_pyramidal.cd=.007
    CA3_pyramidal.qd=.01
    CA3_pyramidal.Cad=.20
    CA3_pyramidal.Si=0
    CA3_pyramidal.ssyn=0
    CA3_pyramidal.inssyn=0
    CA3_pyramidal.Iapp=0*nA
    CA3_pyramidal.counter=0
    #Iapp = 2*uA*((interneuron_size)/(10**8*um**2))
    
    Ca3_interneuron=NeuronGroup(np.round(number_of_ca3_neurons/10+1),buzsaki_eqs,threshold='v>-20*mV',refractory='v>-60*mV',method='euler')
    Ca3_interneuron.v=-60*mV
    Ca3_interneuron.h=.999
    Ca3_interneuron.n=.0001
    Ca3_interneuron.ssyn=0
    
    ca3_pypy_kij=55
    ca3_pypy_kij_sd=ca3_pypy_kij/20
    ca3_pyin_kij=5
    ca3_pyin_kij_sd=ca3_pyin_kij/20
    connectivity2ex,connectivity2in,delayex,delayin=excite_to_excite_wire_it_up(number_of_ca3_neurons,ca3_pypy_kij,ca3_pypy_kij_sd,ca3_pyin_kij,ca3_pyin_kij_sd)
    ca3_inpykij=68
    ca3_inpykij_sd=ca3_inpykij/20
    connectivityin2ex,delayinin=in2ex_wire_it_up(number_of_ca3_neurons,ca3_inpykij,ca3_inpykij_sd)
    

    Ca3_py_in=Synapses(CA3_pyramidal,Ca3_interneuron,pre='ssyn+=3')
    
    Ca3_py_in.connect(i=list(connectivity2in[0:,0]),j=list(connectivity2in[0:,1]))
    Ca3_py_in.delay=delayin[0:]
    print 'ca3'
    
    Ca3_in_py=Synapses(Ca3_interneuron,CA3_pyramidal,pre='inssyn+=50')
    Ca3_in_py.connect(i=list(connectivityin2ex[0:,0]),j=list(connectivityin2ex[0:,1]))
    
    exkijsc=130
    exkijsdsc=exkijsc/20
    scconnectivityex,scconnectivityin,scdelayex,scdelayin=schafer_collaterals(number_of_ca1_neurons,exkijsc,exkijsdsc,0,0)
    SC_py_py=Synapses(CA3_pyramidal,CA1_pyramidal,pre='ssyn+=15')
    SC_py_py.connect(i=list(scconnectivityex[0:,0]),j=list(scconnectivityex[0:,1]))
    SC_py_py.delay=scdelayex[0:]
    SC_py_in=Synapses(CA3_pyramidal,Ca1_interneuron,pre='ssyn+=15')
    SC_py_in.connect(i=list(scconnectivityin[0:,0]),j=list(scconnectivityin[0:,1]))
    SC_py_in.delay=scdelayin[0:]
    
    simulation_len=10000*ms
    net=Network(Ca3_interneuron,CA1_pyramidal,Ca1_interneuron,CA3_pyramidal,Ca1_py_in,Ca1_in_in,Ca1_in_py,Ca3_py_in,Ca3_in_py,SC_py_py,SC_py_in)
    names={'neurongroup':'CA3_pyramidal','neurongroup_1':'Ca1_interneuron','neurongroup_2':'CA1_pyramidal',
           'neurongroup_3':'Ca3_interneuron'}    
    net.add(SpikeMonitor(CA1_pyramidal))
    net.add(SpikeMonitor(CA3_pyramidal))
    net.add(SpikeMonitor(Ca1_interneuron))
    net.add(SpikeMonitor(Ca3_interneuron))
    net.add(StateMonitor(CA1_pyramidal, True ,record=True,dt=.5*ms))
    net.add(StateMonitor(Ca1_interneuron,True,record=True,dt=.5*ms))     
    net.add(StateMonitor(CA3_pyramidal, True ,record=True,dt=.5*ms))
    net.add(StateMonitor(Ca3_interneuron,True,record=True,dt=.5*ms))
    out_spikes_ca1=[]
    out_spikes_ca3=[]
    out_counts=[]
    out_deriv=[]
    thresh_counts=[]
    rcParams['figure.figsize'] = 5, 5
    #for strength_of_input in np.arange(1.2,1.25,.1):
    #for strength_of_input in [2.5,3,3.5]:
    
    for strength_of_input in arange(.5,4.5,.5):
        #strength_of_input=1.5
        dg_cells=1000
        dg_freq=1
        dg_connectivity=0.001
        Ca3str=str(np.round(strength_of_input*15,1))
        dgstr=str(np.round(strength_of_input*50,1))
        Ca3_py_py=Synapses(CA3_pyramidal,CA3_pyramidal,pre='ssyn+='+Ca3str+'''
                                                            counter+=1''')
        Ca3_py_py.connect(i=list(connectivity2ex[0:,0]),j=list(connectivity2ex[0:,1]))
        Ca3_py_py.delay=delayex[0:]
        #dg_cells=1000
        #dg_freq=4
        #dg_connectivity=0.0001
        #
        net.store()
        dentate_gyrus = PoissonGroup(dg_cells, dg_freq*Hz)
        MF_synapses=Synapses(dentate_gyrus,CA3_pyramidal,on_pre='ssyn+='+dgstr)
        MF_synapses.connect(p=dg_connectivity)
        print 'dis'
        net.add(dentate_gyrus)
        net.add(Ca3_py_py)
        net.add(MF_synapses)
        net.store()
        print len(net.objects)
        net.run(simulation_len, report='text')
        plt.cla()
        plt.clf()
        
        plot(net['spikemonitor'].t/ms,net['spikemonitor'].i,'.k')
        plt.axis((0,simulation_len/ms,0,1000))
        plt.axis('off')
        show()
        plt.savefig('aalowstartdgcells'+str(dg_cells)+'dgfreq'+str(dg_freq)+'str'+str(strength_of_input)+'connect'+str(dg_connectivity)+str(innnnn)+'.png',dpi=300)
        plt.cla()
        plt.clf()
        plot(net['spikemonitor_1'].t/ms,net['spikemonitor_1'].i,'.k')
        plt.axis((0,simulation_len/ms,0,1000))
        plt.axis('off')
        show()
        plt.savefig('aaCA3_dgcells'+str(dg_cells)+'dgfreq'+str(dg_freq)+'str'+str(strength_of_input)+'connect'+str(dg_connectivity)+str(innnnn)+'.png',dpi=300)
        plt.cla()
        plt.clf()
        
        plt.plot(net['statemonitor'][0].t,(np.sum(net['statemonitor'].Vd[500:550],0)+np.sum(net['statemonitor_1'].v[50:55],0))/55/mV)
        plt.savefig('aalowstartrecordingdgcells'+str(dg_cells)+'dgfreq'+str(dg_freq)+'str'+str(strength_of_input)+'connect'+str(dg_connectivity)+str(innnnn)+'.eps')
        plt.cla()
        plt.clf()
        plt.plot(net['statemonitor'][0].t,net['statemonitor'].Issyn[50]/nA)
        plt.savefig('aaIsynCa3tartrecordingdgcells'+str(dg_cells)+'dgfreq'+str(dg_freq)+'str'+str(strength_of_input)+'connect'+str(dg_connectivity)+str(innnnn)+'.eps')
        plt.axis((0,simulation_len/ms,-80,40))
        plt.cla()
        plt.clf()
        spikes_ca1=np.where(np.histogram(net['spikemonitor'].t,bins=200)[0]>500)[0]
        filt_spikes=[]
        for sn,spike in enumerate(spikes_ca1):
            if spikes_ca1[sn]-spikes_ca1[sn-1]<3:
                continue
            else:
                filt_spikes.append(spike)
        spikes_ca3=np.where(np.histogram(net['spikemonitor_1'].t,bins=200)[0]>100)[0]
        filt_spikes_ca3=[]
        for sn,spike in enumerate(spikes_ca3):
            if spikes_ca3[sn]-spikes_ca3[sn-1]<3:
                continue
            else:
                filt_spikes_ca3.append(spike)
        out_spikes_ca1.append(len(filt_spikes))
        out_spikes_ca3.append(len(filt_spikes_ca3))
        count_out=np.mean([x[-1] for x in net['statemonitor_2'].counter])
        derviative_out=calc_line_crossings(net)
        out_counts.append(count_out)
        out_deriv.append(derviative_out)
        thresh_counts.append(threshold_based_counter(net))
        if net_loop==0:
            return output_array,net
        net.restore()
        net.remove(dentate_gyrus)
        net.remove(MF_synapses)
        net.remove(Ca3_py_py)
    output_array.append([out_spikes_ca1,out_spikes_ca3,out_counts,out_deriv,thresh_counts])
    return output_array,0
speed=500*um/ms
number_of_ca1_neurons=1000
kij={'ca3pypy':55,'ca3pyin':5}
defaultclock.dt = 0.05*ms
Isyn=0*uamp
areapyr=50000*um**2
test=areapyr
Ip0=-0.3*nA
gLs=0.1  *msiemens*cm**-2*test
gLd=0.1  *msiemens*cm**-2*test
gNa=30  *msiemens*cm**-2*test
gKdr=15  *msiemens*cm**-2*test
gCa=7  *msiemens*cm**-2*test
gKahp=0.8*msiemens*cm**-2*test  
gKC=15  *msiemens*cm**-2*test
VNa=60*mV
VCa=80*mV
VK=-75*mV
VL=-60*mV
Vsyn=0*mV 
vtsyn=0*mV 
gc=1.75*usiemens
pp=0.5
Cm=3  *uF*cm**-2*test
gNMDA=0*msiemens*cm**-2*test
gampa=0*msiemens*cm**-2*test
gbarsyn=1*nS
tausyn= 2*ms
tauisyn=7*ms
scaler=(10**8*um**2/(test))
#interneuron stuff
interneuron_size=20000*um**2
Cmi = 1*uF/cm**2*interneuron_size
gL = 0.1*msiemens/cm**2*interneuron_size

VL2 = -65*mV
gNai = 35*msiemens/cm**2*interneuron_size
gKi = 9*msiemens/cm**2*interneuron_size
Esyn=-75*mV
taguisyn=2*ms
Iapp2=0*nA

net_loop=0 #0  no loop 1=loop
output_array=[]
'''this allows the simulaiton to be run multiple times to capture a range of randomized connectivity values'''
for i in range(1):
    output_array,net=main(output_array,i,net_loop)
