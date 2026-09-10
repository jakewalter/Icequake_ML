clear all;
clc;

load('depz.mat');
load('vs1d.mat');
load('vp1d.mat');
figure
plot(depz,vp1d,'r')
hold on
plot(depz,vs1d,'k')
xlim([0 5000])
xlabel('Depth (m)')
ylabel('Vel. (m/s)')
view(90,90)
set(gca,'fontsize',18)